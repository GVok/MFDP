import os
import json
import time
import sys
import traceback
from typing import Any, Dict, Optional, List, Tuple
from datetime import datetime
from uuid import uuid4
from pathlib import Path
import re
from io import BytesIO

import pika
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import requests

sys.path.append("/app")

from models.enums import TaskStatus
from models.orm_ml_task import MLTaskEntity
from models.orm_prediction import PredictionEntity
from models.orm_ml_request import MLRequestEntity

import torch
import torch.nn as nn
from transformers import CLIPProcessor, CLIPModel, CLIPVisionModelWithProjection
from PIL import Image

load_dotenv()

RABBITMQ_URL = os.getenv("RABBITMQ_URL", "amqp://guest:guest@rabbitmq:5672/")
QUEUE_NAME = os.getenv("RABBITMQ_QUEUE", "ml_tasks_queue")
DATABASE_URL = os.getenv("DATABASE_URL")

IMAGES_DIR = os.getenv("IMAGES_DIR", "/data/images")

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
OLLAMA_KEEP_ALIVE = os.getenv("OLLAMA_KEEP_ALIVE", "30s")
OLLAMA_TIMEOUT_SEC = float(os.getenv("OLLAMA_TIMEOUT_SEC", "120"))
ENHANCE_MAX_TOKENS = int(os.getenv("ENHANCE_MAX_TOKENS", "200"))

HF_TOKEN = os.getenv("HF_TOKEN", "")
HF_T2I_MODEL = os.getenv("HF_T2I_MODEL", "stabilityai/sdxl-turbo")
HF_T2I_WIDTH = int(os.getenv("HF_T2I_WIDTH", "512"))
HF_T2I_HEIGHT = int(os.getenv("HF_T2I_HEIGHT", "512"))
HF_T2I_STEPS = int(os.getenv("HF_T2I_STEPS", "4"))
HF_T2I_GUIDANCE = float(os.getenv("HF_T2I_GUIDANCE", "0.0"))
HF_T2I_SEED = int(os.getenv("HF_T2I_SEED", "123"))
HF_TIMEOUT_SEC = float(os.getenv("HF_TIMEOUT_SEC", "120"))

CLIP_MODEL_NAME = os.getenv("CLIP_MODEL_NAME", "openai/clip-vit-base-patch32")
AES_WEIGHTS_URL = os.getenv(
    "AES_WEIGHTS_URL",
    "https://github.com/LAION-AI/aesthetic-predictor/raw/main/sa_0_4_vit_b_32_linear.pth",
)
AES_MIN = float(os.getenv("AES_MIN", "1.0"))
AES_MAX = float(os.getenv("AES_MAX", "10.0"))
FINAL_W_CLIP = float(os.getenv("FINAL_W_CLIP", "0.7"))
FINAL_W_AES = float(os.getenv("FINAL_W_AES", "0.3"))

BRAND_STYLE_SHORT = os.getenv(
    "BRAND_STYLE_SHORT",
    "modern minimalistic premium visual, soft purple gradients, "
    "deep violet shadows, clean composition, "
    "digital glossy accents, high-end commercial aesthetic",
)

SYSTEM_PROMPT = f"""
You are a Prompt Enhancer for an image generation system.

Your job:
- Take short, messy, mixed-language user prompts (mostly Russian).
- Normalize them into clean, detailed English prompts for image generation.
- Always follow the given brand visual style.

Default brand style:
{BRAND_STYLE_SHORT}

General rules:
1. Output ONLY the final enhanced prompt (no explanations, no comments).
2. Always respond in English.
3. Keep the original user intent (platform, goal, mood) but:
   - make the visual more premium, modern and minimalistic,
   - integrate the brand style (soft purple gradients, clean composition, premium digital aesthetic).
4. If platform/goal are given, adapt composition accordingly:
   - instagram_story / reels: vertical, strong focal point, works with overlaid text.
   - banner / website_visual: more space for text, clear hierarchy.
   - product_shot: focus on product, clean background.
   - promotion / sale: strong typography, high contrast, clear call-to-action space.
5. DO NOT mention 'brand style' or color codes explicitly in the prompt.
   Instead, implicitly apply them as visual characteristics.
""".strip()

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set in .env")

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def connect_with_retry(params: pika.URLParameters, retries: int = 10, delay: int = 5) -> pika.BlockingConnection:
    for attempt in range(1, retries + 1):
        try:
            print(f"Attempting to connect to RabbitMQ ({attempt}/{retries})...")
            return pika.BlockingConnection(params)
        except pika.exceptions.AMQPConnectionError as e:
            print(f"Connection failed: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)
    raise RuntimeError("Failed to connect to RabbitMQ after multiple retries.")


def _strip_emoji(text: str) -> str:
    try:
        import emoji
        return emoji.replace_emoji(text, replace="")
    except Exception:
        return text


def clean_text(prompt: str) -> str:
    p = str(prompt).strip()
    p = _strip_emoji(p)
    p = re.sub(r"<.*?>", "", p)
    p = re.sub(r"[^a-zA-Zа-яА-Я0-9ёЁ ,.!?%+\-]", " ", p)
    p = re.sub(r"([.!?,])\1+", r"\1", p)
    p = re.sub(r"\s+", " ", p).strip()
    p = p.lower()
    return p


def detect_platform(prompt: str) -> str:
    p = prompt.lower()
    if "сторис" in p or "stories" in p or "story" in p:
        return "instagram_story"
    if "reels" in p or "рилс" in p:
        return "instagram_reels"
    if "инста" in p or "instagram" in p:
        return "instagram_post"
    if "тикток" in p or "tiktok" in p or "тт" in p:
        return "tiktok_video"
    if "ютуб" in p or "youtube" in p:
        if "обложк" in p or "thumbnail" in p:
            return "youtube_thumbnail"
        return "youtube_video"
    if "pinterest" in p or "пинтерест" in p:
        return "pinterest_pin"
    if "вк" in p or "vk" in p or "вконтакт" in p:
        return "vk_post"
    if "тг" in p or "телеграм" in p or "telegram" in p:
        return "telegram_post"
    if "баннер" in p or "банер" in p or "banner" in p:
        return "banner"
    if "сайт" in p or "лендинг" in p or "landing" in p:
        return "website_visual"
    if "обложка" in p:
        return "cover_art"
    return "generic"


def detect_goal(prompt: str) -> str:
    p = prompt.lower()
    if any(word in p for word in ["реклама", "промо", "акция", "sale", "скидк"]):
        return "promotion"
    if any(word in p for word in ["бренд", "фирмен", "стиль", "айдентика"]):
        return "branding"
    if any(word in p for word in ["товар", "продукт", "предмет", "каталог", "ecom", "e-commerce"]):
        return "product_shot"
    if any(word in p for word in ["фон", "background", "подложка"]):
        return "background"
    if any(word in p for word in ["пост", "картинка", "визуал", "креатив"]):
        return "general_visual"
    if any(word in p for word in ["beauty", "косметика", "макияж", "скинкеар", "skincare"]):
        return "beauty"
    if any(word in p for word in ["фэшн", "fashion", "одежда", "лук", "стиль"]):
        return "fashion"
    if any(word in p for word in ["технолог", "tech", "стартап", "startup"]):
        return "tech_visual"
    if any(word in p for word in ["новогод", "праздник", "holiday", "зимн", "летн"]):
        return "seasonal"
    if any(word in p for word in ["еда", "food", "ресторан", "доставка"]):
        return "food_and_beverage"
    return "general_visual"


def build_user_content(
    raw_prompt: str,
    cleaned_prompt: Optional[str] = None,
    platform: Optional[str] = None,
    goal: Optional[str] = None,
    brand_style: Optional[str] = None,
) -> str:
    cleaned_prompt = cleaned_prompt or raw_prompt
    brand_style = brand_style or BRAND_STYLE_SHORT
    return (
        "You will receive a short user prompt in Russian (sometimes slangy) "
        "and some structured metadata. Rewrite it as a single, detailed, "
        "English image-generation prompt.\n\n"
        f"Raw prompt (ru): {raw_prompt}\n"
        f"Cleaned prompt (ru): {cleaned_prompt}\n"
        f"Detected platform: {platform}\n"
        f"Detected goal: {goal}\n"
        f"Brand style description: {brand_style}\n\n"
        "Return ONLY the enhanced English prompt, no explanations."
    )


def _ollama_try_post(path: str, payload: Dict[str, Any]) -> requests.Response:
    url = f"{OLLAMA_URL}{path}"
    return requests.post(url, json=payload, timeout=OLLAMA_TIMEOUT_SEC)


def ollama_enhance_prompt(raw_prompt: str, cleaned_prompt: str, platform: str, goal: str) -> str:
    user_content = build_user_content(
        raw_prompt=raw_prompt,
        cleaned_prompt=cleaned_prompt,
        platform=platform,
        goal=goal,
        brand_style=BRAND_STYLE_SHORT,
    )

    last_err: Exception | None = None

    chat_payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": ENHANCE_MAX_TOKENS},
    }

    for attempt in range(1, 4):
        try:
            r = _ollama_try_post("/api/chat", chat_payload)
            if r.status_code == 404:
                break
            r.raise_for_status()
            data = r.json()
            content = ((data.get("message") or {}).get("content") or "").strip()
            if not content:
                raise RuntimeError("Ollama /api/chat returned empty content")
            return content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    gen_payload = {
        "model": OLLAMA_MODEL,
        "prompt": f"{SYSTEM_PROMPT}\n\n{user_content}\n",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": ENHANCE_MAX_TOKENS},
    }

    for attempt in range(1, 4):
        try:
            r = _ollama_try_post("/api/generate", gen_payload)
            if r.status_code == 404:
                break
            r.raise_for_status()
            data = r.json()
            content = (data.get("response") or "").strip()
            if not content:
                raise RuntimeError("Ollama /api/generate returned empty response")
            return content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    v1_payload = {
        "model": OLLAMA_MODEL,
        "stream": False,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": ENHANCE_MAX_TOKENS,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    for attempt in range(1, 4):
        try:
            r = _ollama_try_post("/v1/chat/completions", v1_payload)
            if r.status_code == 404:
                break
            r.raise_for_status()
            data = r.json()
            choices = data.get("choices") or []
            msg = (choices[0].get("message") or {}) if choices else {}
            content = (msg.get("content") or "").strip()
            if not content:
                raise RuntimeError("Ollama /v1/chat/completions returned empty content")
            return content
        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(
        f"Ollama enhance failed. Tried /api/chat, /api/generate, /v1/chat/completions. Last error: {last_err}"
    )


def hf_generate_image_bytes(
    prompt: str,
    *,
    model: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    timeout_sec: float,
) -> bytes:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set (but image_backend=hf was requested)")

    url = f"https://api-inference.huggingface.co/models/{model}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}

    payload = {
        "inputs": prompt,
        "parameters": {
            "width": width,
            "height": height,
            "num_inference_steps": steps,
            "guidance_scale": guidance,
            "seed": seed,
        },
        "options": {"wait_for_model": True},
    }

    last_err: Exception | None = None
    for attempt in range(1, 6):
        try:
            r = requests.post(url, headers=headers, json=payload, timeout=timeout_sec)
            if r.status_code == 503:
                time.sleep(2.0 * attempt)
                continue

            r.raise_for_status()

            ct = (r.headers.get("content-type") or "").lower()
            if "image" not in ct:
                try:
                    j = r.json()
                    raise RuntimeError(f"HF returned non-image response: {j}")
                except Exception:
                    raise RuntimeError(f"HF returned non-image response, content-type={ct}, body={r.text[:300]}")
            return r.content

        except Exception as e:
            last_err = e
            time.sleep(1.5 * attempt)

    raise RuntimeError(f"HF generation failed after retries: {last_err}")


def hf_image_bytes_to_png_bytes(img_bytes: bytes) -> bytes:
    img = Image.open(BytesIO(img_bytes))

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode != "RGB":
        img = img.convert("RGB")

    out = BytesIO()
    img.save(out, format="PNG", optimize=True)
    return out.getvalue()


def save_hf_image_as_png(abs_path: Path, img_bytes: bytes) -> None:
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_bytes(hf_image_bytes_to_png_bytes(img_bytes))


def write_mock_image(path: Path, idx: int, width: int = 512, height: int = 512) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height))
    pixels = img.load()
    for y in range(height):
        for x in range(width):
            r = (x + 17 * idx) % 256
            g = (y + 31 * idx) % 256
            b = (x + y + 47 * idx) % 256
            pixels[x, y] = (r, g, b)
    img.save(path, format="PNG")


_CLIP_MODEL: Optional[CLIPModel] = None
_CLIP_PROC: Optional[CLIPProcessor] = None
_AES_VISION: Optional[CLIPVisionModelWithProjection] = None
_AES_HEAD: Optional[nn.Linear] = None


def _get_device() -> torch.device:
    return torch.device("cpu")


def load_judge_models() -> None:
    global _CLIP_MODEL, _CLIP_PROC, _AES_VISION, _AES_HEAD
    if _CLIP_MODEL is not None and _CLIP_PROC is not None and _AES_VISION is not None and _AES_HEAD is not None:
        return

    device = _get_device()
    print(f"[judge] loading CLIP: {CLIP_MODEL_NAME} on {device} ...")

    _CLIP_PROC = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    _CLIP_MODEL = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    _CLIP_MODEL.to(device).eval()

    print(f"[judge] loading Aesthetic head weights from: {AES_WEIGHTS_URL}")
    _AES_VISION = CLIPVisionModelWithProjection.from_pretrained(CLIP_MODEL_NAME)
    _AES_VISION.to(device).eval()

    _AES_HEAD = nn.Linear(_AES_VISION.config.projection_dim, 1)
    state_dict = torch.hub.load_state_dict_from_url(AES_WEIGHTS_URL, map_location="cpu")
    _AES_HEAD.load_state_dict(state_dict)
    _AES_HEAD.to(device).eval()


def clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def normalize_aesthetic(aes: float) -> float:
    if AES_MAX <= AES_MIN:
        return 0.0
    return clamp01((aes - AES_MIN) / (AES_MAX - AES_MIN))


@torch.no_grad()
def compute_clip_similarity_01(image: Image.Image, text: str) -> float:
    assert _CLIP_MODEL and _CLIP_PROC
    inputs = _CLIP_PROC(text=[text], images=[image], return_tensors="pt", padding=True)
    inputs = {k: v.to(_get_device()) for k, v in inputs.items()}
    out = _CLIP_MODEL(**inputs)

    img = out.image_embeds
    txt = out.text_embeds
    img = img / img.norm(dim=-1, keepdim=True)
    txt = txt / txt.norm(dim=-1, keepdim=True)

    sim = (img * txt).sum(dim=-1).item()
    return clamp01((sim + 1.0) / 2.0)


@torch.no_grad()
def compute_aesthetic_score(image: Image.Image) -> float:
    assert _AES_VISION and _AES_HEAD and _CLIP_PROC
    inputs = _CLIP_PROC(images=[image], return_tensors="pt")
    pixel_values = inputs["pixel_values"].to(_get_device())

    out = _AES_VISION(pixel_values=pixel_values)
    img_emb = out.image_embeds
    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)

    pred = _AES_HEAD(img_emb).squeeze(-1).item()
    return float(pred)


def final_score(clip_n: float, aes_n: float) -> float:
    s = FINAL_W_CLIP * clip_n + FINAL_W_AES * aes_n
    return clamp01(float(s))


def _pick_backends(payload: Dict[str, Any]) -> tuple[str, str, int]:
    enhance_backend = (payload.get("enhance_backend") or "").lower().strip()
    image_backend = (payload.get("image_backend") or "").lower().strip()
    n_images = int(payload.get("n_images") or 1)

    if not enhance_backend:
        mode = (payload.get("mode") or "").lower().strip()
        enhance_backend = "mock" if mode == "mock" else "ollama"

    if image_backend not in ("hf", "mock"):
        image_backend = "mock"
    if enhance_backend not in ("ollama", "mock"):
        enhance_backend = "ollama"

    if image_backend == "hf":
        n_images = 1
    else:
        n_images = max(1, min(n_images, 4))

    return enhance_backend, image_backend, n_images


def run_inference(payload: Dict[str, Any]) -> Dict[str, Any]:
    raw_prompt = payload.get("raw_prompt", "")
    enhance_backend, image_backend, n_images = _pick_backends(payload)

    cleaned = clean_text(raw_prompt)
    platform = detect_platform(cleaned)
    goal = detect_goal(cleaned)

    enhanced: Optional[str] = None
    if enhance_backend == "ollama":
        enhanced = ollama_enhance_prompt(
            raw_prompt=raw_prompt,
            cleaned_prompt=cleaned,
            platform=platform,
            goal=goal,
        )

    prompt_for_image = enhanced or cleaned or raw_prompt

    return {
        "raw_prompt": raw_prompt,
        "cleaned_prompt": cleaned,
        "platform": platform,
        "goal": goal,
        "enhanced_prompt": enhanced,
        "enhance_backend": enhance_backend,
        "image_backend": image_backend,
        "n_images": n_images,
        "seed": HF_T2I_SEED if image_backend == "hf" else None,
        "model": HF_T2I_MODEL if image_backend == "hf" else "mock_image",
        "steps": HF_T2I_STEPS if image_backend == "hf" else None,
        "width": HF_T2I_WIDTH,
        "height": HF_T2I_HEIGHT,
        "guidance": HF_T2I_GUIDANCE if image_backend == "hf" else None,
        "ollama_model": OLLAMA_MODEL,
        "ollama_keep_alive": OLLAMA_KEEP_ALIVE,
        "prompt_for_image": prompt_for_image,
    }


def _should_requeue(exc: Exception) -> bool:
    msg = str(exc).lower()
    if "404" in msg or "401" in msg or "not found" in msg:
        return False
    return True


def process_message(ch, method, properties, body: bytes):
    print(f"Received task message: {body!r}")

    try:
        payload = json.loads(body)
        task_id = int(payload["task_id"])
    except Exception:
        print("Invalid message format. Expected JSON with task_id.")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        return

    db = SessionLocal()
    try:
        task: MLTaskEntity | None = db.query(MLTaskEntity).filter(MLTaskEntity.id == task_id).first()
        if not task:
            print(f"Task {task_id} not found. Ack and drop.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        if task.status == TaskStatus.SUCCESS.value:
            print(f"Task {task_id} already SUCCESS. Ack.")
            ch.basic_ack(delivery_tag=method.delivery_tag)
            return

        task.status = TaskStatus.RUNNING.value
        task.error = None
        db.commit()

        gen_meta = run_inference(task.payload or {})

        req: MLRequestEntity | None = (
            db.query(MLRequestEntity)
            .filter(MLRequestEntity.id == task.request_id, MLRequestEntity.user_id == task.user_id)
            .first()
        )
        if req:
            req.cleaned_prompt = gen_meta.get("cleaned_prompt")
            req.enhanced_prompt = gen_meta.get("enhanced_prompt")
            req.status = TaskStatus.RUNNING.value

        images_dir = Path(IMAGES_DIR)
        images_dir.mkdir(parents=True, exist_ok=True)

        n_images = int(gen_meta.get("n_images") or 1)
        prompt_for_image = gen_meta.get("prompt_for_image") or ""

        generated: List[Tuple[str, Path]] = []

        for i in range(n_images):
            filename = f"{task.request_id}_{task.id}_{i}_{uuid4().hex}.png"
            abs_path = images_dir / filename
            public_path = f"/static/images/{filename}"

            if gen_meta.get("image_backend") == "hf":
                img_bytes = hf_generate_image_bytes(
                    prompt_for_image,
                    model=gen_meta.get("model") or HF_T2I_MODEL,
                    width=int(gen_meta.get("width") or HF_T2I_WIDTH),
                    height=int(gen_meta.get("height") or HF_T2I_HEIGHT),
                    steps=int(gen_meta.get("steps") or HF_T2I_STEPS),
                    guidance=float(gen_meta.get("guidance") or HF_T2I_GUIDANCE),
                    seed=int(gen_meta.get("seed") or HF_T2I_SEED),
                    timeout_sec=HF_TIMEOUT_SEC,
                )
                save_hf_image_as_png(abs_path, img_bytes)
            else:
                write_mock_image(
                    abs_path,
                    idx=i,
                    width=int(gen_meta.get("width") or 512),
                    height=int(gen_meta.get("height") or 512),
                )

            generated.append((public_path, abs_path))

        load_judge_models()

        scored: List[Dict[str, Any]] = []
        for pub_path, abs_path in generated:
            img = Image.open(abs_path).convert("RGB")
            clip_n = compute_clip_similarity_01(img, prompt_for_image)
            aes_raw = compute_aesthetic_score(img)
            aes_n = normalize_aesthetic(aes_raw)
            s = final_score(clip_n, aes_n)

            scored.append(
                {
                    "image_path": pub_path,
                    "image_abs_path": str(abs_path),
                    "clip_n": clip_n,
                    "aes_raw": aes_raw,
                    "aes_n": aes_n,
                    "final": s,
                }
            )

        scored.sort(key=lambda x: x["final"], reverse=True)
        for rank, item in enumerate(scored, start=1):
            item["rank"] = rank

        predictions_out = []
        for item in scored:
            pred = PredictionEntity(
                user_id=task.user_id,
                request_id=task.request_id,
                image_path=item["image_path"],
                clip_score=float(item["clip_n"]),
                aesthetic_score=float(item["aes_n"]),
                final_score=float(item["final"]),
                rank=int(item["rank"]),
                gen_meta={
                    **gen_meta,
                    "saved_at": datetime.utcnow().isoformat(),
                    "image_abs_path": item["image_abs_path"],
                    "provider": "huggingface_inference_api" if gen_meta.get("image_backend") == "hf" else "mock",
                    "judge": {
                        "clip_model": CLIP_MODEL_NAME,
                        "aes_weights_url": AES_WEIGHTS_URL,
                        "aes_min": AES_MIN,
                        "aes_max": AES_MAX,
                        "w_clip": FINAL_W_CLIP,
                        "w_aes": FINAL_W_AES,
                    },
                    "scores": {
                        "clip_n": item["clip_n"],
                        "aes_raw": item["aes_raw"],
                        "aes_n": item["aes_n"],
                        "final": item["final"],
                    },
                },
            )
            db.add(pred)
            predictions_out.append(
                {
                    "image_path": pred.image_path,
                    "rank": pred.rank,
                    "clip_score": pred.clip_score,
                    "aesthetic_score": pred.aesthetic_score,
                    "final_score": pred.final_score,
                }
            )

        best = scored[0] if scored else None

        task.result = {
            "best_image_path": best["image_path"] if best else None,
            "enhanced_prompt": gen_meta.get("enhanced_prompt"),
            "gen_meta": gen_meta,
            "predictions": predictions_out,
        }
        task.status = TaskStatus.SUCCESS.value
        if req:
            req.status = TaskStatus.SUCCESS.value

        db.commit()
        ch.basic_ack(delivery_tag=method.delivery_tag)
        print(f"Task #{task_id} processed successfully. Best={task.result.get('best_image_path')}")

    except Exception as e:
        db.rollback()
        print(f"Error processing task {task_id}: {e}")
        traceback.print_exc()

        try:
            task = db.query(MLTaskEntity).filter(MLTaskEntity.id == task_id).first()
            if task:
                task.status = TaskStatus.FAILED.value
                task.error = str(e)
                db.commit()
        except Exception:
            db.rollback()

        requeue = _should_requeue(e)
        ch.basic_nack(delivery_tag=method.delivery_tag, requeue=requeue)

    finally:
        db.close()


def main():
    params = pika.URLParameters(RABBITMQ_URL)
    connection = connect_with_retry(params)
    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=True)
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(queue=QUEUE_NAME, on_message_callback=process_message, auto_ack=False)

    print("Worker started. Waiting for ML tasks. CTRL+C to exit.")
    channel.start_consuming()


if __name__ == "__main__":
    main()
