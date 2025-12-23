from sqlalchemy.orm import Session

from models.orm_brand_profile import BrandProfileEntity
from services.subscription_service import get_active_plan


def list_profiles(db: Session, user_id: int) -> list[BrandProfileEntity]:
    return (
        db.query(BrandProfileEntity)
        .filter(BrandProfileEntity.user_id == user_id)
        .order_by(BrandProfileEntity.id.desc())
        .all()
    )


def get_profile(db: Session, user_id: int, profile_id: int) -> BrandProfileEntity | None:
    return (
        db.query(BrandProfileEntity)
        .filter(BrandProfileEntity.user_id == user_id, BrandProfileEntity.id == profile_id)
        .first()
    )


def create_profile(db: Session, user_id: int, name: str, style_short: str) -> BrandProfileEntity:
    plan = get_active_plan(db, user_id)

    existing = db.query(BrandProfileEntity).filter(BrandProfileEntity.user_id == user_id).count()
    if plan.brand_profiles_limit <= 0 or existing >= plan.brand_profiles_limit:
        raise ValueError("Brand profiles are not available on your plan")

    prof = BrandProfileEntity(user_id=user_id, name=name, style_short=style_short or "")
    db.add(prof)
    db.commit()
    db.refresh(prof)
    return prof


def delete_profile(db: Session, user_id: int, profile_id: int) -> bool:
    prof = get_profile(db, user_id, profile_id)
    if not prof:
        return False
    db.delete(prof)
    db.commit()
    return True


def build_style_short_from_form(
    *,
    mood: str = "",
    category: str = "",
    lighting: str = "",
    color_vibe: str = "",
    composition: str = "",
    materials: list[str] | None = None,
    avoid: list[str] | None = None,
    no_color_codes: bool = True,
    max_len: int = 350,
) -> str:
    materials = materials or []
    avoid = avoid or []

    MOOD = {
        "minimal": "minimal",
        "premium": "premium",
        "playful": "playful",
        "brutal": "brutal",
        "clean": "clean",
        "elegant": "elegant",
        "futuristic": "futuristic",
        "cozy": "cozy",
    }
    CATEGORY = {
        "food": "food photography",
        "beauty": "beauty cosmetic aesthetic",
        "fashion": "fashion editorial",
        "tech": "modern tech visual",
        "real_estate": "real estate showcase",
        "fitness": "fitness campaign",
        "restaurant": "restaurant promo",
        "ecommerce": "e-commerce product visual",
        "education": "education promo visual",
        "travel": "travel lifestyle",
    }
    LIGHTING = {
        "soft_studio": "soft studio light",
        "natural_daylight": "natural daylight",
        "cinematic": "cinematic lighting",
        "high_key": "high key lighting",
        "low_key": "low key lighting",
        "neon_glow": "subtle neon glow",
    }
    COLOR = {
        "purple_gradient": "soft purple gradient vibe",
        "pastel": "pastel color vibe",
        "monochrome": "monochrome palette",
        "warm": "warm palette",
        "cold": "cool palette",
        "earthy": "earthy tones",
        "vibrant": "vibrant accents",
        "neutral": "neutral palette",
    }
    COMPOSITION = {
        "centered": "centered composition",
        "rule_of_thirds": "rule of thirds composition",
        "lots_of_copy_space": "lots of copy space",
        "product_focus": "product-focused framing",
        "close_up": "close-up framing",
        "wide_scene": "wide scene composition",
    }
    MATERIALS = {
        "glossy": "glossy",
        "matte": "matte",
        "metallic": "metallic",
        "glass": "glass",
        "paper": "paper",
        "plastic": "plastic",
        "fabric": "fabric",
        "ceramic": "ceramic",
    }
    AVOID = {
        "clutter": "clutter",
        "text": "text",
        "faces": "faces",
        "hands": "hands",
        "logos": "logos",
        "watermark": "watermark",
        "busy_background": "busy background",
    }

    parts: list[str] = []

    m = MOOD.get((mood or "").strip(), "")
    if m:
        parts.append(m)

    c = CATEGORY.get((category or "").strip(), "")
    if c:
        parts.append(c)

    l = LIGHTING.get((lighting or "").strip(), "")
    if l:
        parts.append(l)

    cv = COLOR.get((color_vibe or "").strip(), "")
    if cv:
        parts.append(cv)

    comp = COMPOSITION.get((composition or "").strip(), "")
    if comp:
        parts.append(comp)

    mats = [MATERIALS[x] for x in materials if x in MATERIALS]
    if mats:
        parts.append("materials: " + " + ".join(mats))

    av = [AVOID[x] for x in avoid if x in AVOID]
    if av:
        parts.append("avoid " + " + ".join(av))

    if no_color_codes:
        parts.append("no explicit color codes")

    style = ", ".join([p for p in parts if p]).strip()
    if len(style) > max_len:
        style = style[:max_len].rstrip(" ,")
    return style
