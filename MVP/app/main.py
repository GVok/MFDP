from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from api.router import api_router
from web.router import web_router

APP_TITLE = "MVP Prompt Enhancer / Image Judge"
IMAGES_DIR = Path("/data/images")

BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title=APP_TITLE)

templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.state.templates = templates

app.include_router(api_router, prefix="/api")
app.include_router(web_router)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

app.mount("/media", StaticFiles(directory=str(IMAGES_DIR)), name="media")


@app.get("/health")
def health():
    return {"status": "ok"}
