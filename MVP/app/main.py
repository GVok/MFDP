from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.router import api_router
import models

APP_TITLE = "MVP Prompt Enhancer / Image Judge"
IMAGES_DIR = Path("/data/images")

app = FastAPI(title=APP_TITLE)
app.include_router(api_router)

IMAGES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(IMAGES_DIR)), name="static")


@app.get("/health")
def health():
    return {"status": "ok"}
