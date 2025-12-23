from fastapi import APIRouter

from web.pages.auth import router as auth_pages
from web.pages.home import router as home_pages
from web.pages.account import router as account_pages
from web.pages.app import router as app_pages

web_router = APIRouter()
web_router.include_router(home_pages)
web_router.include_router(auth_pages)
web_router.include_router(account_pages)
web_router.include_router(app_pages)