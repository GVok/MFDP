from fastapi import APIRouter

from api.auth import router as auth_router
from api.wallet import router as wallet_router
from api.ml import router as ml_router
from api.admin import router as admin_router
from api.billing import router as billing_router
from api.brand_profiles import router as brand_profiles_router

api_router = APIRouter()
api_router.include_router(auth_router)
api_router.include_router(wallet_router)
api_router.include_router(ml_router)
api_router.include_router(admin_router)
api_router.include_router(billing_router)
api_router.include_router(brand_profiles_router)