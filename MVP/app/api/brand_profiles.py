from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from models.orm_user import UserEntity
from schemas.brand_profiles import BrandProfilesOut, BrandProfileOut, BrandProfileCreateIn
from services.brand_profile_service import list_profiles, create_profile, delete_profile

router = APIRouter(prefix="/brand-profiles", tags=["Brand Profiles"])


@router.get("", response_model=BrandProfilesOut)
def my_profiles(user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    profs = list_profiles(db, user.id)
    return BrandProfilesOut(profiles=profs)


@router.post("", response_model=BrandProfileOut, status_code=201)
def create(data: BrandProfileCreateIn, user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    try:
        prof = create_profile(db, user.id, data.name, data.style_short)
        return prof
    except ValueError as e:
        raise HTTPException(status_code=402, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create brand profile: {e}")


@router.delete("/{profile_id}", status_code=204)
def remove(profile_id: int, user: UserEntity = Depends(get_current_user), db: Session = Depends(get_db)):
    ok = delete_profile(db, user.id, profile_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Brand profile not found")
    return None
