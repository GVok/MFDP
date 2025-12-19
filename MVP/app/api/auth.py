from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from api.deps import get_db, get_current_user
from schemas.auth import RegisterIn, LoginIn, TokenOut, UserOut
from models.orm_user import UserEntity
from models.orm_wallet import WalletEntity
from services.auth_service import hash_password, verify_password, create_access_token


router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/register", status_code=201)
def register(data: RegisterIn, db: Session = Depends(get_db)):
    if db.query(UserEntity).filter(UserEntity.username == data.username).first():
        raise HTTPException(status_code=400, detail="Username already exists")
    if db.query(UserEntity).filter(UserEntity.email == data.email).first():
        raise HTTPException(status_code=400, detail="Email already exists")

    user = UserEntity(
        username=data.username,
        email=data.email,
        password_hash=hash_password(data.password),
        is_admin=False,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    wallet = WalletEntity(user_id=user.id, balance_rub=0)
    db.add(wallet)
    db.commit()

    return {"message": "Registered"}


@router.post("/login", response_model=TokenOut)
def login(data: LoginIn, db: Session = Depends(get_db)):
    user = db.query(UserEntity).filter(UserEntity.username == data.username, UserEntity.is_active.is_(True)).first()
    if not user or not verify_password(data.password, user.password_hash):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    token = create_access_token(user_id=user.id)
    return TokenOut(access_token=token)


@router.get("/me", response_model=UserOut)
def me(user: UserEntity = Depends(get_current_user)):
    return user
