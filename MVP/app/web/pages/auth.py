from fastapi import APIRouter, Request, Depends, Form
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from api.deps import get_db
from models.orm_user import UserEntity
from models.orm_wallet import WalletEntity
from services.auth_service import verify_password, hash_password, create_access_token

router = APIRouter()

@router.get("/login")
def login_page(request: Request):
    tpl = request.app.state.templates
    return tpl.TemplateResponse("auth/login.html", {"request": request, "user": None})

@router.post("/login")
def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    tpl = request.app.state.templates

    user = db.query(UserEntity).filter(UserEntity.username == username, UserEntity.is_active.is_(True)).first()
    if not user or not verify_password(password, user.password_hash):
        return tpl.TemplateResponse(
            "auth/login.html",
            {"request": request, "user": None, "error": "Invalid username or password"},
            status_code=401,
        )

    token = create_access_token(user_id=user.id)
    response = RedirectResponse(url="/app", status_code=303)
    response.set_cookie(key="access_token", value=token, httponly=True, samesite="lax")
    return response

@router.get("/register")
def register_page(request: Request):
    tpl = request.app.state.templates
    return tpl.TemplateResponse("auth/register.html", {"request": request, "user": None})

@router.post("/register")
def register(
    request: Request,
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    tpl = request.app.state.templates

    if db.query(UserEntity).filter(UserEntity.username == username).first():
        return tpl.TemplateResponse("auth/register.html", {"request": request, "user": None, "error": "Username already exists"}, status_code=400)
    if db.query(UserEntity).filter(UserEntity.email == email).first():
        return tpl.TemplateResponse("auth/register.html", {"request": request, "user": None, "error": "Email already exists"}, status_code=400)

    user = UserEntity(username=username, email=email, password_hash=hash_password(password), is_active=True, is_admin=False)
    db.add(user)
    db.commit()
    db.refresh(user)

    wallet = WalletEntity(user_id=user.id, balance_rub=0)
    db.add(wallet)
    db.commit()

    return RedirectResponse(url="/login", status_code=303)

@router.get("/logout")
def logout():
    response = RedirectResponse(url="/login", status_code=303)
    response.delete_cookie("access_token")
    return response
