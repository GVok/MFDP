from fastapi import APIRouter, Request, Depends
from web.deps import get_user_from_cookie
from models.orm_user import UserEntity

router = APIRouter()

@router.get("/")
def home(request: Request, user: UserEntity | None = Depends(get_user_from_cookie)):
    tpl = request.app.state.templates
    return tpl.TemplateResponse("home.html", {"request": request, "user": user})