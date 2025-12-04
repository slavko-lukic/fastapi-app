from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.api.routes import chat


BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


app = FastAPI(title="Chatbot API")

app.include_router(chat.router)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
async def chat_ui(request: Request) -> HTMLResponse:
    """
    Simple browser UI for the chatbot.
    """
    return templates.TemplateResponse("chat.html", {"request": request})
