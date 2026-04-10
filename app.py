from pathlib import Path

from fastapi import FastAPI
from dotenv import load_dotenv
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from api.routes import router


load_dotenv()

app = FastAPI(title="SafePII-RL")
app.include_router(router)

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"

if FRONTEND_DIR.exists():
	app.mount("/frontend", StaticFiles(directory=FRONTEND_DIR), name="frontend")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
	return FileResponse(FRONTEND_DIR / "index.html")
