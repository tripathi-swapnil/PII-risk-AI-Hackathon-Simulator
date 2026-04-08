from fastapi import FastAPI
from dotenv import load_dotenv

from api.routes import router


load_dotenv()

app = FastAPI(title="SafePII-RL")
app.include_router(router)
