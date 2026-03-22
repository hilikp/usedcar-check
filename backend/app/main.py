from fastapi import FastAPI

from app.api.routes import router
from app.db.init_db import init_db


app = FastAPI(title="Used Car Check API", version="0.1.0")
app.include_router(router)


@app.on_event("startup")
def _startup():
    init_db()

