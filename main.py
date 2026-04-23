from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import router
from app.config import CORS_ORIGINS

app = FastAPI(
    title="CLV Prediction and High-Value Identification API",
    description="Enterprise-ready API for CLV regression and premium-customer classification.",
    version="1.0.0",
)

if CORS_ORIGINS == ["*"]:
    allow_origins = ["*"]
else:
    allow_origins = CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/")
def root() -> dict:
    return {
        "message": "CLV platform backend is running.",
        "docs": "/docs",
        "health": "/health",
    }
