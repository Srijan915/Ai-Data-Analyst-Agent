"""FastAPI application factory and middleware configuration."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.router import router

app = FastAPI(
    title="QueryPilotAI API",
    description="Secure REST API for QueryPilotAI, DuckDB SQL execution, and relationship discovery.",
    version="2.0.0",
)

# Compatibility API; the user interface runs in Streamlit.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)

