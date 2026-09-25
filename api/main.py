import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import get_pipeline, is_pipeline_ready, router as chat_router


def _parse_frontend_urls():
    raw = os.getenv("FRONTEND_URLS", "http://localhost:3000")
    return [
        url.strip().rstrip("/")
        for url in raw.split(",")
        if url.strip()
    ]


WARMUP_ON_START = os.getenv("WARMUP_ON_START", "false").lower() == "true"
FRONTEND_URLS = _parse_frontend_urls()
FRONTEND_ORIGIN_REGEX = os.getenv("FRONTEND_ORIGIN_REGEX", "").strip() or None


@asynccontextmanager
async def lifespan(app: FastAPI):
    if WARMUP_ON_START:
        print("LIFESPAN: Warming RAG pipeline...", flush=True)
        get_pipeline()
        print("LIFESPAN: Warmup complete.", flush=True)
    yield


app = FastAPI(
    title="SupportIQ API",
    description="AI-powered customer support backend",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
    allow_origin_regex=FRONTEND_ORIGIN_REGEX,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------
# Routes
# ---------------------------------------------------------

app.include_router(
    chat_router,
    prefix="/api",
)


# ---------------------------------------------------------
# Health check
# ---------------------------------------------------------

@app.get("/api/health")
def health_check():
    ready = is_pipeline_ready()
    return {
        "status": "healthy",
        "service": "supportiq-api",
        "ready": ready,
        "warmup_on_start": WARMUP_ON_START,
    }