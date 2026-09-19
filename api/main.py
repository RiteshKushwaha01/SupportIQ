import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routes.chat import router as chat_router


FRONTEND_URLS = os.getenv(
    "FRONTEND_URLS",
    "http://localhost:3000",
).split(",")

FRONTEND_URLS = [
    url.strip()
    for url in FRONTEND_URLS
    if url.strip()
]



app = FastAPI(
    title="SupportIQ API",
    description="AI-powered customer support backend",
    version="1.0.0",
)


# ---------------------------------------------------------
# CORS
# ---------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_URLS,
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
    return {
        "status": "healthy",
        "service": "supportiq-api",
    }