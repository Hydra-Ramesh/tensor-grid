from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import router

app = FastAPI(
    title="Distributed LLM Inference Engine",
    description="High-performance, continuous batching LLM gateway built with vLLM and Ray",
    version="1.0.0",
    default_response_class=ORJSONResponse,  # V12: Rust-based JSON serialization for hyper-speed
)

# Allow the Next.js frontend to talk to this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the generation routes
app.include_router(router)


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}
