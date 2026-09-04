from fastapi import APIRouter, Depends
from src.api.schemas import (
    GenerateRequest,
    IngestRequest,
    RAGGenerateRequest,
    FeedbackRequest,
)
from src.api.dependencies import (
    verify_api_key,
    get_generation_pipeline,
    get_rag_pipeline,
    get_o1_pipeline,
)
from src.core.use_cases.generation_pipeline import GenerationPipeline
from src.core.use_cases.rag_pipeline import RAGPipeline
from src.core.use_cases.o1_reasoning_pipeline import O1ReasoningPipeline
from src.infrastructure.databases.feedback_redis import save_feedback
from src.infrastructure.databases.vector_qdrant import ingest_document
from src.utils.tracing import tracer
from src.api.middleware import check_rate_limit

router = APIRouter()


@router.post("/o1_generate")
async def o1_generate_endpoint(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key),
    pipeline: O1ReasoningPipeline = Depends(get_o1_pipeline),
):
    """
    Test-Time Compute Endpoint (o1-style).
    Uses heavy reasoning chains before answering. Expect high latency but PhD-level accuracy.
    """
    with tracer.start_as_current_span("o1_generate_endpoint"):
        await check_rate_limit(api_key)
        return await pipeline.process_reasoning(request)


@router.post("/generate")
async def generate_endpoint(
    request: GenerateRequest,
    api_key: str = Depends(verify_api_key),
    pipeline: GenerationPipeline = Depends(get_generation_pipeline),
):
    """
    Endpoint to generate text using the LLM engine.
    Business logic is fully abstracted into the injected GenerationPipeline.
    """
    with tracer.start_as_current_span("generate_endpoint"):
        # 1. API Gateway Edge Security (Rate Limiting)
        with tracer.start_as_current_span("rate_limit_check"):
            await check_rate_limit(api_key)

        # 2. Delegate to Domain Service Layer
        return await pipeline.process_generate(request)


@router.post("/rag_generate")
async def rag_generate_endpoint(
    request: RAGGenerateRequest,
    api_key: str = Depends(verify_api_key),
    pipeline: RAGPipeline = Depends(get_rag_pipeline),
):
    """Retrieval-Augmented Generation endpoint."""
    with tracer.start_as_current_span("rag_generate_endpoint"):
        with tracer.start_as_current_span("rate_limit_check"):
            await check_rate_limit(api_key)

        return await pipeline.process_rag(request)


@router.post("/ingest")
async def ingest_endpoint(
    request: IngestRequest, api_key: str = Depends(verify_api_key)
):
    """Ingests a document into the Qdrant Vector Database."""
    with tracer.start_as_current_span("ingest_document"):
        chunks_added = await ingest_document(request.text, request.source)
        return {"status": "success", "chunks_added": chunks_added}


@router.post("/feedback")
async def feedback_endpoint(
    request: FeedbackRequest, api_key: str = Depends(verify_api_key)
):
    """Logs user feedback for continuous DPO/RLHF training."""
    with tracer.start_as_current_span("feedback_endpoint"):
        save_feedback(
            request.request_id, request.prompt, request.response, request.rating
        )
        return {"status": "success", "message": "Feedback recorded for DPO flywheel."}
