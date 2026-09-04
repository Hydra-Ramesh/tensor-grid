from fastapi import Header, HTTPException
from src.core.config import settings
from src.core.use_cases.tool_executor import ToolRegistry
from src.core.use_cases.generation_pipeline import GenerationPipeline
from src.core.use_cases.rag_pipeline import RAGPipeline
from src.core.use_cases.o1_reasoning_pipeline import O1ReasoningPipeline

# Singletons for dependency injection
_tool_registry = ToolRegistry()
_generation_pipeline = GenerationPipeline(_tool_registry)
_rag_pipeline = RAGPipeline(_generation_pipeline)
_o1_pipeline = O1ReasoningPipeline()


def verify_api_key(x_api_key: str = Header(...)):
    """Validates the API key."""
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key


def get_generation_pipeline() -> GenerationPipeline:
    """Dependency injection provider for the GenerationPipeline."""
    return _generation_pipeline


def get_rag_pipeline() -> RAGPipeline:
    """Dependency injection provider for the RAGPipeline."""
    return _rag_pipeline


def get_o1_pipeline() -> O1ReasoningPipeline:
    """Dependency injection provider for the O1ReasoningPipeline."""
    return _o1_pipeline
