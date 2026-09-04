from pydantic import BaseModel, Field
from typing import Optional


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="The input text prompt.")
    max_tokens: int = Field(128, description="Maximum number of tokens to generate.")
    temperature: float = Field(0.7, description="Sampling temperature.")
    top_p: float = Field(0.95, description="Nucleus sampling probability.")
    top_k: int = Field(-1, description="Top-k sampling.")
    stream: bool = Field(
        True,
        description="Whether to stream the response back using Server-Sent Events.",
    )
    lora_id: Optional[str] = Field(None, description="The LoRA adapter ID to use.")
    guided_json: Optional[str] = Field(
        None, description="Optional JSON schema for guided decoding."
    )
    image_url: Optional[str] = Field(
        None, description="Optional Base64 or HTTP URL for Vision Language Models."
    )


class GenerateResponse(BaseModel):
    request_id: str
    text: str
    tokens_generated: int


class FeedbackRequest(BaseModel):
    request_id: str = Field(..., description="The ID of the generation request.")
    prompt: str = Field(..., description="The original prompt.")
    response: str = Field(..., description="The generated response.")
    rating: int = Field(..., description="1 for positive, -1 for negative.")


class IngestRequest(BaseModel):
    text: str = Field(..., description="The text content to ingest.")
    source: str = Field(..., description="The source or filename.")


class RAGGenerateRequest(GenerateRequest):
    pass  # Inherits prompt, max_tokens, etc.
