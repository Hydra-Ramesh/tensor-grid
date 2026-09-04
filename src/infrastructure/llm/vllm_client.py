from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm import SamplingParams
from src.core.config import settings
from typing import AsyncGenerator
from src.api.schemas import GenerateRequest


import json


async def stream_response(
    request: GenerateRequest, request_id: str, engine: AsyncLLMEngine
) -> AsyncGenerator[str, None]:

    # V11: Intercept logits at the GPU layer to enforce absolute JSON validity (XGrammar)
    sampling_params = SamplingParams(
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        guided_json=request.guided_json,
    )

    results_generator = engine.generate(request.prompt, sampling_params, request_id)
    async for request_output in results_generator:
        text = request_output.outputs[0].text
        yield f"data: {json.dumps({'text': text})}\n\n"


def get_engine() -> AsyncLLMEngine:
    """
    Initializes and returns the vLLM asynchronous engine.
    """
    engine_args = AsyncEngineArgs(
        model=settings.model_name,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        max_model_len=settings.max_model_len,
        tensor_parallel_size=settings.tensor_parallel_size,
        pipeline_parallel_size=2,  # V13: Enables pipeline parallelism across nodes
        distributed_executor_backend="ray",  # V13: Required for Ring Attention distributed workers
        enable_lora=settings.enable_lora,
        max_loras=settings.max_loras,
        max_lora_rank=settings.max_lora_rank,
        enable_prefix_caching=settings.enable_prefix_caching,
        enable_chunked_prefill=settings.enable_chunked_prefill,
        speculative_model="[medusa-checkpoint-id]",  # V14: Upgraded from draft model to Medusa tree-attention heads
        num_speculative_tokens=5,  # V14: Predict 5 tokens simultaneously per forward pass
        use_v2_block_manager=True,  # V14: Required for Medusa Tree-Attention routing
        kv_cache_dtype=settings.kv_cache_dtype,
        swap_space=settings.swap_space,
        rope_scaling={"type": "dynamic", "factor": settings.rope_scaling_factor},
        disable_log_requests=True,  # Disable standard logging to reduce overhead
        trust_remote_code=True,
    )

    print(f"Initializing vLLM Engine with model: {settings.model_name}")
    print(f"GPU Utilization config: {settings.gpu_memory_utilization}")

    engine = AsyncLLMEngine.from_engine_args(engine_args)
    return engine


# Global instance initialized lazily
_engine = None


def get_engine_instance() -> AsyncLLMEngine:
    global _engine
    if _engine is None:
        _engine = get_engine()
    return _engine
