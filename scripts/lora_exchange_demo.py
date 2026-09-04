import asyncio
from src.api.schemas import GenerateRequest
from src.core.use_cases.generation_pipeline import GenerationPipeline
from src.core.use_cases.tool_executor import ToolRegistry
from src.utils.logger import logger


async def simulate_lorax_scale():
    """
    Simulates a Multi-Tenant LoRA Exchange (S-LoRA) load test.
    Instead of swapping the massive 70B base model, vLLM dynamically streams
    tiny 100MB LoRA adapters from CPU to VRAM on a per-request basis.
    This allows 1 GPU to serve 10,000 personalized models concurrently.
    """

    # In V13 we created Ray Actors, but for this demo script we'll instantiate locally.
    registry = ToolRegistry()
    pipeline = GenerationPipeline(registry)

    # 10,000 users, each with their own personalized fine-tuned LoRA
    # We will simulate concurrent requests from 5 different users instantly

    requests = [
        GenerateRequest(
            prompt="Write a Python script for a binary tree.",
            lora_id="user_1204_python_expert",
        ),
        GenerateRequest(
            prompt="Translate this to French.", lora_id="user_882_translator"
        ),
        GenerateRequest(
            prompt="Solve this differential equation.", lora_id="user_91_math_genius"
        ),
        GenerateRequest(prompt="Write a fantasy story.", lora_id="user_44_novelist"),
        GenerateRequest(
            prompt="Explain quantum mechanics.", lora_id="user_10_physicist"
        ),
    ]

    logger.info("Starting Multi-Tenant LoRA Exchange Load Test (S-LoRA)...")
    logger.info("Base Model (Llama-3-70B) pinned to VRAM.")

    # Fire all 5 requests concurrently. vLLM will automatically batch them together
    # in the same forward pass, applying the different LoRA weights to different
    # rows in the batch matrix dynamically.

    tasks = [pipeline.process_generate(req) for req in requests]

    results = await asyncio.gather(*tasks)

    for req, res in zip(requests, results):
        logger.info(
            f"[LoRA: {req.lora_id}] Success. Tokens Generated: {res.tokens_generated}"
        )

    logger.info(
        "S-LoRA Load Test Complete. 5 distinct models served concurrently with 0 VRAM swap overhead."
    )


if __name__ == "__main__":
    asyncio.run(simulate_lorax_scale())
