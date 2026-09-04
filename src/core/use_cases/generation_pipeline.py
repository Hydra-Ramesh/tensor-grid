from fastapi.responses import EventSourceResponse
from src.api.schemas import GenerateRequest, GenerateResponse
from src.infrastructure.llm.vllm_client import get_engine_instance, stream_response
from src.infrastructure.databases.cache_redis import check_cache, set_cache
from src.infrastructure.ai_models.semantic_router import classify_prompt
from src.core.use_cases.tool_executor import ToolRegistry
from src.utils.tracing import tracer
import uuid
import json


class GenerationPipeline:
    """
    Pure Business Logic Orchestrator for Text and Vision Generation.
    Strictly isolated from Databases (Qdrant, Neo4j).
    """

    def __init__(self, tool_registry: ToolRegistry):
        self.tool_registry = tool_registry

    async def process_generate(
        self, request: GenerateRequest
    ) -> GenerateResponse | EventSourceResponse:
        request_id = str(uuid.uuid4())

        with tracer.start_as_current_span("generate_pipeline"):
            # 1. Semantic Routing & Tools (Mixture of Experts)
            if not request.lora_id:
                with tracer.start_as_current_span("semantic_router"):
                    routed_domain = await classify_prompt(request.prompt)
                    if routed_domain == "tools-lora":
                        with tracer.start_as_current_span("tool_execution"):
                            # Agentic Reflection & Auto-Healing Loop
                            max_retries = 3
                            for attempt in range(max_retries):
                                try:
                                    augmented_prompt = await self.tool_registry.execute_tool_for_prompt(
                                        request.prompt
                                    )
                                    if augmented_prompt:
                                        request.prompt = augmented_prompt
                                    break  # Success
                                except Exception as e:
                                    if attempt == max_retries - 1:
                                        raise e
                                    request.prompt += f"\n\n[SYSTEM ERROR]: The previous tool execution failed with: {str(e)}. Please fix the syntax or parameters and try again."
                    elif routed_domain:
                        request.lora_id = routed_domain

            # 2. Semantic Caching
            with tracer.start_as_current_span("semantic_cache"):
                cached_response = await check_cache(request.prompt)
                if cached_response:
                    if request.stream:

                        async def stream_cached():
                            yield f"data: {json.dumps({'text': cached_response})}\n\n"

                        return EventSourceResponse(stream_cached())
                    return GenerateResponse(
                        request_id="cached", text=cached_response, tokens_generated=0
                    )

            # 2.5 Multi-Modal Vision Integration
            if request.image_url:
                with tracer.start_as_current_span("vision_encoding"):
                    request.prompt = f"User: <image>\n{request.prompt}\n(Image URL: {request.image_url})\nAssistant:"

            # 3. LLM Execution
            with tracer.start_as_current_span("vllm_execution"):
                engine = get_engine_instance()
                if request.stream:
                    return EventSourceResponse(
                        stream_response(request, request_id, engine)
                    )

                results = []
                async for chunk in stream_response(request, request_id, engine):
                    if chunk.startswith("data: "):
                        try:
                            data = json.loads(chunk[6:])
                            results.append(data.get("text", ""))
                        except json.JSONDecodeError:
                            pass

                final_text = "".join(results)
                await set_cache(request.prompt, final_text)

                return GenerateResponse(
                    request_id=request_id,
                    text=final_text,
                    tokens_generated=len(final_text.split()),
                )
