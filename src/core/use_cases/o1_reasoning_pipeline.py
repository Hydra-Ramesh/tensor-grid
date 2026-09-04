from fastapi.responses import EventSourceResponse
from src.api.schemas import GenerateRequest, GenerateResponse
from src.infrastructure.llm.vllm_client import get_engine_instance, stream_response
from src.utils.tracing import tracer
from src.utils.logger import logger
import uuid
import json


class O1ReasoningPipeline:
    """
    Test-Time Compute Pipeline (o1-style).
    Instead of answering immediately, the model is forced into a hidden "Thinking" loop,
    generating thousands of tokens of reasoning, critiquing itself, and finally generating
    a verified answer. The user only receives the final answer.
    """

    async def process_reasoning(
        self, request: GenerateRequest
    ) -> GenerateResponse | EventSourceResponse:
        request_id = str(uuid.uuid4())

        with tracer.start_as_current_span("o1_test_time_compute"):
            # 1. Force the model into a thinking phase
            reasoning_prompt = f"""
            You are a hyper-logical advanced AI. Before answering the user, you MUST follow these steps:
            1. Enclose your internal chain-of-thought inside <thinking>...</thinking> tags.
            2. Inside the thinking block, break down the problem, critique your own logic, and double-check your math.
            3. Once you are 100% certain, output your final verified answer inside <final_answer>...</final_answer> tags.
            
            User Query: {request.prompt}
            """

            # We clone the request but hide the system prompt overhead from the schema
            reasoning_request = GenerateRequest(
                prompt=reasoning_prompt,
                max_tokens=request.max_tokens
                * 4,  # Allocate massive compute for thinking
                temperature=0.6,  # Slightly higher temperature for creative problem solving paths
                stream=False,  # Stream is intercepted, user only gets final answer
            )

            engine = get_engine_instance()

            # 2. Execute the massive forward pass (Test-Time Compute)
            logger.info(f"[{request_id}] Entering Test-Time Compute Reasoning Phase...")
            results = []
            async for chunk in stream_response(reasoning_request, request_id, engine):
                if chunk.startswith("data: "):
                    try:
                        data = json.loads(chunk[6:])
                        results.append(data.get("text", ""))
                    except json.JSONDecodeError:
                        pass

            raw_output = "".join(results)

            # 3. Parse out the <final_answer> to return to the user, discarding the <thinking> compute
            final_answer = raw_output
            if "<final_answer>" in raw_output:
                try:
                    final_answer = (
                        raw_output.split("<final_answer>")[1]
                        .split("</final_answer>")[0]
                        .strip()
                    )
                    logger.info(
                        f"[{request_id}] Reasoning successful. Extracted final answer."
                    )
                except IndexError:
                    pass

            return GenerateResponse(
                request_id=request_id,
                text=final_answer,
                tokens_generated=len(
                    raw_output.split()
                ),  # Bill the user for the thinking tokens too
            )
