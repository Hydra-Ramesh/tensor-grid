from fastapi.responses import EventSourceResponse
from src.api.schemas import GenerateRequest, GenerateResponse, RAGGenerateRequest
from src.infrastructure.databases.vector_qdrant import retrieve_context
from src.infrastructure.databases.graph_neo4j import retrieve_graph_context
from src.core.use_cases.generation_pipeline import GenerationPipeline
from src.utils.tracing import tracer


class RAGPipeline:
    """
    Pure Business Logic Orchestrator for Retrieval-Augmented Generation.
    Strictly handles context fetching (Graph + Vector) and delegates to GenerationPipeline.
    """

    def __init__(self, generation_pipeline: GenerationPipeline):
        self.generation_pipeline = generation_pipeline

    async def process_rag(
        self, request: RAGGenerateRequest
    ) -> GenerateResponse | EventSourceResponse:
        with tracer.start_as_current_span("rag_pipeline"):
            # Hybrid GraphRAG Retrieval
            context_blocks = await retrieve_context(request.prompt)
            graph_blocks = await retrieve_graph_context(request.prompt)

            context_str = (
                "\n\n".join(context_blocks)
                if isinstance(context_blocks, list)
                else context_blocks
            )

            augmented_prompt = f"System: Use the following context to answer the user.\n\n[Vector Database Context]:\n{context_str}\n\n[Graph Database Context]:\n{graph_blocks}\n\nUser: {request.prompt}"

            gen_req = GenerateRequest(
                prompt=augmented_prompt,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=request.stream,
            )

            return await self.generation_pipeline.process_generate(gen_req)
