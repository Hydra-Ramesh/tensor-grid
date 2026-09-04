import ray
from ray import serve
from src.api.schemas import GenerateRequest, RAGGenerateRequest
from src.core.use_cases.generation_pipeline import GenerationPipeline
from src.core.use_cases.rag_pipeline import RAGPipeline
from src.core.use_cases.tool_executor import ToolRegistry
from src.utils.logger import logger

# Initialize Ray Serve cluster
ray.init(ignore_reinit_error=True)
serve.start(detached=True)


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 1})
class ToolActor:
    """Microservice handling the Agentic Reflection Loop."""

    def __init__(self):
        self.registry = ToolRegistry()

    async def execute(self, prompt: str):
        return await self.registry.execute_tool_for_prompt(prompt)


@serve.deployment(num_replicas=4, ray_actor_options={"num_gpus": 1})
class GenerationActor:
    """Microservice handling LLM Inference and Semantic Routing."""

    def __init__(self, tool_actor):
        # We simulate injecting the tool actor via Serve Handles
        self.tool_actor = tool_actor
        # We initialize the base pipeline here, bypassing standard injection for the distributed cluster
        # In a true cluster, GenerationPipeline wouldn't exist as a monolith, its functions would be here.
        # But for architecture demonstration, we wrap it.
        from src.core.use_cases.tool_executor import ToolRegistry

        self.pipeline = GenerationPipeline(ToolRegistry())

    async def __call__(self, request: GenerateRequest):
        logger.info(f"[GenerationActor] Processing: {request.prompt[:20]}")
        return await self.pipeline.process_generate(request)


@serve.deployment(num_replicas=2, ray_actor_options={"num_cpus": 2})
class RAGActor:
    """Microservice handling Graph and Vector retrieval."""

    def __init__(self, generation_actor):
        self.generation_actor = generation_actor
        # Mock initialization for the orchestrator
        from src.core.use_cases.generation_pipeline import GenerationPipeline
        from src.core.use_cases.tool_executor import ToolRegistry

        gen_pipe = GenerationPipeline(ToolRegistry())
        self.pipeline = RAGPipeline(gen_pipe)

    async def __call__(self, request: RAGGenerateRequest):
        logger.info(f"[RAGActor] Fetching context for: {request.prompt[:20]}")
        # In a real Ray Serve setup, we would call the generation_actor handle here via gRPC.
        return await self.pipeline.process_rag(request)


# To bind and deploy:
# tool_actor = ToolActor.bind()
# generation_actor = GenerationActor.bind(tool_actor)
# rag_actor = RAGActor.bind(generation_actor)
