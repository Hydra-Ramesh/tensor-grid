import asyncio
import json
import random
from vllm import AsyncLLMEngine
from src.utils.logger import logger

# Simulated Teacher Model
# In production, this would be Llama-3-70B running on an 8xH100 cluster
TEACHER_MODEL = "meta-llama/Meta-Llama-3-70B-Instruct"

PROMPT_TEMPLATES = [
    "Write a complex {domain} question that requires deep reasoning.",
    "Generate a tricky edge-case scenario involving {domain}.",
    "Explain a highly advanced concept in {domain} as if I am an expert.",
]

DOMAINS = ["Machine Learning", "Quantum Physics", "System Design", "Rust Programming"]


async def generate_synthetic_data(engine: AsyncLLMEngine, num_samples: int = 100):
    """
    Teacher model generates diverse questions and answers them autonomously.
    This creates the Distillation Dataset to fine-tune the 8B Student models.
    """
    dataset = []

    logger.info(
        f"Starting Synthetic Distillation Flywheel using Teacher: {TEACHER_MODEL}"
    )

    for i in range(num_samples):
        domain = random.choice(DOMAINS)
        template = random.choice(PROMPT_TEMPLATES)

        # 1. Teacher generates a hard question
        question_prompt = template.format(domain=domain)
        # Mocking the generation for architecture setup
        generated_question = (
            f"[Synthetic Question]: How do you optimize {domain} at scale?"
        )

        # 2. Teacher answers its own question perfectly
        # In real execution, we await the vLLM stream here.
        generated_answer = f"[Synthetic Answer]: The optimal approach for {domain} involves distributed parallel processing and memory-mapped IO."

        dataset.append({"instruction": generated_question, "output": generated_answer})

        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i+1} synthetic samples...")

    # 3. Save the dataset for the Student model's nightly LoRA fine-tuning
    with open("data/synthetic_dataset.jsonl", "w") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        f"Distillation Flywheel Complete. Saved {num_samples} rows to data/synthetic_dataset.jsonl"
    )


if __name__ == "__main__":
    # engine = get_engine_instance() # We mock this for the script
    engine = None
    asyncio.run(generate_synthetic_data(engine, num_samples=100))
