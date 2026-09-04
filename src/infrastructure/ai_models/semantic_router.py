import numpy as np
import asyncio
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger

# Reuse the same lightweight CPU encoder as Vector DB and Semantic Cache
encoder = SentenceTransformer("all-MiniLM-L6-v2")

# Define semantic clusters (Centroids) for domains
# In a true FAANG environment, these would be averaged over thousands of historical prompts.
DOMAINS = {
    "coder-lora": [
        "Write a python script",
        "How do I debug this C++ code?",
        "Implement a binary search tree in Java",
        "Fix the syntax error in my React component",
        "def hello_world():",
    ],
    "math-lora": [
        "Solve this differential equation",
        "What is the integral of x^2?",
        "Prove the Pythagorean theorem",
        "Calculate the probability of drawing a full house",
        "1 + 1 =",
    ],
    "sql-lora": [
        "Write a query to SELECT from users",
        "Join the tables on id",
        "GROUP BY department HAVING count > 5",
        "What is a left outer join?",
        "Create a database schema for an e-commerce site",
    ],
    "tools-lora": [
        "What is the weather in Tokyo?",
        "Fetch the current temperature in New York",
        "Can you check the weather for me?",
        "Calculate 25 * 45",
        "What is 100 divided by 4?",
    ],
}

# Pre-compute centroids on import
DOMAIN_CENTROIDS = {}
for domain, prompts in DOMAINS.items():
    embeddings = encoder.encode(prompts)
    centroid = np.mean(embeddings, axis=0)
    # Normalize for cosine similarity
    DOMAIN_CENTROIDS[domain] = centroid / np.linalg.norm(centroid)


def _encode_prompt(prompt: str):
    return encoder.encode([prompt])[0]


async def classify_prompt(prompt: str) -> str:
    """Returns the best lora_id for the given prompt based on semantic similarity."""
    # CPU-bound sentence encoding must run in a thread to prevent blocking concurrent API requests
    emb = await asyncio.to_thread(_encode_prompt, prompt)
    emb = emb / np.linalg.norm(emb)

    best_score = -1.0
    best_domain = None

    for domain, centroid in DOMAIN_CENTROIDS.items():
        score = np.dot(emb, centroid)
        if score > best_score:
            best_score = score
            best_domain = domain

    # Routing Threshold: If it doesn't match well, don't use a specialized LoRA (returns None)
    if best_score > 0.45:
        logger.info(
            {
                "event": "semantic_routing",
                "prompt_start": prompt[:30],
                "routed_to": best_domain,
                "score": float(best_score),
            }
        )
        return best_domain

    return None
