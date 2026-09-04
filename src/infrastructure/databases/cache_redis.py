import json
import redis.asyncio as redis
import numpy as np
import asyncio
from sentence_transformers import SentenceTransformer
from src.utils.logger import logger
from src.engine.config import settings

# Initialize Redis client
REDIS_URL = settings.redis_url
redis_client = redis.from_url(REDIS_URL, decode_responses=False)

# We use a tiny, ultra-fast embedding model that runs easily on CPU.
# This converts text into a 384-dimensional vector.
embedder = SentenceTransformer("all-MiniLM-L6-v2")

SIMILARITY_THRESHOLD = 0.95


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    return float(dot_product / (norm_a * norm_b))


def _encode_prompt(prompt: str) -> np.ndarray:
    return embedder.encode(prompt)


async def check_cache(prompt: str) -> str | None:
    """
    Checks if a semantically similar prompt exists in the cache.
    Returns the cached response if found, else None.
    """
    # 1. Embed the incoming prompt using thread to avoid blocking
    prompt_vector = await asyncio.to_thread(_encode_prompt, prompt)

    # 2. In a real production system with millions of rows, we would use
    # RedisSearch (RediSearch vector similarity). For this demo, we do a
    # simple scan of keys and manual cosine similarity to avoid requiring
    # the RedisStack image.

    # Get all cached keys (format: cache:prompt_text)
    keys = await redis_client.keys("cache:*")

    for key in keys:
        # Decode the byte key
        key_str = key.decode("utf-8")
        cached_prompt = key_str.replace("cache:", "")

        # Get the cached vector from redis
        cached_data = await redis_client.get(key)
        if cached_data:
            data = json.loads(cached_data.decode("utf-8"))
            cached_vector = np.array(data["vector"])

            # Calculate similarity
            similarity = cosine_similarity(prompt_vector, cached_vector)

            if similarity >= SIMILARITY_THRESHOLD:
                logger.info(
                    {
                        "event": "cache_hit",
                        "similarity": similarity,
                        "cached_prompt": cached_prompt,
                    }
                )
                return data["response"]

    logger.info({"event": "cache_miss", "prompt": prompt})
    return None


async def set_cache(prompt: str, response: str):
    """
    Saves the prompt vector and response to Redis.
    """
    prompt_vector = (await asyncio.to_thread(_encode_prompt, prompt)).tolist()

    key = f"cache:{prompt}"
    data = {"vector": prompt_vector, "response": response}

    # Cache for 24 hours
    await redis_client.setex(key, 86400, json.dumps(data))
    logger.info({"event": "cache_set", "prompt": prompt})
