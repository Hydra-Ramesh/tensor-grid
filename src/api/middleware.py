import time
from fastapi import HTTPException, status
import redis.asyncio as redis
from src.utils.logger import logger
from src.engine.config import settings

REDIS_URL = settings.redis_url
redis_client = redis.from_url(REDIS_URL)

# Rate limit settings
RATE_LIMIT_REQUESTS = settings.rate_limit_requests
RATE_LIMIT_WINDOW = settings.rate_limit_window


async def check_rate_limit(api_key: str):
    """
    Implements a simple counter-based rate limit using Redis.
    (A true Token Bucket would use Lua scripts for atomicity, but this suffices for the demo).
    """
    current_time = int(time.time())
    window_start = current_time - (current_time % RATE_LIMIT_WINDOW)

    redis_key = f"rate_limit:{api_key}:{window_start}"

    # Increment the counter
    current_requests = await redis_client.incr(redis_key)

    # Set expiration on the key if it's new
    if current_requests == 1:
        await redis_client.expire(redis_key, RATE_LIMIT_WINDOW * 2)

    if current_requests > RATE_LIMIT_REQUESTS:
        logger.warning({"event": "rate_limit_exceeded", "api_key": api_key})
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later.",
        )
