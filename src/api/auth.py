from fastapi import Security, HTTPException, status
from fastapi.security.api_key import APIKeyHeader
from src.utils.logger import logger

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

# In production, these would be securely hashed in a database.
# For the resume project, we hardcode a demo key.
VALID_API_KEYS = {"sk-faang-resume-project-12345": "test_user_1"}


async def verify_api_key(api_key: str = Security(api_key_header)) -> str:
    """
    Verifies the provided API key. Returns the associated API key if valid.
    Raises 401 Unauthorized if invalid.
    """
    if not api_key:
        logger.warning({"event": "auth_failure", "reason": "missing_key"})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="API Key is missing"
        )

    user_id = VALID_API_KEYS.get(api_key)
    if not user_id:
        logger.warning(
            {"event": "auth_failure", "reason": "invalid_key", "key": api_key}
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API Key"
        )

    return api_key
