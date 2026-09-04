import pytest
from httpx import AsyncClient
from src.api.main import app

@pytest.mark.asyncio
async def test_health_check():
    """Verify the health check endpoint returns 200 OK."""
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

@pytest.mark.asyncio
async def test_generate_requires_auth():
    """Verify that the /generate endpoint rejects requests without an API key."""
    payload = {
        "prompt": "Test",
        "max_tokens": 10
    }
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/generate", json=payload)
    
    assert response.status_code == 401
    assert response.json()["detail"] == "API Key is missing"

@pytest.mark.asyncio
async def test_generate_invalid_auth():
    """Verify that the /generate endpoint rejects invalid API keys."""
    payload = {
        "prompt": "Test",
        "max_tokens": 10
    }
    headers = {
        "X-API-Key": "invalid-key-123"
    }
    async with AsyncClient(app=app, base_url="http://test") as ac:
        response = await ac.post("/generate", json=payload, headers=headers)
    
    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid API Key"
