import pytest
import sys
from unittest.mock import patch, MagicMock

# 1. Mock vLLM module completely BEFORE any imports
sys.modules['vllm'] = MagicMock()
sys.modules['vllm.engine'] = MagicMock()
sys.modules['vllm.engine.async_llm_engine'] = MagicMock()
sys.modules['vllm.engine.arg_utils'] = MagicMock()

from fastapi.testclient import TestClient

# 2. Mock get_engine_instance
with patch("src.infrastructure.llm.vllm_client.get_engine_instance") as mock_get_engine:
    mock_engine = MagicMock()
    mock_get_engine.return_value = mock_engine

    from src.main import app

client = TestClient(app)

client = TestClient(app)

@patch("src.core.use_cases.o1_reasoning_pipeline.stream_response")
def test_o1_generate_endpoint_mock(mock_stream_response):
    # Mock the generator
    async def mock_generator(*args, **kwargs):
        yield "data: {\"text\": \"<thinking>This is a test</thinking>\"}\n\n"
        yield "data: {\"text\": \"<final_answer>42</final_answer>\"}\n\n"
        
    mock_stream_response.return_value = mock_generator()

    response = client.post(
        "/api/v1/o1_generate",
        json={"prompt": "What is the meaning of life?"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    # The pipeline should strip out the thinking tags
    assert data["text"] == "42"

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
