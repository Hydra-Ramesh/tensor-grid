from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Server Config
    host: str = "0.0.0.0"
    port: int = 8000

    # vLLM Engine Config
    model_name: str = "meta-llama/Meta-Llama-3-70B-Instruct"
    gpu_memory_utilization: float = 0.95  # Maximize KV Cache on massive GPUs
    max_model_len: int = 32768  # Full production context window
    tensor_parallel_size: int = 4  # Distribute across 4x A100/H100s
    enable_lora: bool = True
    max_loras: int = 32  # Support 32 concurrent LoRAs
    max_lora_rank: int = 64

    # V5 Performance Config
    enable_prefix_caching: bool = True
    enable_chunked_prefill: bool = (
        True  # V9: Prevents large prompts from blocking generation
    )
    speculative_model: str = (
        "meta-llama/Meta-Llama-3-8B-Instruct"  # V11: Massive speedup via 8B Draft Model verification
    )
    num_speculative_tokens: int = 5
    kv_cache_dtype: str = "fp8"  # V6: Quantize KV cache for 2x concurrency
    rope_scaling_factor: float = 2.0  # V7: Double the context window via RoPE scaling
    swap_space: int = 64  # Offload up to 64GB KV Cache to Enterprise CPU RAM

    # Rate Limiter & Cache
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # KV Cache Persistence
    kv_cache_mount_path: str = "/mnt/nvme0n1/kv_cache"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
