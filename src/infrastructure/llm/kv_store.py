import os
import torch
from typing import Optional
from src.core.config import settings
from src.utils.logger import logger


class PersistentKVCache:
    """
    Simulates dumping and loading the exact mathematical state of the Transformer's
    Attention matrices (KV Cache) to a persistent SSD volume.

    This enables Zero-Prefill Architecture: bypassing the O(N^2) attention computation
    for massive documents by streaming the pre-computed tensors directly into VRAM.
    """

    def __init__(self):
        self.mount_path = settings.kv_cache_mount_path
        os.makedirs(self.mount_path, exist_ok=True)
        logger.info(f"Initialized Persistent SSD KV-Cache at {self.mount_path}")

    def save_cache(self, document_id: str, kv_tensors: torch.Tensor):
        """Serialize the attention matrix to SSD."""
        file_path = os.path.join(self.mount_path, f"{document_id}.pt")
        # In a real FAANG cluster, this uses highly optimized PCIe DMA transfers.
        # We simulate with PyTorch save.
        torch.save(kv_tensors, file_path)
        logger.info(f"[PersistentKVCache] Saved attention state for {document_id}")

    def load_cache(self, document_id: str) -> Optional[torch.Tensor]:
        """Stream the attention matrix from SSD directly to VRAM."""
        file_path = os.path.join(self.mount_path, f"{document_id}.pt")
        if os.path.exists(file_path):
            logger.info(
                f"[PersistentKVCache] Loading zero-prefill state for {document_id}"
            )
            # map_location='cuda' handles the DMA transfer
            return torch.load(file_path, map_location="cpu")  # using CPU for mock
        return None
