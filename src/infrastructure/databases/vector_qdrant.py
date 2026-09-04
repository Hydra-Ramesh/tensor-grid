from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from src.utils.logger import logger
from src.infrastructure.ai_models.embeddings import encode_chunks, encode_query
from src.infrastructure.ai_models.reranker import rank_pairs
import uuid
import os
import asyncio

QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
# Connect using Qdrant's high-speed gRPC binary protocol instead of HTTP/JSON
qdrant = AsyncQdrantClient(
    host=QDRANT_HOST, port=6334, grpc_port=6334, prefer_grpc=True
)
COLLECTION_NAME = "enterprise_docs"


async def init_qdrant():
    try:
        try:
            await qdrant.get_collection(collection_name=COLLECTION_NAME)
            logger.info("Qdrant collection already exists.")
        except Exception:
            await qdrant.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=384, distance=Distance.COSINE),
            )
            logger.info("Created Qdrant collection: enterprise_docs")
    except Exception as e:
        logger.warning(
            {
                "event": "qdrant_init_failed",
                "warning": "Qdrant is unreachable. RAG endpoints will fail until Qdrant is started.",
                "error": str(e),
            }
        )


async def ingest_document(text: str, source: str) -> int:
    """Chunks text, embeds it using the AI module, and saves to Qdrant."""
    chunk_size = 500
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    # Offload PyTorch encoding to a separate thread
    embeddings = await asyncio.to_thread(encode_chunks, chunks)

    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={"text": chunk, "source": source},
        )
        for chunk, embedding in zip(chunks, embeddings)
    ]

    await qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(chunks)


async def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieves chunks from Qdrant and Re-Ranks them using a Cross-Encoder."""
    # Prevent PyTorch from freezing the async event loop during dense embedding
    query_vector = await asyncio.to_thread(encode_query, query)

    try:
        # 1. Over-fetch using Dense Vectors (Semantic Search)
        search_result = await qdrant.search(
            collection_name=COLLECTION_NAME, query_vector=query_vector, limit=10
        )

        if not search_result:
            return ""

        chunks = [hit.payload["text"] for hit in search_result]

        # 2. Re-Rank the retrieved chunks mathematically
        pairs = [[query, chunk] for chunk in chunks]

        # Prevent PyTorch from freezing the async event loop during re-ranking
        scores = await asyncio.to_thread(rank_pairs, pairs)

        # 3. Sort chunks by score in descending order and slice the top_k
        ranked_chunks = [
            chunk for _, chunk in sorted(zip(scores, chunks), reverse=True)
        ]
        best_chunks = ranked_chunks[:top_k]

        return "\n\n".join(best_chunks)
    except Exception as e:
        logger.error({"event": "rag_retrieval_failed", "error": str(e)})
        return ""
