from sentence_transformers import SentenceTransformer

# Dense Embedding Model (Conceptual Retrieval)
# Loaded globally so it only initializes once in memory
encoder = SentenceTransformer("all-MiniLM-L6-v2")


def encode_query(query: str):
    """Encodes a string query into a 384-dimensional vector."""
    return encoder.encode(query).tolist()


def encode_chunks(chunks: list[str]):
    """Encodes a list of string chunks into a list of 384-dimensional vectors."""
    return encoder.encode(chunks).tolist()
