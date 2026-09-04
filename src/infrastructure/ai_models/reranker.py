from sentence_transformers import CrossEncoder

# Cross-Encoder Re-Ranker (Precise Scoring)
# Loaded globally so it only initializes once in memory
ranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rank_pairs(pairs: list[list[str]]):
    """
    Takes a list of [query, document] pairs and returns a list of relevance scores.
    """
    return ranker.predict(pairs)
