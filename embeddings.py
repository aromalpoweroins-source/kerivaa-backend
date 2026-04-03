import math
from typing import Iterable, List, Sequence

from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_CACHE = "./models"
model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model from the local cache folder."""
    global model
    if model is None:
        try:
            model = SentenceTransformer(MODEL_NAME, cache_folder=MODEL_CACHE)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load embedding model 'all-MiniLM-L6-v2'. "
                "If this is the first run, make sure the model can be downloaded "
                "or is already cached in ./models."
            ) from exc
    return model


def _normalize(vector: Sequence[float]) -> List[float]:
    """Scale a vector to unit length for cosine similarity."""
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        return [0.0 for _ in vector]
    return [v / norm for v in vector]


def get_embedding(text: str) -> List[float]:
    """Generate one embedding vector using all-MiniLM-L6-v2."""
    if not text:
        return [0.0] * 384
    vector = _get_model().encode(text, normalize_embeddings=True)
    return vector.tolist()


def get_embeddings(texts: Sequence[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts."""
    if not texts:
        return []
    vectors = _get_model().encode(list(texts), normalize_embeddings=True)
    return [vector.tolist() for vector in vectors]


def cosine_similarity(vector_a: Sequence[float], vector_b: Sequence[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must be the same length.")
    return sum(a * b for a, b in zip(vector_a, vector_b))


def rank_texts_by_similarity(query: str, texts: Sequence[str], top_k: int = 5) -> List[dict]:
    """Rank candidate texts by embedding similarity to the query."""
    if top_k <= 0 or not texts:
        return []

    query_vector = get_embedding(query)
    text_vectors = get_embeddings(texts)

    scored = []
    for text, vector in zip(texts, text_vectors):
        scored.append(
            {
                "text": text,
                "score": round(cosine_similarity(query_vector, vector), 4),
            }
        )

    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def build_destination_search_text(destination: dict, experiences: Iterable[dict] | None = None) -> str:
    """Flatten destination and experience data into one searchable text block."""
    parts = [
        str(destination.get("name", "")),
        str(destination.get("region", "")),
        str(destination.get("tags", "")),
        str(destination.get("description", "")),
    ]

    if experiences:
        for exp in experiences:
            parts.extend(
                [
                    str(exp.get("name", "")),
                    str(exp.get("tags", "")),
                    str(exp.get("best_time_of_day", "")),
                ]
            )

    return " | ".join(part for part in parts if part)
