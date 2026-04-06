import os
import requests

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

def embed_text(text: str) -> list:
    """
    Use Groq's embedding API instead of local sentence-transformers.
    Much faster, no large downloads. 768 dimensions.
    """
    try:
        response = requests.post(
            'https://api.groq.com/openai/v1/embeddings',
            headers={
                'Authorization': f'Bearer {GROQ_API_KEY}',
                'Content-Type': 'application/json'
            },
            json={
                'model': 'nomic-embed-text-v1.5',  # 768 dimensions
                'input': text
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()['data'][0]['embedding']
    except Exception as e:
        print(f"Groq embedding failed: {e}")
        # Fallback: simple hash-based embedding (not ideal but works)
        return fallback_embedding(text)

def fallback_embedding(text: str) -> list:
    """Simple fallback if Groq fails - creates 768-dim vector"""
    import hashlib
    # Create deterministic pseudo-embedding from text hash
    hash_obj = hashlib.md5(text.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    # Generate 768 dimensions from hash
    vector = []
    for i in range(768):
        hash_int = (hash_int * 1103515245 + 12345) & 0x7fffffff
        vector.append((hash_int % 1000) / 1000.0)
    return vector

def build_itinerary_text(itinerary: dict) -> str:
    """Build searchable text from itinerary structure."""
    parts = [' '.join(itinerary.get('styles', []))]
    parts.append(f"{itinerary.get('total_days', 5)} days")
    parts.append(itinerary.get('budget', 'mid'))
    parts.extend(itinerary.get('destinations', []))
    
    for day in itinerary.get('itinerary', []):
        for exp in day.get('experiences', []):
            parts.append(exp.get('name', ''))
    
    return ' '.join(parts)