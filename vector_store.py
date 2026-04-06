import json
from database import supabase
from embeddings import embed_text

def save_itinerary(user_text: str, itinerary: dict, source: str = 'user', quality: float = 1.0):
    """
    Save an itinerary to the vector store with its embedding.
    Called automatically after generating a narrative.
    """
    # Build searchable text from the itinerary
    itin_text = build_itinerary_text(itinerary)
    
    # Combine user request + itinerary content for better matching
    combined = f"{user_text} {itin_text}"
    
    # Generate vector embedding
    vector = embed_text(combined)
    
    # Save to Supabase
    supabase.table('stored_itineraries').insert({
        'user_text': user_text,
        'duration_days': itinerary.get('total_days'),
        'budget': itinerary.get('budget'),
        'styles': ','.join(itinerary.get('styles', [])),
        'itinerary_json': json.dumps(itinerary),
        'embedding': vector,
        'source': source,
        'quality_score': quality
    }).execute()
    
    print(f"Saved itinerary to vector store: {user_text[:50]}...")

def find_similar(user_text: str, limit: int = 3) -> list:
    """
    Find similar past itineraries based on vector similarity.
    Returns top N matches with similarity score.
    """
    # Generate embedding for the search query
    vector = get_embedding(user_text)
    
    # Call the Supabase function match_itineraries
    result = supabase.rpc('match_itineraries', {
        'query_embedding': vector,
        'match_threshold': 0.6,  # Minimum 60% similarity
        'match_count': limit
    }).execute()
    
    if result.data:
        print(f"Found {len(result.data)} similar itineraries")
        return result.data
    else:
        return []

def mark_as_selected(itinerary_id: int):
    """Mark an itinerary as saved/rated highly (quality boost)."""
    supabase.table('stored_itineraries').update({
        'was_selected': True,
        'quality_score': 1.5
    }).eq('id', itinerary_id).execute()


def build_itinerary_text(itinerary: dict) -> str:
    """
    Flatten a generated itinerary into one searchable text string.
    """
    parts = [
        str(itinerary.get('budget', '')),
        ' '.join(itinerary.get('styles', [])),
        ' '.join(itinerary.get('destinations', [])),
    ]

    for day in itinerary.get('itinerary', []):
        parts.append(str(day.get('destination', '')))
        parts.append(str(day.get('type', '')))
        parts.append(str(day.get('narrative', '')))

        for exp in day.get('experiences', []):
            parts.append(str(exp.get('name', '')))
            parts.append(str(exp.get('tags', '')))
            parts.append(str(exp.get('best_time_of_day', '')))

    return ' | '.join(part for part in parts if part)
