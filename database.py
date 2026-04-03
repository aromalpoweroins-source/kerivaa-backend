import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

_supabase_client = None

def get_supabase():
    """Lazy-initialize the Supabase client."""
    global _supabase_client
    if _supabase_client is None:
        url = os.getenv('SUPABASE_URL', '')
        key = os.getenv('SUPABASE_KEY', '')
        if not url or url.startswith('your-') or not key or key.startswith('your-'):
            raise RuntimeError(
                'Supabase credentials not configured. '
                'Update SUPABASE_URL and SUPABASE_KEY in .env'
            )
        _supabase_client = create_client(url, key)
    return _supabase_client

# Keep `supabase` as a property-like accessor for backwards compatibility
class _SupabaseProxy:
    """Proxy that lazily initializes the Supabase client on first use."""
    def __getattr__(self, name):
        return getattr(get_supabase(), name)

supabase = _SupabaseProxy()

def get_all_destinations(budget: str) -> list:
    """Get all destinations matching the budget level."""
    result = supabase.table('destinations') \
        .select('*') \
        .eq('budget_level', budget) \
        .execute()
    return result.data

def get_experiences_for_destination(destination_id: int) -> list:
    """Get all experiences for a destination, best first."""
    result = supabase.table('experiences') \
        .select('*') \
        .eq('destination_id', destination_id) \
        .execute()
    data = result.data
    data.sort(
        key=lambda x: float(x.get('popularity_score', 1.0)),
        reverse=True
    )
    return data

def get_travel_time(from_id: int, to_id: int) -> float:
    """
    Get travel hours between two destinations.
    Returns 999 if the pair is not in the database.
    999 means --- never use this route.
    """
    result = supabase.table('travel_times') \
        .select('travel_hours') \
        .eq('from_destination_id', from_id) \
        .eq('to_destination_id', to_id) \
        .execute()
    if result.data:
        return float(result.data[0]['travel_hours'])
    return 999

def save_interaction_log(session_id: str, user_text: str,
                         intent: dict, itinerary: dict) -> None:
    """Log every trip generation for analytics."""
    import json
    try:
        supabase.table('interaction_logs').insert({
            'session_id': session_id,
            'user_text': user_text,
            'intent_json': json.dumps(intent),
            'itinerary_json': json.dumps(itinerary),
            'destinations_shown': itinerary.get('destinations', [])
        }).execute()
    except Exception as e:
        print(f'Logging error (non-fatal): {e}')