import uuid
from datetime import datetime
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from planner import plan_trip
from intent import extract_intent
from rag import generate_with_rag
from database import save_interaction_log

app = FastAPI(title='Kerivaa Planning Engine')

# CORS --- allows Lovable frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

# ── REQUEST MODELS ───────────────────────────────────────────────

class TripRequest(BaseModel):
    duration_days: int
    budget: str
    styles: List[str]
    month: Optional[int] = None

class TextRequest(BaseModel):
    text: str

class FeedbackRequest(BaseModel):
    session_id: str
    rating: Optional[int] = None
    itinerary_saved: Optional[bool] = False
    itinerary_shared: Optional[bool] = False
    time_spent_seconds: Optional[int] = None

# ── ENDPOINTS ────────────────────────────────────────────────────

@app.get('/')
def root():
    return {
        'name': 'Kerivaa Planning Engine',
        'status': 'running',
        'version': '1.0'
    }

@app.get('/health')
def health():
    return {
        'status': 'healthy',
        'time': datetime.utcnow().isoformat()
    }

@app.post('/plan')
def plan_structured(request: TripRequest):
    """
    Endpoint 1: structured input.
    Receives duration, budget, styles as structured data.
    Returns itinerary WITH narrative.
    Used by the filter-based form in Lovable.
    """
    structured = plan_trip(
        duration=request.duration_days,
        budget=request.budget,
        styles=request.styles,
        month=request.month
    )

    if 'error' in structured:
        return structured

    return generate_narrative(structured, request.styles)

@app.post('/plan-from-text')
def plan_from_text(request: TextRequest):
    """
    Endpoint 2: natural language input.
    Receives plain English trip description.
    Extracts intent with Ollama, plans trip, writes narrative.
    Returns complete enriched itinerary.
    Used by the text input box in Lovable.
    """
    try:
        session_id = str(uuid.uuid4())
        
        intent = extract_intent(request.text)
        
        structured = plan_trip(
            duration=intent['duration_days'],
            budget=intent['budget'],
            styles=intent['styles'],
            month=intent.get('month')
        )
        
        if 'error' in structured:
            return {
                'error': structured['error'],
                'session_id': session_id
            }
        
        enriched = generate_with_rag(request.text, structured)
        
        try:
            save_interaction_log(
                session_id, request.text, intent, enriched
            )
        except Exception as e:
            print(f'Logging failed (non-fatal): {e}')
        
        return {
            **enriched,
            'session_id': session_id,
            'intent_extracted': intent
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {'status': 'error', 'detail': str(e)}

@app.post('/feedback')
def record_feedback(request: FeedbackRequest):
    """
    Endpoint 3: user feedback.
    Called by Lovable when user rates or saves an itinerary.
    """
    from database import supabase
    
    try:
        supabase.table('interaction_logs') \
            .update({
                'rating': request.rating,
                'itinerary_saved': request.itinerary_saved,
                'itinerary_shared': request.itinerary_shared,
                'time_spent_seconds': request.time_spent_seconds
            }) \
            .eq('session_id', request.session_id) \
            .execute()
        return {'status': 'logged'}
    except Exception as e:
        return {'status': 'error', 'detail': str(e)}
