"""
Seed the vector store with 20+ diverse Kerala itineraries.
Run this ONCE before launch to give RAG examples to learn from.
"""
from intent import extract_intent
from planner import plan_trip
from rag import generate_with_rag

# Diverse Kerala trip requests covering all styles, durations, budgets
SEED_REQUESTS = [
    # Romantic trips
    "5 day slow romantic honeymoon Kerala backwaters",
    "3 day romantic anniversary trip Alleppey houseboat",
    "7 day romantic Kerala hill stations Munnar Vagamon",
    
    # Family trips
    "5 day family trip Kerala wildlife Wayanad Thekkady",
    "6 day family vacation Kerala beaches and culture",
    "4 day family trip with kids Kochi Munnar",
    
    # Adventure trips
    "7 day adventure Kerala trekking Wayanad Munnar",
    "5 day adventure trip waterfalls wildlife Kerala",
    "4 day offbeat Kerala hidden places local experience",
    
    # Culture & Heritage
    "5 day culture heritage temples Kerala food tour",
    "6 day Kerala pilgrimage Guruvayur temples",
    "4 day Kochi heritage walk spice markets",
    
    # Relaxation & Nature
    "3 day slow relaxed backwaters Alleppey",
    "5 day nature wildlife Kerala bird watching",
    "7 day ayurveda wellness retreat Kerala",
    
    # Budget trips
    "5 day budget Kerala backpacker local stays",
    "3 day cheap Kerala trip low cost",
    
    # Luxury trips
    "5 day luxury Kerala premium resorts",
    "7 day high end Kerala exclusive experiences",
    
    # Mixed experiences
    "6 day beach and backwaters Kerala south",
    "5 day hill station tea gardens Munnar",
    "4 day Kerala food tour local cuisine",
]

def seed_database():
    print(f"Seeding {len(SEED_REQUESTS)} itineraries...\n")
    
    for i, text in enumerate(SEED_REQUESTS, 1):
        print(f"[{i}/{len(SEED_REQUESTS)}] Processing: {text[:50]}...")
        
        try:
            # Extract intent
            intent = extract_intent(text)
            
            # Generate structured itinerary
            structured = plan_trip(
                duration=intent['duration_days'],
                budget=intent['budget'],
                styles=intent['styles']
            )
            
            if 'error' in structured:
                print(f"  ⚠️  Planner error: {structured['error']}")
                continue
            
            # Generate RAG-enhanced narrative (this auto-saves to vector store)
            # We call the RAG components directly to mark as 'seed'
            from vector_store import save_itinerary
            
            enriched = generate_with_rag(text, structured)
            
            # Re-save with higher quality score as a seed
            save_itinerary(text, enriched, source='seed', quality=1.5)
            
            print(f"  ✅ Saved: {enriched['destinations']} | {enriched['total_days']} days")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print(f"\n✅ Seeding complete! Your RAG system now has examples.")

if __name__ == "__main__":
    # Make sure Ollama is running before executing
    print("Make sure Ollama is running: ollama serve")
    input("Press Enter to start seeding...")
    seed_database()