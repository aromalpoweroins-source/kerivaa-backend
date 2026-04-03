import json
from llm import call_llm
from vector_store import find_similar, save_itinerary

def generate_with_rag(user_text: str, structured: dict) -> dict:
    """
    RAG-enhanced narrative generation.
    1. Finds 3 similar past itineraries
    2. Uses them as examples in the prompt
    3. Generates narrative with context
    4. Saves new itinerary for future retrieval
    """
    # Step 1: Retrieve similar itineraries from vector store
    similar = find_similar(user_text, limit=3)
    
    # Step 2: Build examples section for the prompt
    examples_text = ''
    if similar:
        examples_text = '\n\nSimilar Kerala trips for reference:\n'
        for i, match in enumerate(similar):
            past = json.loads(match['itinerary_json'])
            dests = ', '.join(past.get('destinations', []))
            styles = match.get('styles', 'unknown')
            
            examples_text += f"\nExample {i+1}: {match['duration_days']}d {styles} → {dests}\n"
            
            # Include sample narrative from first day if available
            days = past.get('itinerary', [])
            if days and 'narrative' in days[0] and days[0]['narrative']:
                sample = days[0]['narrative'][:150]
                examples_text += f"Sample prose: {sample}...\n"
    
    # Step 3: Build current itinerary text
    itin_text = ''
    for day in structured.get('itinerary', []):
        exps = ', '.join([e['name'] for e in day.get('experiences', [])])
        dest = day.get('destination', '')
        
        if day.get('type') == 'travel':
            itin_text += f"Day {day['day']}: Travel from {day.get('travel_from', '')} to {dest}\n"
        else:
            itin_text += f"Day {day['day']} in {dest}: {exps}\n"
    
    # Step 4: Create enhanced prompt with examples
    styles = ', '.join(structured.get('styles', []))
    
    prompt = f'''You are a premium Kerala travel writer for Kerivaa.

Write a beautiful day-by-day travel narrative for this Kerala itinerary.

Traveller style: {styles}

Itinerary:
{itin_text}{examples_text}

Writing rules:
- One paragraph per day. 4-5 sentences each.
- Be specific to Kerala: backwater canals, spice smells, tea garden mist at dawn, Chinese fishing nets, monsoon light, village life, temple bells, coconut groves.
- Tone: warm and personal, like a knowledgeable local friend writing to you.
- Never use these words: hidden gem, paradise, breathtaking, off the beaten path, magical.
- Start each day with exactly: Day [N] --- [Destination Name]
- For travel days, describe the journey --- what you see from the window, how the landscape changes.
- Return ONLY the narrative text. No headings. No JSON. No markdown.

If examples are provided above, match their warmth and detail level while being original.'''
    
    # Step 5: Generate narrative
    narrative_text = call_llm(prompt, max_tokens=2000)
    
    # Step 6: Parse into days
    paragraphs = [p.strip() for p in narrative_text.split('\n\n') if p.strip()]
    
    # Step 7: Attach narrative to each day
    enriched_days = []
    for i, day in enumerate(structured.get('itinerary', [])):
        day_copy = dict(day)
        day_copy['narrative'] = paragraphs[i] if i < len(paragraphs) else ''
        enriched_days.append(day_copy)
    
    enriched = {**structured, 'itinerary': enriched_days}
    
    # Step 8: Save to vector store for future learning
    try:
        save_itinerary(user_text, enriched, source='user')
    except Exception as e:
        print(f"Vector save warning (non-fatal): {e}")
    
    return enriched