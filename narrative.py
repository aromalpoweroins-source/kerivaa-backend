from llm import call_llm

def generate_narrative(itinerary: dict, styles: list) -> dict:
    """
    Takes structured itinerary dict.
    Adds a narrative paragraph to each day.
    Returns enriched itinerary with narrative field on each day.
    """
    style_string = ', '.join(styles)
    
    itinerary_text = ''
    for day in itinerary.get('itinerary', []):
        exp_list = ', '.join(
            [e['name'] for e in day.get('experiences', [])]
        )
        dest = day.get('destination', '')
        day_type = day.get('type', 'experience')
        
        if day_type == 'travel':
            from_dest = day.get('travel_from', '')
            hrs = day.get('travel_hours', 0)
            itinerary_text += (
                f"Day {day['day']}: Travel from {from_dest} "
                f"to {dest} ({hrs} hours)\n"
            )
        else:
            itinerary_text += (
                f"Day {day['day']} in {dest}: {exp_list}\n"
            )
    
    prompt = f"""You are a premium Kerala travel writer for Kerivaa.
Write a beautiful day-by-day travel narrative for this Kerala itinerary.
The traveller style is: {style_string}

Itinerary:
{itinerary_text}

Writing rules:
- One paragraph per day. 4-5 sentences each.
- Be specific to Kerala: mention backwater canals, spice smells,
  tea garden mist at dawn, Chinese fishing nets, monsoon light,
  village life, temple bells, coconut groves.
- Tone: warm and personal, like a knowledgeable local friend writing to you.
- Do NOT use these words: hidden gem, paradise, breathtaking,
  off the beaten path, gem, magical.
- Start each day with exactly: Day [N] --- [Destination Name]
- For travel days write about the journey itself --- what you see
  from the window, how the landscape changes.
- Return ONLY the narrative text. No headings. No JSON. No markdown.
"""
    
    narrative_text = call_llm(prompt, max_tokens=2000)
    
    paragraphs = [p.strip() for p in narrative_text.split('\n\n')
                  if p.strip()]
    
    enriched_days = []
    for i, day in enumerate(itinerary.get('itinerary', [])):
        day_copy = dict(day)
        day_copy['narrative'] = (
            paragraphs[i] if i < len(paragraphs) else ''
        )
        enriched_days.append(day_copy)
    
    return {**itinerary, 'itinerary': enriched_days}