import json
from llm import call_llm

def extract_intent(user_text: str) -> dict:
    """
    Takes plain English trip description.
    Returns structured dict with duration, budget, styles etc.
    Calls Ollama via llm.py.
    """
    prompt = f"""
You are a travel intent extractor for Kerivaa, a Kerala trip planner.
Extract the user intent from the text below.
Return ONLY a valid JSON object. No explanation. No markdown.
No text before or after the JSON. Just the raw JSON.

User text: "{user_text}"

Return exactly this JSON structure:
{{
    "duration_days": <integer, default 5 if unclear>,
    "budget": <"low" or "mid" or "luxury">,
    "styles": <list of strings from: romantic, adventure, family,
        offbeat, culture, nature, pilgrimage, slow, food,
        heritage, wildlife, beach, water, local>,
    "group_type": <"couple" or "family" or "solo" or "friends">,
    "pace": <"slow" or "moderate" or "packed">
}}

Rules:
- honeymoon or wife or partner or anniversary = romantic, couple
- kids or children or family = family
- not touristy or hidden or local = offbeat
- backwaters = water, slow, romantic
- trek or hike or adventure = adventure
- week = 7 days, few days = 4, weekend = 2
- budget or cheap or low cost = low
- luxury or premium or high end = luxury
- Return ONLY valid JSON. Nothing else.
"""
    
    try:
        raw = call_llm(prompt, max_tokens=400, expect_json=True)
        raw = raw.strip()
    except (ConnectionError, TimeoutError, Exception) as e:
        print(f'LLM call failed: {e}')
        print('Falling back to keyword-based extraction...')
        return _keyword_fallback(user_text)
    
    if raw.startswith('```'):
        lines = raw.split('\n')
        raw = '\n'.join(
            l for l in lines
            if not l.startswith('```')
        )
    
    try:
        result = json.loads(raw)
        
        if 'duration_days' not in result:
            result['duration_days'] = 5
        if 'budget' not in result:
            result['budget'] = 'mid'
        if 'styles' not in result or not result['styles']:
            result['styles'] = ['nature']
        if 'group_type' not in result:
            result['group_type'] = 'couple'
        if 'pace' not in result:
            result['pace'] = 'moderate'
        
        # Post-process: enforce keyword rules the LLM may miss
        result = _post_process(result, user_text)
        
        return result
    
    except json.JSONDecodeError as e:
        print(f'Intent JSON parse failed: {e}')
        print(f'Raw output was: {raw[:200]}')
        return _keyword_fallback(user_text)


def _post_process(result: dict, user_text: str) -> dict:
    """
    Apply deterministic keyword rules on top of LLM output
    to correct common misinterpretations.
    """
    text_lower = user_text.lower()
    styles = result.get('styles', [])
    
    # backwaters → must include water, slow, romantic
    if 'backwater' in text_lower or 'backwaters' in text_lower:
        for s in ['water', 'slow', 'romantic']:
            if s not in styles:
                styles.append(s)
    
    # honeymoon → romantic, couple, and default budget stays mid
    if 'honeymoon' in text_lower:
        if 'romantic' not in styles:
            styles.append('romantic')
        result['group_type'] = 'couple'
        # honeymoon doesn't imply luxury unless explicitly stated
        if not any(w in text_lower for w in ['luxury', 'premium', 'high end']):
            result['budget'] = 'mid'
        result['pace'] = 'slow'
    
    # adventure → packed pace
    if 'adventure' in text_lower or 'trek' in text_lower:
        if 'adventure' not in styles:
            styles.append('adventure')
        result['pace'] = 'packed'
    
    # family keywords
    if any(w in text_lower for w in ['family', 'kids', 'children']):
        if 'family' not in styles:
            styles.append('family')
        result['group_type'] = 'family'
    
    result['styles'] = styles
    return result


def _keyword_fallback(text: str) -> dict:
    """
    Rule-based fallback when LLM is unavailable.
    Parses keywords from the user text directly.
    """
    text_lower = text.lower()
    
    # Duration
    import re
    duration = 5
    num_match = re.search(r'(\d+)\s*day', text_lower)
    if num_match:
        duration = int(num_match.group(1))
    elif 'week' in text_lower:
        duration = 7
    elif 'weekend' in text_lower:
        duration = 2
    elif 'few days' in text_lower:
        duration = 4
    
    # Budget
    budget = 'mid'
    if any(w in text_lower for w in ['budget', 'cheap', 'low cost', 'affordable']):
        budget = 'low'
    elif any(w in text_lower for w in ['luxury', 'premium', 'high end', 'luxurious']):
        budget = 'luxury'
    
    # Styles
    style_keywords = {
        'romantic': ['romantic', 'honeymoon', 'anniversary', 'couple'],
        'adventure': ['adventure', 'trek', 'hike', 'trekking'],
        'family': ['family', 'kids', 'children'],
        'offbeat': ['offbeat', 'hidden', 'not touristy', 'local'],
        'culture': ['culture', 'cultural', 'temple', 'heritage'],
        'nature': ['nature', 'scenic', 'green', 'hills'],
        'slow': ['slow', 'relaxing', 'peaceful', 'calm', 'backwater'],
        'water': ['backwater', 'houseboat', 'lake', 'water'],
        'food': ['food', 'cuisine', 'culinary'],
        'wildlife': ['wildlife', 'safari', 'animals'],
        'beach': ['beach', 'coast', 'sea'],
        'pilgrimage': ['pilgrimage', 'temple', 'spiritual'],
    }
    styles = []
    for style, keywords in style_keywords.items():
        if any(kw in text_lower for kw in keywords):
            styles.append(style)
    if not styles:
        styles = ['nature']
    
    # Group type
    group_type = 'couple'
    if any(w in text_lower for w in ['family', 'kids', 'children']):
        group_type = 'family'
    elif any(w in text_lower for w in ['solo', 'alone', 'myself']):
        group_type = 'solo'
    elif any(w in text_lower for w in ['friends', 'group', 'gang']):
        group_type = 'friends'
    elif any(w in text_lower for w in ['honeymoon', 'romantic', 'couple', 'wife', 'partner']):
        group_type = 'couple'
    
    # Pace
    pace = 'moderate'
    if any(w in text_lower for w in ['slow', 'relaxing', 'peaceful', 'calm', 'backwater', 'honeymoon']):
        pace = 'slow'
    elif any(w in text_lower for w in ['packed', 'adventure', 'action']):
        pace = 'packed'
    
    return {
        'duration_days': duration,
        'budget': budget,
        'styles': styles,
        'group_type': group_type,
        'pace': pace
    }