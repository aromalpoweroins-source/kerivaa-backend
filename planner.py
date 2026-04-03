from datetime import date
from database import (
    get_all_destinations,
    get_experiences_for_destination,
    get_travel_time
)

# ── HELPERS ──────────────────────────────────────────────────────

def parse_tags(tag_string: str) -> list:
    """Convert 'romantic,slow,water' to ['romantic','slow','water']."""
    if not tag_string:
        return []
    return [t.strip().lower() for t in tag_string.split(',')]

def parse_ideal_days(val) -> float:
    """Convert '2-3' or '2' or 2 to a midpoint float."""
    val = str(val).strip()
    if '-' in val:
        parts = val.split('-')
        return (float(parts[0]) + float(parts[1])) / 2
    try:
        return float(val)
    except:
        return 2.0

def count_matching_experiences(dest_id: int,
                               user_styles: list) -> int:
    """How many experiences at this destination match user styles?"""
    experiences = get_experiences_for_destination(dest_id)
    count = 0
    for exp in experiences:
        exp_tags = parse_tags(exp.get('tags', ''))
        if any(s in exp_tags for s in user_styles):
            count += 1
    return count

def score_destination(dest: dict, user_styles: list,
                      user_budget: str, user_month: int,
                      matching_exp_count: int) -> float:
    """
    Score how well a destination fits the user request.
    Returns 0.0 to 1.0. Higher is better.
    """
    dest_tags = parse_tags(dest.get('tags', ''))
    matches = sum(1 for s in user_styles if s in dest_tags)
    tag_score = matches / len(user_styles) if user_styles else 0
    
    exp_score = min(matching_exp_count / 5, 1.0)
    
    season_score = 1.0
    if user_month and dest.get('best_season'):
        good_months = []
        for m in dest['best_season'].split(','):
            m = m.strip()
            if m.isdigit():
                good_months.append(int(m))
        if good_months and user_month not in good_months:
            season_score = 0.5
    
    budget_score = 1.0 if dest.get('budget_level') == user_budget else 0.7
    
    final = (
        tag_score * 0.45 +
        exp_score * 0.30 +
        season_score * 0.15 +
        budget_score * 0.10
    )
    return round(final, 3)

def pick_experiences_for_day(dest_id: int, user_styles: list,
                             used_exp_ids: set,
                             max_hours: float = 8,
                             max_count: int = 3,
                             pace: str = 'moderate') -> list:
    """
    Pick the best experiences for one day at a destination.
    Respects energy balance, time limits, and avoids repeats.
    """
    all_exp = get_experiences_for_destination(dest_id)
    candidates = []
    for exp in all_exp:
        if exp['id'] in used_exp_ids:
            continue
        exp_tags = parse_tags(exp.get('tags', ''))
        if any(s in exp_tags for s in user_styles):
            candidates.append(exp)
    
    def sort_key(e):
        time_order = {
            'morning': 0, 'any': 1,
            'afternoon': 2, 'evening': 3
        }
        time_score = time_order.get(
            e.get('best_time_of_day', 'any'), 1
        )
        pop_score = float(e.get('popularity_score', 1.0))
        return (time_score, -pop_score)
    
    candidates.sort(key=sort_key)
    
    if pace == 'slow':
        max_count = min(max_count, 2)
        max_hours = 6
    elif pace == 'packed':
        max_hours = 10
    
    selected = []
    total_hours = 0
    high_energy_count = 0
    seen_categories = set()
    
    for exp in candidates:
        if len(selected) >= max_count:
            break
        
        exp_hours = float(exp.get('duration_hours', 2))
        exp_energy = exp.get('energy_level', 'low')
        exp_tags = parse_tags(exp.get('tags', ''))
        
        if total_hours + exp_hours > max_hours:
            continue
        
        if exp_energy == 'high' and high_energy_count >= 1:
            continue
        
        main_tag = exp_tags[0] if exp_tags else 'general'
        if main_tag in seen_categories:
            continue
        
        seen_categories.add(main_tag)
        selected.append(exp)
        total_hours += exp_hours
        if exp_energy == 'high':
            high_energy_count += 1
        used_exp_ids.add(exp['id'])
    
    return selected

def format_experience(exp: dict) -> dict:
    """Clean an experience row for the API response."""
    return {
        'id': exp['id'],
        'name': exp['name'],
        'duration_hours': float(exp.get('duration_hours', 2)),
        'energy_level': exp.get('energy_level', 'low'),
        'tags': exp.get('tags', ''),
        'best_time_of_day': exp.get('best_time_of_day', 'any'),
        'hidden_gem': exp.get('hidden_gem', False)
    }

# ── MAIN PLANNING FUNCTION ───────────────────────────────────────

def plan_trip(duration: int, budget: str,
              styles: list, month: int = None) -> dict:
    """
    Main planning engine.
    Takes trip parameters and returns a structured itinerary dict.
    No AI used here --- pure logic.
    """
    if month is None:
        month = date.today().month
    
    pace = 'slow' if 'slow' in styles else (
        'packed' if 'adventure' in styles else 'moderate')
    
    # ── Decision 1: How many destinations ────────────────────────
    if duration <= 3:
        max_destinations = 1
    elif duration <= 6:
        max_destinations = 2
    elif duration <= 10:
        max_destinations = 3
    else:
        max_destinations = min(4, duration // 3)
    
    # ── Decision 2: Which destinations ───────────────────────────
    all_dests = get_all_destinations(budget)
    scored = []
    for dest in all_dests:
        exp_count = count_matching_experiences(dest['id'], styles)
        if exp_count < 2:
            continue
        score = score_destination(
            dest, styles, budget, month, exp_count
        )
        scored.append((score, dest))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    
    selected_dests = []
    for score, dest in scored:
        if len(selected_dests) == 0:
            selected_dests.append(dest)
            continue
        if len(selected_dests) >= max_destinations:
            break
        
        last = selected_dests[-1]
        travel_hrs = get_travel_time(last['id'], dest['id'])
        if travel_hrs <= 4.5:
            selected_dests.append(dest)
    
    if not selected_dests:
        return {'error': 'No matching destinations found'}
    
    # ── Decision 3: Days per destination ─────────────────────────
    num_moves = len(selected_dests) - 1
    experience_days = max(1, duration - num_moves)
    
    total_weight = sum(
        parse_ideal_days(d.get('ideal_days', '2'))
        for d in selected_dests
    )
    
    days_per_dest = []
    remaining = experience_days
    for i, dest in enumerate(selected_dests):
        if i == len(selected_dests) - 1:
            days = max(1, remaining)
        else:
            weight = parse_ideal_days(dest.get('ideal_days', '2'))
            raw = (weight / total_weight) * experience_days
            days = max(1, round(raw))
        days_per_dest.append(days)
        remaining -= days
    
    # ── Decisions 4 + 5: Build day by day ────────────────────────
    itinerary = []
    day_number = 1
    used_exp_ids = set()
    
    for dest_index, dest in enumerate(selected_dests):
        dest_days = days_per_dest[dest_index]
        
        for day_in_dest in range(dest_days):
            is_travel_arrival = (
                dest_index > 0 and day_in_dest == 0
            )
            
            if is_travel_arrival:
                prev = selected_dests[dest_index - 1]
                travel_hrs = get_travel_time(
                    prev['id'], dest['id']
                )
                
                if travel_hrs <= 2.5:
                    exps = pick_experiences_for_day(
                        dest['id'], styles, used_exp_ids,
                        max_hours=4, max_count=1, pace='slow'
                    )
                    day_type = 'travel_and_arrive'
                else:
                    exps = []
                    day_type = 'travel'
                
                itinerary.append({
                    'day': day_number,
                    'type': day_type,
                    'destination': dest['name'],
                    'travel_from': prev['name'],
                    'travel_hours': travel_hrs,
                    'experiences': [format_experience(e) for e in exps]
                })
            else:
                exps = pick_experiences_for_day(
                    dest['id'], styles, used_exp_ids,
                    max_hours=8, max_count=3, pace=pace
                )
                itinerary.append({
                    'day': day_number,
                    'type': 'experience',
                    'destination': dest['name'],
                    'region': dest.get('region', ''),
                    'experiences': [format_experience(e) for e in exps]
                })
            
            day_number += 1
    
    return {
        'total_days': duration,
        'budget': budget,
        'styles': styles,
        'pace': pace,
        'destinations': [d['name'] for d in selected_dests],
        'itinerary': itinerary
    }