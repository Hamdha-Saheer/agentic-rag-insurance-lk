# agents/domain_router.py
from config import DOMAINS

# Keywords that signal each domain.
# Refined to eliminate intersection conflicts between motor and health claims.
DOMAIN_KEYWORDS = {
    'motor': [
        'vehicle', 'car', 'bike', 'motorcycle', 'three-wheeler', 'tuk',
        'accident', 'motor', 'traffic', 'driving', 'road', 'collision',
        'third party', 'comprehensive', 'motor tariff', 'registration',
        'driving license', 'insurance sticker', 'ocs', 'compensation scheme'
    ],
    'health': [
        'hospital', 'medical', 'agrahara', 'surgery', 'treatment', 'doctor',
        'medicine', 'health', 'clinic', 'admission', 'reimbursement',
        "president's fund", 'critical illness', 'icu', 'dental', 'semi',
        'pension', 'pensioner', 'retired', 'government employee', 'nitf',
        'hospitalization claim', 'medical expense'  # Fixed: Replaced ambiguous 'accident claim'
    ],
    'life': [
        'life', 'death', 'mortality', 'beneficiary', 'premium', 'endowment',
        'whole life', 'term life', 'maturity', 'surrender', 'long-term',
        'long term', 'life cover', 'nominee', 'annuity', 'policy lapse'
    ],
    'general': [
        'travel', 'property', 'home', 'fire', 'student', 'suraksha',
        'flood', 'theft', 'burglary', 'marine', 'cargo', 'baggage',
        'passport', 'abroad', 'overseas', 'product information document'
    ]
}

def route_domain(question: str, user_profile: dict = None) -> str:
    """
    Classify a user question into the best insurance domain using keyword frequencies
    and robust user profile prioritization checks.
    
    Returns: 'motor' | 'life' | 'health' | 'general'
    """
    if not question:
        return 'general'

    question_lower = question.lower()
    scores = {domain: 0 for domain in DOMAINS}

    # Score each domain by keyword matches
    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in question_lower:
                scores[domain] += 1

    # Boost scores using user profile context safely
    if isinstance(user_profile, dict):
        vehicle = user_profile.get('vehicle', 'No Vehicle')
        job = user_profile.get('job', '')

        # Standardize strings to prevent falsy matches
        vehicle_str = str(vehicle).strip().lower() if vehicle else 'no vehicle'
        job_str = str(job).strip().lower() if job else ''

        # Explicitly ignore common empty/null variants
        if vehicle_str not in ('none', 'no vehicle', 'false', ''):
            scores['motor'] += 0.5
        
        # Boost Health if the user is a govt employee or retired (Agrahara alignment)
        if 'government' in job_str or 'retired' in job_str or 'retire' in job_str:
            scores['health'] += 1.0  # Safe deterministic priority boost

    # Return highest-scoring domain; fallback strictly to general domain
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'general'

def get_domain_description(domain: str) -> str:
    """Human-readable label for display in the UI."""
    descriptions = {
        'motor': 'Motor Insurance (Vehicle & Traffic)',
        'health': 'Health Insurance (Agrahara, SEMI, Pension)',
        'life': 'Life & Long-Term Insurance',
        'general': 'General Insurance (Travel, Property, Student)',
    }
    return descriptions.get(domain, 'General Insurance')

# --- Quick Verification Harness ---
if __name__ == "__main__":
    tests = [
        ('What is the Agrahara health coverage limit?', None),
        ('My car was in an accident. How do I claim?', None),
        ('How do I submit an accident claim?', None), # Verify collision is gone
        ('What is the pension benefit for retired employees?', {'job': 'Retired'}),
        ('Does travel insurance cover lost baggage?', None),
    ]
    
    print("--- Running Keyword Routing Verification ---")
    for q, p in tests:
        print(f"[{route_domain(q, p).upper():<7}] -> {q}")