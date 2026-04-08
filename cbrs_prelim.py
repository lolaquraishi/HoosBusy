"""
Content-Based Recommender System (CBRS) for University Event Recommendations
=============================================================================
Milestone Presentation Demo

Input:  User preferences (onboarding + behavioral), Event metadata
Output: Ranked list of recommended events per user (sorted by cosine similarity)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# SECTION 1: FEATURE DEFINITIONS (our shared vocabulary)
# =============================================================================
# Every feature and its possible values. Order matters — it defines vector positions.

FEATURE_SCHEMA = {
    # --- Temporal ---
    "start_time":       ["morning", "afternoon", "evening", "night"],
    "end_time":         ["morning", "afternoon", "evening", "night"],
    "day_of_week":      ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],

    # --- Logistics ---
    "cost":             ["free", "optional", "fully_paid"],
    "setting":          ["indoor", "outdoor", "hybrid"],
    "location":         ["on_grounds", "off_grounds"],
    "audience":         ["undergrad", "grad", "all"],

    # --- Primary Categories (~20) ---
    "primary_category": [
        "athletics", "academic", "professional", "religious",
        "cultural", "social", "physical_health", "mental_health",
        "wellness", "volunteering", "greek_life", "music",
        "art", "outdoors", "food_drink", "gaming_tech",
        "entertainment", "community_activism", "hobbies",
        "travel", "performing_arts", "lgbtq", "science_environment"
    ],

    # --- Subcategories (abbreviated for demo — expand as needed) ---
    "subcategory": [
        # Athletics
        "recreational", "club_sports", "varsity", "intramural",
        "free_play", "pickup_game", "tryout", "watch_party",
        # Academic
        "study_group", "tutoring", "lecture", "panel_discussion",
        "debate", "research_showcase", "hackathon", "workshop",
        "guest_speaker", "trivia",
        # Professional
        "networking", "career_fair", "info_session", "resume_prep",
        "industry_panel", "mentorship", "internship_info", "entrepreneurship",
        # Music
        "dj_set", "live_concert", "a_cappella", "band_performance",
        "open_mic", "karaoke", "jam_session", "choir", "orchestra",
        # Art
        "visual_arts", "painting", "photography", "arts_crafts",
        "poetry", "creative_writing", "improv_comedy", "gallery_opening",
        # Social
        "hangout", "game_night", "party", "speed_friending",
        "mixer", "themed_event", "holiday_social",
        # Outdoors
        "hiking", "camping", "rock_climbing", "kayaking",
        "trail_run", "stargazing",
        # Food
        "free_food", "potluck", "cooking_class", "food_tour",
        "tasting_event", "baking",
        # Wellness / Mental Health
        "yoga", "fitness_class", "mindfulness", "support_group",
        "self_care", "meditation",
        # Entertainment
        "movie_screening", "comedy_show", "drag_show", "escape_room",
        # Gaming & Tech
        "video_games", "board_games", "tabletop_rpg", "esports",
        "coding_workshop", "ctf_security",
        # Volunteering
        "community_service", "fundraiser", "food_bank",
        "environmental_cleanup", "awareness_campaign",
    ],

    # --- Vibe / Latent Features ---
    "energy_level":      ["low", "medium", "high"],
    "social_intensity":  ["solo_friendly", "small_group", "large_group", "crowd"],
    "commitment_level":  ["drop_in", "recurring", "multi_session"],
    "skill_barrier":     ["none", "beginner", "intermediate", "advanced"],
    "mood":              ["chill", "energetic", "reflective", "competitive", "creative", "supportive"],
}

# Precompute the vector index mapping: feature_name -> {value -> index}
def build_index_map(schema: dict) -> tuple[dict, int]:
    """
    Returns:
        index_map: {"start_time": {"morning": 0, "afternoon": 1, ...}, ...}
        total_dims: total length of the feature vector
    """
    index_map = {}
    offset = 0
    for feature_name, values in schema.items():
        index_map[feature_name] = {v: offset + i for i, v in enumerate(values)}
        offset += len(values)
    return index_map, offset

INDEX_MAP, VECTOR_DIM = build_index_map(FEATURE_SCHEMA)
print(f"Vector dimensionality: {VECTOR_DIM}")


# =============================================================================
# SECTION 2: EVENT REPRESENTATION
# =============================================================================

@dataclass
class Event:
    """A single event with all its features."""
    name: str
    # Temporal
    start_time: str = "evening"
    end_time: str = "night"
    day_of_week: list[str] = field(default_factory=lambda: ["fri"])
    # Logistics
    cost: str = "free"
    setting: str = "indoor"
    location: str = "on_grounds"
    audience: str = "all"
    # Categories (can have multiple)
    primary_category: list[str] = field(default_factory=list)
    subcategory: list[str] = field(default_factory=list)
    # Vibe
    energy_level: str = "medium"
    social_intensity: str = "large_group"
    commitment_level: str = "drop_in"
    skill_barrier: str = "none"
    mood: list[str] = field(default_factory=lambda: ["chill"])


# Initially setting all features to 0.0
# Encode each feature based on the INDEX_MAP, setting the corresponding index to 1.0 if the value is present
def encode_event(event: Event) -> np.ndarray:
    """
    Convert an Event into a binary feature vector.

    One-hot for single-value features, multi-hot for list features.
    """
    vec = np.zeros(VECTOR_DIM)
    # Single-value features (one-hot)
    for feature_name in ["start_time", "end_time", "cost", "setting",
                         "location", "audience", "energy_level",
                         "social_intensity", "commitment_level", "skill_barrier"]:
        value = getattr(event, feature_name)
        if value in INDEX_MAP[feature_name]:
            vec[INDEX_MAP[feature_name][value]] = 1.0

    # Multi-value features (multi-hot)
    for feature_name in ["day_of_week", "primary_category", "subcategory", "mood"]:
        values = getattr(event, feature_name)
        for v in values:
            if v in INDEX_MAP[feature_name]:
                vec[INDEX_MAP[feature_name][v]] = 1.0

    return vec


# =============================================================================
# SECTION 3: USER PROFILE
# =============================================================================

@dataclass
class UserProfile:
    """Tracks a user's evolving preference vector."""
    name: str
    vector: np.ndarray = field(default_factory=lambda: np.zeros(VECTOR_DIM))
    interaction_count: int = 0


# Initialize the user profile vector based on onboarding selections.
# For each selected feature, set the corresponding index in the vector to 1.0.
def initialize_from_onboarding(
    user: UserProfile,
    selected_categories: list[str],
    selected_subcategories: list[str],
    selected_moods: list[str],
    preferred_times: Optional[list[str]] = None,
    preferred_days: Optional[list[str]] = None,
    preferred_energy: Optional[str] = None,
    preferred_social: Optional[str] = None,
) -> None:
    """
    Build the initial user profile vector from onboarding selections.
    Sets selected features to 1.0 in the user's vector.
    """
    user.vector = np.zeros(VECTOR_DIM)

    # Category preferences
    for cat in selected_categories:
        if cat in INDEX_MAP["primary_category"]:
            user.vector[INDEX_MAP["primary_category"][cat]] = 1.0

    for sub in selected_subcategories:
        if sub in INDEX_MAP["subcategory"]:
            user.vector[INDEX_MAP["subcategory"][sub]] = 1.0

    # Mood/vibe preferences
    for mood in selected_moods:
        if mood in INDEX_MAP["mood"]:
            user.vector[INDEX_MAP["mood"][mood]] = 1.0

    # Optional: time preferences
    if preferred_times:
        for t in preferred_times:
            if t in INDEX_MAP["start_time"]:
                user.vector[INDEX_MAP["start_time"][t]] = 1.0

    if preferred_days:
        for d in preferred_days:
            if d in INDEX_MAP["day_of_week"]:
                user.vector[INDEX_MAP["day_of_week"][d]] = 1.0

    if preferred_energy and preferred_energy in INDEX_MAP["energy_level"]:
        user.vector[INDEX_MAP["energy_level"][preferred_energy]] = 1.0

    if preferred_social and preferred_social in INDEX_MAP["social_intensity"]:
        user.vector[INDEX_MAP["social_intensity"][preferred_social]] = 1.0

# Update the user profile vector based on an interaction with an event.
# Uses an exponential moving average (decay) to blend the event vector into the user profile (essentially updating the user's preferences based on their behavior).
def update_from_interaction(
    user: UserProfile,
    event_vector: np.ndarray,
    decay: float = 0.9
) -> None:
    """
    Update the user profile after they interact with (attend/like) an event.

    Uses exponential moving average:
        new_profile = decay * old_profile + (1 - decay) * event_vector

    Higher decay = slower change (old preferences persist).
    Lower decay  = faster adaptation to new interests.
    """
    user.vector = decay * user.vector + (1 - decay) * event_vector
    user.interaction_count += 1


# =============================================================================
# SECTION 4: RECOMMENDATION ENGINE
# =============================================================================

# We compare the user to an event by 
# 1. Dot Product: summing the dot product of their features (which counts the number of shared features)
# 2. Normalize: We divide by the product of the vector sizes to get a cosine similarity score between 0 and 1. This way, we're measuring similarity in preference over size of the profile (since an event could have many features but only a few match the user's preferences)
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))

# For each event, we compute the cosine similarity between the user's preference vector and the event's feature vector. We then rank all events by their similarity score and return the top-N recommendations.
def recommend_events(
    user: UserProfile,
    events: list[Event],
    attended_events: list[str],
    top_n: int = 5
) -> list[tuple[str, float]]:
    """
    Score all events against the user profile and return top-N recommendations.

    Returns:
        List of (event_name, similarity_score) tuples, sorted descending.
    """
    # Encode all events into a matrix (num_events x VECTOR_DIM)
    event_list = []
    for e in events:
        if e.name not in attended_events:
           event_list.append(encode_event(e))

    #event_vectors = np.array([encode_event(e) for e in events])
    event_vectors = np.array(event_list)

    # Compute all similarities at once via matrix multiplication
    user_norm = np.linalg.norm(user.vector)
    if user_norm == 0:
        return [(e.name, 0.0) for e in events[:top_n]]

    event_norms = np.linalg.norm(event_vectors, axis=1)
    # Avoid division by zero
    event_norms[event_norms == 0] = 1.0

    scores = event_vectors @ user.vector / (event_norms * user_norm)

    # Rank and return top N
    ranked_indices = np.argsort(scores)[::-1][:top_n]
    return [(events[i].name, round(scores[i], 4)) for i in ranked_indices]


# =============================================================================
# SECTION 5: DEMO — SAMPLE EVENTS + MOCK USER
# =============================================================================

def create_sample_events() -> list[Event]:
    """Create a diverse set of sample events for demonstration."""
    return [
        Event(
            name="UPC Open Mic Night",
            start_time="evening", end_time="night",
            day_of_week=["fri"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["music", "entertainment"],
            subcategory=["open_mic"],
            energy_level="medium", social_intensity="large_group",
            commitment_level="drop_in", skill_barrier="none",
            mood=["energetic", "creative"],
        ),
        Event(
            name="Humpback Rock Hike",
            start_time="morning", end_time="afternoon",
            day_of_week=["sat"],
            cost="free", setting="outdoor", location="off_grounds", audience="all",
            primary_category=["outdoors", "physical_health"],
            subcategory=["hiking"],
            energy_level="high", social_intensity="small_group",
            commitment_level="drop_in", skill_barrier="beginner",
            mood=["energetic", "chill"],
        ),
        Event(
            name="AI Research Panel",
            start_time="afternoon", end_time="evening",
            day_of_week=["wed"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["academic", "professional"],
            subcategory=["panel_discussion", "research_showcase"],
            energy_level="low", social_intensity="large_group",
            commitment_level="drop_in", skill_barrier="none",
            mood=["reflective"],
        ),
        Event(
            name="Pickup Basketball",
            start_time="afternoon", end_time="evening",
            day_of_week=["tue", "thu"],
            cost="free", setting="outdoor", location="on_grounds", audience="all",
            primary_category=["athletics"],
            subcategory=["pickup_game", "free_play"],
            energy_level="high", social_intensity="small_group",
            commitment_level="drop_in", skill_barrier="beginner",
            mood=["competitive", "energetic"],
        ),
        Event(
            name="Pottery Workshop",
            start_time="afternoon", end_time="evening",
            day_of_week=["sat"],
            cost="optional", setting="indoor", location="on_grounds", audience="all",
            primary_category=["art", "hobbies"],
            subcategory=["arts_crafts"],
            energy_level="low", social_intensity="small_group",
            commitment_level="drop_in", skill_barrier="none",
            mood=["creative", "chill"],
        ),
        Event(
            name="Career Fair",
            start_time="morning", end_time="afternoon",
            day_of_week=["thu"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["professional"],
            subcategory=["career_fair", "networking"],
            energy_level="medium", social_intensity="crowd",
            commitment_level="drop_in", skill_barrier="none",
            mood=["competitive"],
        ),
        Event(
            name="Yoga on the Lawn",
            start_time="morning", end_time="morning",
            day_of_week=["mon", "wed", "fri"],
            cost="free", setting="outdoor", location="on_grounds", audience="all",
            primary_category=["physical_health", "wellness"],
            subcategory=["yoga", "self_care"],
            energy_level="low", social_intensity="small_group",
            commitment_level="recurring", skill_barrier="none",
            mood=["chill", "reflective"],
        ),
        Event(
            name="HooHacks Hackathon",
            start_time="morning", end_time="night",
            day_of_week=["sat", "sun"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["academic", "gaming_tech"],
            subcategory=["hackathon", "coding_workshop"],
            energy_level="high", social_intensity="small_group",
            commitment_level="multi_session", skill_barrier="intermediate",
            mood=["competitive", "creative"],
        ),
        Event(
            name="Board Game Night",
            start_time="evening", end_time="night",
            day_of_week=["fri"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["social", "gaming_tech"],
            subcategory=["board_games", "game_night"],
            energy_level="low", social_intensity="small_group",
            commitment_level="drop_in", skill_barrier="none",
            mood=["chill", "competitive"],
        ),
        Event(
            name="Volunteering at Food Bank",
            start_time="morning", end_time="afternoon",
            day_of_week=["sat"],
            cost="free", setting="indoor", location="off_grounds", audience="all",
            primary_category=["volunteering"],
            subcategory=["food_bank", "community_service"],
            energy_level="medium", social_intensity="small_group",
            commitment_level="drop_in", skill_barrier="none",
            mood=["supportive"],
        ),
        Event(
            name="Jazz Ensemble Concert",
            start_time="evening", end_time="night",
            day_of_week=["sat"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["music", "performing_arts"],
            subcategory=["orchestra", "live_concert"],
            energy_level="medium", social_intensity="crowd",
            commitment_level="drop_in", skill_barrier="none",
            mood=["chill", "creative"],
        ),
        Event(
            name="CTF Cybersecurity Competition",
            start_time="evening", end_time="night",
            day_of_week=["fri"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["gaming_tech", "academic"],
            subcategory=["ctf_security", "coding_workshop"],
            energy_level="high", social_intensity="small_group",
            commitment_level="drop_in", skill_barrier="intermediate",
            mood=["competitive", "creative"],
        ),
        Event(
            name="Springfest",
            start_time="evening", end_time="night",
            day_of_week=["sat"],
            cost="free", setting="indoor", location="on_grounds", audience="all",
            primary_category=["music", "entertainment", "social"],
            subcategory=["live_concert"],
            energy_level="high", social_intensity="crowd",
            commitment_level="drop_in", skill_barrier="none",
            mood=["energetic"],
        ),
    ]


def run_demo():
    """Full demonstration of the CBRS pipeline."""

    print("=" * 60)
    print("CBRS EVENT RECOMMENDER — DEMO")
    print("=" * 60)

    # --- Step 1: Load events ---
    events = create_sample_events()
    print(f"\nLoaded {len(events)} events")
    print(f"Vector dimensionality: {VECTOR_DIM}\n")

    # --- Step 2: Simulate onboarding for a user ---
    user = UserProfile(name="Outdoorsy Music Fan")
    initialize_from_onboarding(
        user,
        selected_categories=["music", "art", "outdoors"],
        selected_subcategories=["open_mic", "jam_session", "hiking", "arts_crafts", "poetry"],
        selected_moods=["creative", "energetic"],
        preferred_times=["evening"],
        preferred_days=["fri", "sat"],
        preferred_energy="medium",
        preferred_social="crowd",
    )
    print(f"User '{user.name}' onboarded.")
    print(f"  Categories: music, art, outdoors")
    print(f"  Moods: creative, chill")
    print(f"  Non-zero dims in profile: {int(np.count_nonzero(user.vector))}")

    # --- Step 3: Get initial recommendations (cold start, onboarding only) ---
    print("\n--- INITIAL RECOMMENDATIONS (onboarding only) ---")
    recs = recommend_events(user, events, attended_events=[],top_n=5)
    for rank, (name, score) in enumerate(recs, 1):
        print(f"  {rank}. {name:40s}  score: {score:.4f}")

    # --- Step 4: Simulate user attending some events ---
    print("\n--- SIMULATING USER BEHAVIOR ---")
    attended_events = ["Humpback Rock Hike", "Jazz Ensemble Concert", "Springfest"]
    for event_name in attended_events:
        event = next(e for e in events if e.name == event_name)
        event_vec = encode_event(event)
        update_from_interaction(user, event_vec, decay=0.85)
        print(f"  Attended: {event_name}")

    # --- Step 5: Get updated recommendations ---
    print(f"\n--- UPDATED RECOMMENDATIONS (after {len(attended_events)} interactions) ---")
    recs = recommend_events(user, events, attended_events, top_n=5)
    for rank, (name, score) in enumerate(recs, 1):
        print(f"  {rank}. {name:40s}  score: {score:.4f}")
'''
    # --- Step 6: Show the vector for one event (for presentation) ---
    print("\n--- SAMPLE EVENT VECTOR ---")
    sample = events[0]  # UPC Open Mic Night
    vec = encode_event(sample)
    print(f"Event: {sample.name}")
    print(f"Vector shape: ({VECTOR_DIM},)")
    print(f"Non-zero positions: {list(np.nonzero(vec)[0])}")
    print(f"Non-zero count: {int(np.count_nonzero(vec))}")
'''

if __name__ == "__main__":
    run_demo()