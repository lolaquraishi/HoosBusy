"""
cbrs.py -- Content-Based Recommender System core logic.

Flow: load schema/events -> build index maps -> encode events as vectors ->
create user profile -> update profile from interactions -> score and rank events.
"""

import json
import numpy as np


# ── Load data ──────────────────────────────────────────────────────────────────

def load_schema(path):
    with open(path, "r") as f:
        return json.load(f)

def load_events(path):
    with open(path, "r") as f:
        return json.load(f)

def load_archetypes(path):
    with open(path, "r") as f:
        return json.load(f)


# ── Build index maps ───────────────────────────────────────────────────────────
# Each feature's values are assigned consecutive positions in the vector.
# e.g. index_map["mood"]["chill"] = 3 means dimension 3 represents "chill" mood.
#
# Features are split into two groups:
#   Interest: what kind of event it is (category, subcategory)
#   Context:  when/where/how it happens (time, day, cost, setting, etc.)

INTEREST_FEATURE_KEYS = ["category", "subcategory"]
CONTEXT_FEATURE_KEYS  = ["start_time", "day_of_week", "cost", "setting", "location",
                          "social_intensity", "commitment_level", "mood", "energy_level", "skill_barrier"]

# Interest similarity carries most of the final score; context is a smaller nudge.
INTEREST_WEIGHT = 0.75
CONTEXT_WEIGHT  = 0.25

# How much each interaction type shifts the user profile vector.
INTERACTION_WEIGHTS = {
    "interested":  0.30,
    "skip":       -0.15,
    "attended":    0.5,
    "remove":    -0.80
}


def get_categories_and_subcategories(schema):
    """Pull the ordered category and subcategory lists out of the schema hierarchy."""
    hierarchy = schema["category_hierarchy"]
    categories = list(hierarchy.keys())
    seen, subcategories = set(), []
    for subs in hierarchy.values():
        for s in subs:
            if s not in seen:
                seen.add(s)
                subcategories.append(s)
    return categories, subcategories

def get_visible_subcategories(schema, selected_categories):
    """Return subcategories that belong to the currently selected categories."""
    hierarchy = schema["category_hierarchy"]
    visible = []
    for cat in selected_categories:
        for sub in hierarchy.get(cat, []):
            if sub not in visible:
                visible.append(sub)
    return visible

def build_index_map(feature_keys, schema, categories, subcategories):
    """
    Assign a vector position to every value of every feature in feature_keys.
    Returns (index_map, total vector length).
    """
    index_map = {}
    position  = 0
    for key in feature_keys:
        if key == "category":
            values = categories
        elif key == "subcategory":
            values = subcategories
        else:
            values = schema[key]
        index_map[key] = {v: position + i for i, v in enumerate(values)}
        position += len(values)
    return index_map, position


def setup_vector_space(schema):
    """
    Build both index maps (interest and context) from the schema.
    Call once at startup; pass the results everywhere that needs them.
    """
    categories, subcategories = get_categories_and_subcategories(schema)
    interest_index, interest_dim = build_index_map(
        INTEREST_FEATURE_KEYS, schema, categories, subcategories)
    context_index, context_dim = build_index_map(
        CONTEXT_FEATURE_KEYS, schema, categories, subcategories)
    return interest_index, interest_dim, context_index, context_dim


# ── Event encoding ─────────────────────────────────────────────────────────────
# Turns a raw event dict into two numpy vectors (interest, context).
# Categories get weight 1.0; subcategories get 1.5 so specific tags dominate.

CATEGORY_ENCODE_WEIGHT    = 1
SUBCATEGORY_ENCODE_WEIGHT = 1.5


def encode_event(event, interest_index, interest_dim, context_index, context_dim):
    """Convert an event dict into (interest_vector, context_vector)."""
    iv = np.zeros(interest_dim)
    cv = np.zeros(context_dim)

    for cat in event.get("primary_category", []):
        if cat in interest_index["category"]:
            iv[interest_index["category"][cat]] = CATEGORY_ENCODE_WEIGHT

    for sub in event.get("subcategory", []):
        if sub in interest_index["subcategory"]:
            iv[interest_index["subcategory"][sub]] = SUBCATEGORY_ENCODE_WEIGHT

    for m in event.get("mood", []):
        if m in context_index["mood"]:
            cv[context_index["mood"][m]] = 1.0

    energy = event.get("energy_level", "")
    if energy in context_index["energy_level"]:
        cv[context_index["energy_level"][energy]] = 1.0

    skill = event.get("skill_barrier", "")
    if skill in context_index["skill_barrier"]:
        cv[context_index["skill_barrier"][skill]] = 1.0

    start = event.get("start_time", "")
    if start in context_index["start_time"]:
        cv[context_index["start_time"][start]] = 1.0

    for day in event.get("day_of_week", []):
        if day in context_index["day_of_week"]:
            cv[context_index["day_of_week"][day]] = 1.0

    cost = event.get("cost", "")
    if cost in context_index["cost"]:
        cv[context_index["cost"][cost]] = 1.0

    setting = event.get("setting", "")
    if setting in context_index["setting"]:
        cv[context_index["setting"][setting]] = 1.0

    location = event.get("location", "")
    if location in context_index["location"]:
        cv[context_index["location"][location]] = 1.0

    social = event.get("social_intensity", "")
    if social in context_index["social_intensity"]:
        cv[context_index["social_intensity"][social]] = 1.0

    commitment = event.get("commitment_level", "")
    if commitment in context_index["commitment_level"]:
        cv[context_index["commitment_level"][commitment]] = 1.0

    return iv, cv


# ── User profile ───────────────────────────────────────────────────────────────
# A profile is just a dict holding the user's interest and context vectors
# plus bookkeeping (interaction count, attended/interested event sets).
#
# Onboarding weights:
#   Selected category                -> 1.0
#   Subcategories implied by that    -> 0.35  (implicit interest)
#   Explicitly selected subcategory  -> 1.5   (strongest signal)
# max() ensures an explicit pick always beats the implicit fallback.

def make_profile(name, interest_dim, context_dim):
    return {
        "name":              name,
        "interest_vec":      np.zeros(interest_dim),
        "context_vec":       np.zeros(context_dim),
        "interaction_count": 0,
        "attended_ids":      set(),
        "interested_ids":    set()
    }


def initialize_from_onboarding(profile, schema, interest_index, context_index,
                                selected_categories, selected_subcategories, selected_moods,
                                preferred_times=None, preferred_days=None,
                                preferred_energy=None, preferred_social=None):
    """Populate the profile vectors from the user's onboarding form selections."""
    hierarchy = schema["category_hierarchy"]

    for cat in selected_categories:
        if cat in interest_index["category"]:
            profile["interest_vec"][interest_index["category"][cat]] = CATEGORY_ENCODE_WEIGHT
        # Implicitly boost all subcategories under a selected category
        for sub in hierarchy.get(cat, []):
            if sub in interest_index["subcategory"]:
                pos = interest_index["subcategory"][sub]
                profile["interest_vec"][pos] = max(profile["interest_vec"][pos], 0.35)

    for sub in selected_subcategories:
        if sub in interest_index["subcategory"]:
            pos = interest_index["subcategory"][sub]
            profile["interest_vec"][pos] = max(profile["interest_vec"][pos], SUBCATEGORY_ENCODE_WEIGHT)

    for mood in selected_moods:
        if mood in context_index["mood"]:
            profile["context_vec"][context_index["mood"][mood]] = 1.0

    if preferred_energy and preferred_energy in context_index["energy_level"]:
        profile["context_vec"][context_index["energy_level"][preferred_energy]] = 1.0

    if preferred_times:
        for t in preferred_times:
            if t in context_index["start_time"]:
                profile["context_vec"][context_index["start_time"][t]] = 1.0

    if preferred_days:
        for d in preferred_days:
            if d in context_index["day_of_week"]:
                profile["context_vec"][context_index["day_of_week"][d]] = 1.0

    if preferred_social and preferred_social in context_index["social_intensity"]:
        profile["context_vec"][context_index["social_intensity"][preferred_social]] = 1.0


# ── Profile updates ────────────────────────────────────────────────────────────
# EMA rule: new_profile = decay * old_profile + weight * event_vector
#
# Positive weight -> profile shifts toward the event.
# Negative weight -> profile nudges away, clipped at 0 (no negative dimensions).

def update_from_interaction(profile, event, interest_index, interest_dim,
                             context_index, context_dim,
                             interaction_type="interested", decay=0.85):
    """Shift the user profile toward or away from an event based on the interaction."""
    iv, cv = encode_event(event, interest_index, interest_dim, context_index, context_dim)
    weight = INTERACTION_WEIGHTS.get(interaction_type, 0.10)

    if weight > 0:
        profile["interest_vec"] = decay * profile["interest_vec"] + weight * iv
        profile["context_vec"]  = decay * profile["context_vec"]  + weight * cv
    else:
        profile["interest_vec"] = np.clip(profile["interest_vec"] + weight * iv, 0, None)
        profile["context_vec"]  = np.clip(profile["context_vec"]  + weight * cv, 0, None)

    if interaction_type == "interested":
        profile["interested_ids"].add(event["event_id"])
    elif interaction_type == "attended":
        profile["attended_ids"].add(event["event_id"])
    profile["interaction_count"] += 1


# ── Scoring and recommendations ────────────────────────────────────────────────
# Final score = 0.75 * interest_similarity + 0.25 * context_similarity

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_event(profile, event, interest_index, interest_dim, context_index, context_dim):
    """Return the blended recommendation score for one event against one profile."""
    iv, cv = encode_event(event, interest_index, interest_dim, context_index, context_dim)
    i_sim = cosine_similarity(profile["interest_vec"], iv)
    c_sim = cosine_similarity(profile["context_vec"],  cv)
    return round(INTEREST_WEIGHT * i_sim + CONTEXT_WEIGHT * c_sim, 4)


def recommend_events(profile, events, interest_index, interest_dim,
                     context_index, context_dim, top_n=15, include_interested=False):
    """
    Score every event and return the top N, skipping ones the user already attended.
    Returns [(event_dict, score), ...] sorted highest score first.
    """
    results = []
    for event in events:
        if event["event_id"] in profile["attended_ids"]:
            continue
        if not include_interested and event["event_id"] in profile["interested_ids"]:
            continue
        s = score_event(profile, event, interest_index, interest_dim, context_index, context_dim)
        results.append((event, s))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]