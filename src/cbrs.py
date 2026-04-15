"""
cbrs.py  --  Content-Based Recommender System Core
====================================================
Sections:
  1. Load data (schema, events, archetypes)
  2. Build index maps from the schema
  3. Event encoding
  4. User profile: creation and onboarding initialization
  5. Profile update from interactions
  6. Cosine similarity and recommendation
"""

import json
import numpy as np


# =============================================================================
# SECTION 1: LOAD DATA
# =============================================================================

def load_schema(path):
    with open(path, "r") as f:
        return json.load(f)

def load_events(path):
    with open(path, "r") as f:
        return json.load(f)

def load_archetypes(path):
    with open(path, "r") as f:
        return json.load(f)


# =============================================================================
# SECTION 2: BUILD INDEX MAPS
# =============================================================================
# Features are split into two groups so interest similarity (what kind of event)
# can be weighted more heavily than context similarity (when/where/how).
#
# Interest: categories, subcategories, mood, energy_level, skill_barrier
# Context:  start_time, day_of_week, cost, setting, location,
#           social_intensity, commitment_level
#
# Categories and subcategories are derived from the hierarchy in the schema
# rather than listed separately.
#
# An index map tells us where in the vector each value lives.
# e.g. index_map["mood"]["chill"] = 3  means position 3 is the "chill" dimension.

INTEREST_FEATURE_KEYS = ["category", "subcategory", "mood", "energy_level", "skill_barrier"]
CONTEXT_FEATURE_KEYS  = ["start_time", "day_of_week", "cost", "setting", "location",
                          "social_intensity", "commitment_level"]

INTEREST_WEIGHT = 0.75   # interest similarity contributes 75% of the final score
CONTEXT_WEIGHT  = 0.25   # context similarity contributes 25%

# How strongly each interaction type shifts the profile (see Section 5)
INTERACTION_WEIGHTS = {
    "interested":  0.30,
    "skip":       -0.05,
    "attended":    0.5
}


def get_categories_and_subcategories(schema):
    """Extract ordered category and subcategory lists from the hierarchy."""
    hierarchy = schema["category_hierarchy"]
    categories = list(hierarchy.keys())
    seen, subcategories = set(), []
    for subs in hierarchy.values():
        for s in subs:
            if s not in seen:
                seen.add(s)
                subcategories.append(s)
    return categories, subcategories


def build_index_map(feature_keys, schema, categories, subcategories):
    """
    Build a position lookup for each feature and its values.
    Returns (index_map, total_dimensions).
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
    Call once after loading the schema.
    Returns the index maps and dimensions needed everywhere else.
    """
    categories, subcategories = get_categories_and_subcategories(schema)
    interest_index, interest_dim = build_index_map(
        INTEREST_FEATURE_KEYS, schema, categories, subcategories)
    context_index, context_dim = build_index_map(
        CONTEXT_FEATURE_KEYS, schema, categories, subcategories)
    return interest_index, interest_dim, context_index, context_dim


# =============================================================================
# SECTION 3: EVENT ENCODING
# =============================================================================
# Events are plain dicts loaded from JSON.
# encode_event() turns one event dict into two numpy vectors.
#
# Categories encode at 1.0 and subcategories at 0.6 to reflect that
# the user's onboarding weighting (up to 1.5 for explicit subcategories)
# should still dominate after EMA updates — a broadly tagged event
# shouldn't fully override a specific user preference.

CATEGORY_ENCODE_WEIGHT    = 1.0
SUBCATEGORY_ENCODE_WEIGHT = 0.6


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
        if m in interest_index["mood"]:
            iv[interest_index["mood"][m]] = 1.0

    energy = event.get("energy_level", "")
    if energy in interest_index["energy_level"]:
        iv[interest_index["energy_level"][energy]] = 1.0

    skill = event.get("skill_barrier", "")
    if skill in interest_index["skill_barrier"]:
        iv[interest_index["skill_barrier"][skill]] = 1.0

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


# =============================================================================
# SECTION 4: USER PROFILE AND ONBOARDING
# =============================================================================
# A user profile is a plain dict:
#   {
#     "name":              str,
#     "interest_vec":      np.ndarray,
#     "context_vec":       np.ndarray,
#     "interaction_count": int,
#     "attended_ids":      set
#   }
#
# Onboarding weighting tiers:
#   Selected primary category        -> 1.0
#   Subcategories under that primary -> 0.35  (implicit interest)
#   Explicitly selected subcategory  -> 1.5   (strongest signal)
# The max() call ensures an explicit selection always wins over an implicit one.

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
    """Set profile vectors from onboarding form selections."""
    hierarchy = schema["category_hierarchy"]

    for cat in selected_categories:
        if cat in interest_index["category"]:
            profile["interest_vec"][interest_index["category"][cat]] = 1.0
        for sub in hierarchy.get(cat, []):
            if sub in interest_index["subcategory"]:
                pos = interest_index["subcategory"][sub]
                profile["interest_vec"][pos] = max(profile["interest_vec"][pos], 0.35)

    for sub in selected_subcategories:
        if sub in interest_index["subcategory"]:
            pos = interest_index["subcategory"][sub]
            profile["interest_vec"][pos] = max(profile["interest_vec"][pos], 1.5)

    for mood in selected_moods:
        if mood in interest_index["mood"]:
            profile["interest_vec"][interest_index["mood"][mood]] = 1.0

    if preferred_energy and preferred_energy in interest_index["energy_level"]:
        profile["interest_vec"][interest_index["energy_level"][preferred_energy]] = 1.0

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


# =============================================================================
# SECTION 5: PROFILE UPDATE FROM INTERACTIONS
# =============================================================================
# EMA update rule:
#   new_profile = decay * old_profile + weight * event_vector
#
# "interested" uses weight +0.20: profile shifts toward the event.
# "skip"       uses weight -0.05: profile nudges away; clipped at 0.

def update_from_interaction(profile, event, interest_index, interest_dim,
                             context_index, context_dim,
                             interaction_type="interested", decay=0.85):
    """Update the user profile after an interaction with an event."""
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


# =============================================================================
# SECTION 6: COSINE SIMILARITY AND RECOMMENDATION
# =============================================================================
# Final score = 0.75 * interest_similarity + 0.25 * context_similarity

def cosine_similarity(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def score_event(profile, event, interest_index, interest_dim, context_index, context_dim):
    """Compute blended recommendation score for one event against one user profile."""
    iv, cv = encode_event(event, interest_index, interest_dim, context_index, context_dim)
    i_sim = cosine_similarity(profile["interest_vec"], iv)
    c_sim = cosine_similarity(profile["context_vec"],  cv)
    return round(INTEREST_WEIGHT * i_sim + CONTEXT_WEIGHT * c_sim, 4)


def recommend_events(profile, events, interest_index, interest_dim,
                     context_index, context_dim, top_n=10):
    """
    Score all events and return top N, excluding already-attended ones.
    Returns a list of (event_dict, score) sorted highest score first.
    """
    results = []
    for event in events:
        if event["event_id"] in profile["attended_ids"] or event["event_id"] in profile["interested_ids"]:
            continue
        s = score_event(profile, event, interest_index, interest_dim, context_index, context_dim)
        results.append((event, s))
    results.sort(key=lambda x: x[1], reverse=True)
    return results[:top_n]