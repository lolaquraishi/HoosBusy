"""
simulation.py  --  Archetype Simulation Demo
=============================================
Runs the CBRS recommendation loop on a set of pre-defined archetype users
and measures how well the estimated preference profile converges toward
each archetype's true preferences over a series of interactions.

Edit the settings below to change how the simulation runs.
Run with: python simulation.py
"""

import numpy as np
import os
import cbrs
import matplotlib.pyplot as plt

# =============================================================================
# SETTINGS
# =============================================================================

SCHEMA_PATH     = os.path.join(os.path.dirname(__file__), "../data", "feature_schema.json")
EVENTS_PATH     = os.path.join(os.path.dirname(__file__), "../data", "events.json")
ARCHETYPES_PATH = os.path.join(os.path.dirname(__file__), "../data", "archetypes.json")

NUM_STEPS   = 10    # number of interaction steps to simulate per user
TOP_N       = 15    # events recommended per step
DECAY       = 0.90  # EMA decay: how much of the old profile is kept each update
RANDOM_SEED = 42    # set to None for a different result each run

# Noise applied to onboarding (0.4 = each preference scaled down by 0-40%)
# Higher noise = estimated profile starts further from ground truth
ONBOARDING_NOISE = 0.4

# =============================================================================
# HELPERS
# =============================================================================

def build_ground_truth(archetype, interest_index, interest_dim, context_index, context_dim):
    """
    Build the ideal profile for an archetype directly from its preferences dict.
    Each preference value (0.0-1.0) maps directly into the vector position
    for that feature value. This is the target we measure convergence toward.
    """
    gt = cbrs.make_profile(archetype["name"] + " [ground truth]", interest_dim, context_dim)
    prefs = archetype.get("preferences", {})
    for feature, positions in interest_index.items():
        for value, pos in positions.items():
            if value in prefs:
                gt["interest_vec"][pos] = prefs[value]
    for feature, positions in context_index.items():
        for value, pos in positions.items():
            if value in prefs:
                gt["context_vec"][pos] = prefs[value]
    return gt


def build_onboarding_profile(archetype, interest_index, interest_dim,
                              context_index, context_dim, rng):
    """
    Build a noisy onboarding profile by scaling each true preference value
    down by a random factor, simulating imperfect self-expression at onboarding.
    """
    noise   = archetype.get("onboarding_noise", ONBOARDING_NOISE)
    profile = cbrs.make_profile(archetype["name"], interest_dim, context_dim)
    prefs   = archetype.get("preferences", {})
    for feature, positions in interest_index.items():
        for value, pos in positions.items():
            if value in prefs:
                scale = rng.uniform(1.0 - noise, 1.0)
                profile["interest_vec"][pos] = prefs[value] * scale
    for feature, positions in context_index.items():
        for value, pos in positions.items():
            if value in prefs:
                scale = rng.uniform(1.0 - noise, 1.0)
                profile["context_vec"][pos] = prefs[value] * scale
    return profile


def profile_similarity(profile, ground_truth):
    """
    Cosine similarity between estimated and ground-truth profiles.
    Both vectors are unit-normalized so only direction matters, not magnitude.
    Returns the blended (interest + context) similarity (0.0 to 1.0).
    """
    def unit(v):
        n = np.linalg.norm(v)
        return v / n if n > 0 else v
    i_sim = cbrs.cosine_similarity(unit(profile["interest_vec"]),
                                    unit(ground_truth["interest_vec"]))
    c_sim = cbrs.cosine_similarity(unit(profile["context_vec"]),
                                    unit(ground_truth["context_vec"]))
    return round(cbrs.INTEREST_WEIGHT * i_sim + cbrs.CONTEXT_WEIGHT * c_sim, 4)


# =============================================================================
# MAIN
# =============================================================================

def run_simulation():
    print("=" * 60)
    print("CBRS ARCHETYPE SIMULATION")
    print("=" * 60)

    # --- Load data ---
    schema     = cbrs.load_schema(SCHEMA_PATH)
    events     = cbrs.load_events(EVENTS_PATH)
    archetypes = cbrs.load_archetypes(ARCHETYPES_PATH)

    interest_index, interest_dim, context_index, context_dim = cbrs.setup_vector_space(schema)

    rng = np.random.default_rng(RANDOM_SEED)

    print(f"\nLoaded {len(events)} events, {len(archetypes)} archetypes")
    print(f"Vector dims  -- Interest: {interest_dim}, Context: {context_dim}")
    print(f"Steps: {NUM_STEPS}  |  Top-N: {TOP_N}  |  Decay: {DECAY}\n")

    similarity_scores = []
    all_similarity_scores = {}
    # --- Run each archetype ---
    for archetype in archetypes:
        similarity_scores_over_steps = []
        print(f"--- {archetype['name'].upper()} ---")
        print(f"{archetype['description']}\n")

        ground_truth = build_ground_truth(
            archetype, interest_index, interest_dim, context_index, context_dim)
        estimated = build_onboarding_profile(
            archetype, interest_index, interest_dim, context_index, context_dim, rng)
        
        print(f"Initial similarity score: {profile_similarity(estimated, ground_truth)}")

        # --- Interaction loop ---
        for step in range(NUM_STEPS):
            recs = cbrs.recommend_events(
                estimated, events,
                interest_index, interest_dim, context_index, context_dim,
                top_n=TOP_N, include_interested=True
            )
            if not recs:
                break

            # Compute ground-truth scores for the recommended events to determine interactions
            gt_scores = []
            for event, _ in recs:
                gt_score = cbrs.score_event(
                    ground_truth, event,
                    interest_index, interest_dim, context_index, context_dim
                )
                gt_scores.append(gt_score)

            total = sum(gt_scores)
            weights = [s / total for s in gt_scores] if total > 0 else [1.0 / len(recs)] * len(recs)

            chosen_idx = rng.choice(len(recs), p=weights)
            chosen_event, _ = recs[chosen_idx]
            
            if chosen_event["event_id"] in estimated["interested_ids"]:
                interested = True
            else:
                interested = False

            ATTEND_THRESHOLD = 0.5
            GT_THRESHOLD = 0.4
            if interested:
                interaction = "attended" if gt_scores[chosen_idx] >= ATTEND_THRESHOLD else "remove"
            else:
                if gt_scores[chosen_idx] >= ATTEND_THRESHOLD:
                    interaction = "attended"
                else:
                    interaction = "interested" if gt_scores[chosen_idx] >= GT_THRESHOLD else "skip"
            
            cbrs.update_from_interaction(
                estimated, chosen_event,
                interest_index, interest_dim, context_index, context_dim,
                interaction_type=interaction, decay=DECAY
            )

            print(f"Step {step}: Action={interaction}, Event={chosen_event['name']}. Updated similarity score: {profile_similarity(estimated, ground_truth)}")

            similarity_scores_over_steps.append(profile_similarity(estimated, ground_truth))

            all_similarity_scores[archetype["name"]] = similarity_scores_over_steps


        final_sim = profile_similarity(estimated, ground_truth)
        similarity_scores.append(final_sim)
        print(f"Final similarity  (after {NUM_STEPS} steps):  {final_sim:.4f}")

        print(f"\nTop recommendations (full pool):")
        attended_backup = estimated["attended_ids"].copy()
        estimated["attended_ids"] = set()
        final_recs_full = cbrs.recommend_events(
            estimated, events,
            interest_index, interest_dim, context_index, context_dim,
            top_n=5
        )
        estimated["attended_ids"] = attended_backup
        for rank, (event, score) in enumerate(final_recs_full, 1):
            print(f"  {rank}. {event['name']:<40} score: {score:.4f}")
        print()

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, scores in all_similarity_scores.items():
        ax.plot(range(1, len(scores) + 1), scores, marker="o", label=name)

    ax.set_xlabel("Step")
    ax.set_ylabel("Profile Similarity")
    ax.set_title("Profile Similarity Convergence")
    ax.set_ylim(0.65, 1)
    ax.grid(True, alpha=0.3)
    avg_scores = np.mean(list(all_similarity_scores.values()), axis=0)
    ax.plot(range(1, len(avg_scores) + 1), avg_scores, 
            color="black", linewidth=2.5, linestyle="--", marker="s", label="Average")
    ax.annotate(f"{avg_scores[-1]:.4f}", 
                xy=(len(avg_scores), avg_scores[-1]),
                xytext=(8, 0), textcoords="offset points",
                va="center", fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig("similarity_over_time.png", dpi=150)
    plt.show()


if __name__ == "__main__":
    run_simulation()