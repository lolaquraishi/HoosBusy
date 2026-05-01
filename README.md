# CS4710 Team 3 — HoosBusy
### Members: Zachary Forino, Lulya Haile, Alexander James, Lola Quaraishi, Diya Tomar

A content-based event recommender system for UVA students. Users complete a short onboarding form to initialize a preference profile, then interact with recommended events (Attend / Interested / Skip / Remove) to refine it over time. Recommendations are ranked by weighted cosine similarity between the user's profile vector and each event's feature vector.

## Repository structure

```
CS4710-Team3/
├── data/
│   ├── feature_schema.json   # Feature definitions and category hierarchy
│   ├── events.json           # Event dataset (100 events)
│   └── archetypes.json       # Simulated user archetypes for evaluation
├── src/
│   ├── cbrs.py               # Core recommender logic (encoding, scoring, profile updates)
│   ├── simulation.py         # Archetype-based simulation and evaluation
│   ├── app.py                # Tkinter app entry point
│   └── demo_ui.py            # UI screens (onboarding, feed, add event)
└── archive/
    └── cbrs_prelim.py        # Early prototype from milestone demo
```

## Requirements

```
pip install numpy matplotlib
```

Tkinter is included with most standard Python installations. Python 3.8+ recommended.

## Running the interactive UI

```
cd src
python app.py
```

This opens the desktop application. Complete the onboarding form to generate an initial profile, then browse recommended events and interact with them to update your profile in real time.

## Running the simulation

```
cd src
python simulation.py
```

This runs the recommendation loop across all ten archetypes defined in `data/archetypes.json`, printing per-step similarity scores to the console and saving a convergence plot as `similarity_over_time.png` in the working directory.

Simulation settings (number of steps, decay rate, top-N, random seed) can be adjusted at the top of `simulation.py`.
