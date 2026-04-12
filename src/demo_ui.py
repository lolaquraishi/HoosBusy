"""
demo_ui.py  --  Interactive Demo UI
=====================================
Two-screen Tkinter application:
  Screen 1 -- Onboarding: user fills out their preferences and submits.
  Screen 2 -- Event feed: recommended events ranked by score.
              Interested or Skip buttons update the profile and re-rank the list.
"""

import tkinter as tk
import os
from tkinter import ttk

import cbrs

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../data", "feature_schema.json")
EVENTS_PATH = os.path.join(os.path.dirname(__file__), "../data", "events.json")
TOP_N = 10


# =============================================================================
# ONBOARDING SCREEN
# =============================================================================

class OnboardingScreen(tk.Frame):

    def __init__(self, parent, schema, on_submit):
        super().__init__(parent)
        self.schema    = schema
        self.on_submit = on_submit
        self._build()

    def _build(self):
        tk.Label(self, text="Event Recommender -- Onboarding",
                 font=("TkDefaultFont", 14, "bold")).pack(pady=10)
        tk.Label(self, text="Select your interests to get personalized recommendations.",
                 wraplength=500).pack()

        # Scrollable area
        canvas  = tk.Canvas(self, borderwidth=0)
        vscroll = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))

        # Name
        name_frame = tk.LabelFrame(inner, text="Your name", padx=6, pady=6)
        name_frame.pack(fill="x", padx=10, pady=6)
        self.name_var = tk.StringVar()
        tk.Entry(name_frame, textvariable=self.name_var, width=30).pack(anchor="w")

        # Category checkboxes -- derive from hierarchy
        categories = list(self.schema["category_hierarchy"].keys())
        self.cat_vars = self._checkbox_group(inner, "Interest areas", categories, columns=4)

        # Subcategory checkboxes -- show a useful subset
        display_subs = [
            "hiking", "rock_climbing", "trail_run", "stargazing",
            "live_concert", "open_mic", "a_cappella", "karaoke",
            "hackathon", "coding_workshop", "ctf_security", "board_games",
            "arts_crafts", "creative_writing", "poetry", "photography",
            "networking", "career_fair", "resume_prep", "info_session",
            "yoga", "meditation", "mindfulness", "self_care",
            "food_bank", "community_service", "environmental_cleanup",
            "pickup_game", "intramural", "varsity",
            "panel_discussion", "research_showcase", "lecture", "trivia",
            "video_games", "esports", "tabletop_rpg", "game_night",
            "party", "mixer", "speed_friending", "hangout",
            "cooking_class", "potluck", "improv_comedy", "movie_screening",
        ]
        self.sub_vars = self._checkbox_group(inner, "Specific interests", display_subs, columns=4)

        # Mood checkboxes
        self.mood_vars = self._checkbox_group(inner, "Vibe you are looking for",
                                              self.schema["mood"], columns=6)

        # Dropdowns
        pref_frame = tk.LabelFrame(inner, text="Preferences", padx=6, pady=6)
        pref_frame.pack(fill="x", padx=10, pady=6)
        self.energy_var = self._labeled_dropdown(
            pref_frame, "Energy level:", self.schema["energy_level"], "medium", col=0)
        self.social_var = self._labeled_dropdown(
            pref_frame, "Social setting:", self.schema["social_intensity"], "small_group", col=2)

        # Day and time checkboxes
        self.day_vars  = self._checkbox_group(inner, "Preferred days",
                                              self.schema["day_of_week"], columns=7)
        self.time_vars = self._checkbox_group(inner, "Preferred times",
                                              self.schema["start_time"], columns=4)

        tk.Button(inner, text="Get Recommendations",
                  command=self._submit, width=25).pack(pady=16)

    def _checkbox_group(self, parent, label, options, columns=4):
        frame = tk.LabelFrame(parent, text=label, padx=6, pady=6)
        frame.pack(fill="x", padx=10, pady=6)
        vars_dict = {}
        for i, opt in enumerate(options):
            var = tk.BooleanVar()
            vars_dict[opt] = var
            tk.Checkbutton(frame, text=opt.replace("_", " "),
                           variable=var).grid(row=i // columns, column=i % columns,
                                              sticky="w", padx=4, pady=1)
        return vars_dict

    def _labeled_dropdown(self, parent, label, options, default, col=0):
        tk.Label(parent, text=label).grid(row=0, column=col, sticky="w", padx=(0, 4))
        var = tk.StringVar(value=default)
        cb  = ttk.Combobox(parent, textvariable=var, values=options,
                           state="readonly", width=16)
        cb.grid(row=0, column=col + 1, sticky="w", padx=(0, 20))
        return var

    def _submit(self):
        name = self.name_var.get().strip() or "User"
        self.on_submit(
            name                   = name,
            selected_categories    = [k for k, v in self.cat_vars.items()  if v.get()],
            selected_subcategories = [k for k, v in self.sub_vars.items()  if v.get()],
            selected_moods         = [k for k, v in self.mood_vars.items() if v.get()],
            preferred_days         = [k for k, v in self.day_vars.items()  if v.get()] or None,
            preferred_times        = [k for k, v in self.time_vars.items() if v.get()] or None,
            preferred_energy       = self.energy_var.get() or None,
            preferred_social       = self.social_var.get() or None,
        )


# =============================================================================
# EVENT FEED SCREEN
# =============================================================================

class EventFeedScreen(tk.Frame):

    def __init__(self, parent, profile, events, schema,
                 interest_index, interest_dim, context_index, context_dim):
        super().__init__(parent)
        self.profile        = profile
        self.events         = events
        self.schema         = schema
        self.interest_index = interest_index
        self.interest_dim   = interest_dim
        self.context_index  = context_index
        self.context_dim    = context_dim
        self._build()
        self._refresh()

    def _build(self):
        header = tk.Frame(self)
        header.pack(fill="x", padx=10, pady=8)
        tk.Label(header, text="Your Recommendations",
                 font=("TkDefaultFont", 13, "bold")).pack(side="left")
        self.status_var = tk.StringVar()
        tk.Label(header, textvariable=self.status_var).pack(side="right")

        # Horizontal rule (Frame with height 1 as a divider replacement)
        tk.Frame(self, height=1, bg="gray").pack(fill="x", padx=10)

        canvas  = tk.Canvas(self, borderwidth=0)
        vscroll = tk.Scrollbar(self, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        self.feed = tk.Frame(canvas)
        self._feed_win = canvas.create_window((0, 0), window=self.feed, anchor="nw")
        self.feed.bind("<Configure>",
                       lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(self._feed_win, width=e.width))
        canvas.bind_all("<MouseWheel>",
                        lambda e: canvas.yview_scroll(-1 * (e.delta // 120), "units"))
        self._canvas = canvas

    def _refresh(self):
        for widget in self.feed.winfo_children():
            widget.destroy()
        self._canvas.yview_moveto(0)

        recs = cbrs.recommend_events(
            self.profile, self.events,
            self.interest_index, self.interest_dim,
            self.context_index,  self.context_dim,
            top_n=TOP_N
        )

        self.status_var.set(
            f"Interactions: {self.profile['interaction_count']}  |  "
            f"Interested: {len(self.profile['attended_ids'])}  |  "
            f"Showing top {len(recs)}"
        )

        if not recs:
            tk.Label(self.feed, text="No more events to show.",
                     font=("TkDefaultFont", 11)).pack(pady=30)
            return

        for rank, (event, score) in enumerate(recs, start=1):
            self._draw_card(rank, event, score)

    def _draw_card(self, rank, event, score):
        card = tk.Frame(self.feed, relief="ridge", bd=1)
        card.pack(fill="x", padx=10, pady=4, ipady=4)

        # Name row
        top = tk.Frame(card)
        top.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(top, text=f"#{rank}", width=3).pack(side="left")
        tk.Label(top, text=event["name"],
                 font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=4)
        tk.Label(top, text=f"Score: {score:.2f}").pack(side="right")

        # Detail row
        cats   = ", ".join(event.get("primary_category", []))
        detail = (f"{cats}  |  {event.get('energy_level','')} energy  |  "
                  f"{event.get('setting','')}  |  {event.get('cost','')}")
        tk.Label(card, text=detail, font=("TkDefaultFont", 9)).pack(anchor="w", padx=12)

        # Timing row
        moods  = ", ".join(event.get("mood", []))
        days   = "/".join(event.get("day_of_week", []))
        timing = f"Mood: {moods}  |  {event.get('start_time','')}  |  {days}"
        tk.Label(card, text=timing, font=("TkDefaultFont", 9)).pack(anchor="w", padx=12)

        # Buttons
        btns = tk.Frame(card)
        btns.pack(anchor="w", padx=8, pady=(4, 2))
        tk.Button(btns, text="Interested",
                  command=lambda e=event: self._on_interested(e),
                  width=12).pack(side="left", padx=(0, 6))
        tk.Button(btns, text="Skip",
                  command=lambda e=event: self._on_skip(e),
                  width=8).pack(side="left")

    def _on_interested(self, event):
        cbrs.update_from_interaction(
            self.profile, event,
            self.interest_index, self.interest_dim,
            self.context_index,  self.context_dim,
            interaction_type="interested"
        )
        self._refresh()

    def _on_skip(self, event):
        cbrs.update_from_interaction(
            self.profile, event,
            self.interest_index, self.interest_dim,
            self.context_index,  self.context_dim,
            interaction_type="skip"
        )
        self._refresh()


# =============================================================================
# APP
# =============================================================================

class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CBRS Event Recommender Demo")
        self.geometry("750x650")

        self.schema = cbrs.load_schema(SCHEMA_PATH)
        self.events = cbrs.load_events(EVENTS_PATH)

        (self.interest_index, self.interest_dim,
         self.context_index,  self.context_dim) = cbrs.setup_vector_space(self.schema)

        self._show_onboarding()

    def _show_onboarding(self):
        self._clear()
        OnboardingScreen(self, self.schema, on_submit=self._on_submit).pack(
            fill="both", expand=True)

    def _on_submit(self, name, selected_categories, selected_subcategories,
                   selected_moods, preferred_days, preferred_times,
                   preferred_energy, preferred_social):
        profile = cbrs.make_profile(name, self.interest_dim, self.context_dim)
        cbrs.initialize_from_onboarding(
            profile, self.schema, self.interest_index, self.context_index,
            selected_categories    = selected_categories,
            selected_subcategories = selected_subcategories,
            selected_moods         = selected_moods,
            preferred_times        = preferred_times,
            preferred_days         = preferred_days,
            preferred_energy       = preferred_energy,
            preferred_social       = preferred_social,
        )
        self._show_feed(profile)

    def _show_feed(self, profile):
        self._clear()
        EventFeedScreen(
            self, profile, self.events, self.schema,
            self.interest_index, self.interest_dim,
            self.context_index,  self.context_dim,
        ).pack(fill="both", expand=True)

    def _clear(self):
        for widget in self.winfo_children():
            widget.destroy()


if __name__ == "__main__":
    App().mainloop()