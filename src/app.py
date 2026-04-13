"""
app.py  --  Application entry point
=====================================
Handles screen navigation and shared state.
Run with: python app.py
"""

import json
import tkinter as tk
import os

import cbrs
from demo_ui import OnboardingScreen, EventFeedScreen, AddEventScreen

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "../data", "feature_schema.json")
EVENTS_PATH = os.path.join(os.path.dirname(__file__), "../data", "events.json")


class App(tk.Tk):

    def __init__(self):
        super().__init__()
        self.title("CBRS Event Recommender Demo")
        self.geometry("750x650")

        self.schema = cbrs.load_schema(SCHEMA_PATH)
        self.events = cbrs.load_events(EVENTS_PATH)

        (self.interest_index, self.interest_dim,
         self.context_index,  self.context_dim) = cbrs.setup_vector_space(self.schema)

        self.current_profile = None
        self._show_onboarding()

    def _clear(self):
        for w in self.winfo_children():
            w.destroy()

    def _show_onboarding(self):
        self._clear()
        OnboardingScreen(self, self.schema,
                         on_submit=self._on_submit).pack(fill="both", expand=True)

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
        self.current_profile = profile
        self._show_feed()

    def _show_feed(self):
        self._clear()
        EventFeedScreen(
            self, self.current_profile, self.events, self.schema,
            self.interest_index, self.interest_dim,
            self.context_index,  self.context_dim,
            on_add_event=self._show_add_event,
        ).pack(fill="both", expand=True)

    def _show_add_event(self):
        self._clear()
        AddEventScreen(
            self, self.schema, EVENTS_PATH,
            on_done=self._on_event_added,
        ).pack(fill="both", expand=True)

    def _on_event_added(self, new_event=None):
        self.events = cbrs.load_events(EVENTS_PATH)  # reload so new event is scoreable
        if self.current_profile is not None:
            self._show_feed()
        else:
            self._show_onboarding()


if __name__ == "__main__":
    App().mainloop()