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
import json

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
        self.cat_vars = self._checkbox_group(
            inner, "Interest areas", categories, columns=4,
            on_change=self._refresh_subcategories
        )

        # Subcategory frame -- populated dynamically
        self.sub_frame = tk.LabelFrame(inner, text="Specific interests", padx=6, pady=6)
        self.sub_frame.pack(fill="x", padx=10, pady=6)
        self.sub_vars = {}
        self._refresh_subcategories()  # initialize as empty

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


    def _checkbox_group(self, parent, label, options, columns=4, on_change=None):
        frame = tk.LabelFrame(parent, text=label, padx=6, pady=6)
        frame.pack(fill="x", padx=10, pady=6)
        vars_dict = {}
        for i, opt in enumerate(options):
            var = tk.BooleanVar()
            if on_change:
                var.trace_add("write", on_change)  # fires on check/uncheck
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

    def _refresh_subcategories(self, *args):
        # Clear existing checkboxes
        for widget in self.sub_frame.winfo_children():
            widget.destroy()

        # Determine which subcategories to show based on selected categories
        hierarchy = self.schema["category_hierarchy"]
        selected_cats = [k for k, v in self.cat_vars.items() if v.get()]

        # Collect eligible subcategories in hierarchy order, filtered by the
        # hardcoded display list so you don't show every subcategory
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
        display_set = set(display_subs)

        visible_subs = []
        for cat in selected_cats:
            for sub in hierarchy.get(cat, []):
                if sub in display_set and sub not in visible_subs:
                    visible_subs.append(sub)

        if not visible_subs:
            tk.Label(self.sub_frame,
                     text="Select a category above to see specific interests.",
                     fg="gray").grid(row=0, column=0, sticky="w")
            # Clear any stale sub_vars so deselecting a category drops its subs
            self.sub_vars = {}
            return

        # Preserve prior selections for subcategories still visible
        new_sub_vars = {}
        columns = 4
        for i, sub in enumerate(visible_subs):
            var = self.sub_vars.get(sub, tk.BooleanVar())
            new_sub_vars[sub] = var
            tk.Checkbutton(
                self.sub_frame,
                text=sub.replace("_", " "),
                variable=var
            ).grid(row=i // columns, column=i % columns, sticky="w", padx=4, pady=1)

        self.sub_vars = new_sub_vars


# =============================================================================
# EVENT FEED SCREEN
# =============================================================================
class EventFeedScreen(tk.Frame):

    def __init__(self, parent, profile, events, schema,
                 interest_index, interest_dim, context_index, context_dim, on_add_event=None):
        super().__init__(parent)
        self.profile        = profile
        self.events         = events
        self.schema         = schema
        self.interest_index = interest_index
        self.interest_dim   = interest_dim
        self.context_index  = context_index
        self.context_dim    = context_dim
        self.on_add_event   = on_add_event
        self._build()
        self._refresh()

    def _build(self):
        header = tk.Frame(self)
        header.pack(fill="x", padx=10, pady=8)
        tk.Label(header, text="Your Recommendations",
                 font=("TkDefaultFont", 13, "bold")).pack(side="left")

        if self.on_add_event:
            tk.Button(header, text="+ Add Event",
                      command=self.on_add_event).pack(side="left", padx=6)

        self.status_var = tk.StringVar()
        tk.Label(header, textvariable=self.status_var).pack(side="right", padx=6)

        self.interested_count_var = tk.StringVar()
        tk.Button(header, textvariable=self.interested_count_var,
                  command=self._show_interested_popup,
                  relief="flat", cursor="hand2",
                  font=("TkDefaultFont", 9, "underline"),
                  fg="blue").pack(side="right")

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
            f"Attended: {len(self.profile['attended_ids'])}"
        )
        self.interested_count_var.set(
            f"Interested: {len(self.profile['interested_ids'])}  |  "
        )

        if not recs:
            tk.Label(self.feed, text="No more events to show.",
                     font=("TkDefaultFont", 11)).pack(pady=30)
            return

        for rank, (event, score) in enumerate(recs, start=1):
            self._draw_card(rank, event, score)

    def _show_interested_popup(self):
        popup = tk.Toplevel(self)
        popup.title("Interested Events")
        popup.geometry("620x520")
        popup.grab_set()

        tk.Label(popup, text="Events You're Interested In",
                 font=("TkDefaultFont", 13, "bold")).pack(pady=(12, 2))
        tk.Label(popup, text="Click Attended to mark as going, or remove interest.",
                 fg="gray", font=("TkDefaultFont", 9)).pack()

        canvas  = tk.Canvas(popup, borderwidth=0)
        vscroll = tk.Scrollbar(popup, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner = tk.Frame(canvas)
        win   = canvas.create_window((0, 0), window=inner, anchor="nw")
        inner.bind("<Configure>",
                   lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(win, width=e.width))

        def refresh_popup():
            for w in inner.winfo_children():
                w.destroy()
            current = [e for e in self.events
                       if e["event_id"] in self.profile["interested_ids"]]
            if not current:
                tk.Label(inner, text="No interested events yet.",
                         fg="gray").pack(pady=20)
                return
            for event in current:
                self._draw_interested_card(inner, event, refresh_popup)

        refresh_popup()
        tk.Button(popup, text="Close", command=popup.destroy,
                  width=12).pack(pady=10)

    def _draw_interested_card(self, parent, event, refresh_callback):
        card = tk.Frame(parent, relief="ridge", bd=1)
        card.pack(fill="x", padx=10, pady=4, ipady=4)

        top = tk.Frame(card)
        top.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(top, text=event["name"],
                 font=("TkDefaultFont", 11, "bold")).pack(side="left")

        cats   = ", ".join(event.get("primary_category", []))
        detail = (f"{cats}  |  {event.get('energy_level','')} energy  |  "
                  f"{event.get('setting','')}  |  {event.get('cost','')}")
        tk.Label(card, text=detail,
                 font=("TkDefaultFont", 9)).pack(anchor="w", padx=12)

        days  = "/".join(event.get("day_of_week", []))
        moods = ", ".join(event.get("mood", []))
        tk.Label(card, text=f"Mood: {moods}  |  {event.get('start_time','')}  |  {days}",
                 font=("TkDefaultFont", 9)).pack(anchor="w", padx=12)

        btns = tk.Frame(card)
        btns.pack(anchor="w", padx=8, pady=(4, 4))

        def on_attended(e=event):
            self.profile["interested_ids"].discard(e["event_id"])
            cbrs.update_from_interaction(
                self.profile, e,
                self.interest_index, self.interest_dim,
                self.context_index,  self.context_dim,
                interaction_type="attended"
            )
            self._refresh()
            refresh_callback()

        def on_remove(e=event):
            self.profile["interested_ids"].discard(e["event_id"])
            self._refresh()
            refresh_callback()

        tk.Button(btns, text="Attended",
                  command=on_attended, width=12).pack(side="left", padx=(0, 6))
        tk.Button(btns, text="No Longer Interested",
                  command=on_remove, width=20).pack(side="left")

    def _draw_card(self, rank, event, score):
        card = tk.Frame(self.feed, relief="ridge", bd=1)
        card.pack(fill="x", padx=10, pady=4, ipady=4)

        top = tk.Frame(card)
        top.pack(fill="x", padx=8, pady=(6, 2))
        tk.Label(top, text=f"#{rank}", width=3).pack(side="left")
        tk.Label(top, text=event["name"],
                 font=("TkDefaultFont", 11, "bold")).pack(side="left", padx=4)
        tk.Label(top, text=f"Score: {score:.2f}").pack(side="right")

        cats   = ", ".join(event.get("primary_category", []))
        detail = (f"{cats}  |  {event.get('energy_level','')} energy  |  "
                  f"{event.get('setting','')}  |  {event.get('cost','')}")
        tk.Label(card, text=detail, font=("TkDefaultFont", 9)).pack(anchor="w", padx=12)

        moods  = ", ".join(event.get("mood", []))
        days   = "/".join(event.get("day_of_week", []))
        timing = f"Mood: {moods}  |  {event.get('start_time','')}  |  {days}"
        tk.Label(card, text=timing, font=("TkDefaultFont", 9)).pack(anchor="w", padx=12)

        btns = tk.Frame(card)
        btns.pack(anchor="w", padx=8, pady=(4, 2))
        tk.Button(btns, text="Attend",
                  command=lambda e=event: self._on_attend(e),
                  width=12).pack(side="left", padx=(0, 6))
        tk.Button(btns, text="Interested",
                  command=lambda e=event: self._on_interested(e),
                  width=12).pack(side="left", padx=(0, 6))
        tk.Button(btns, text="Skip",
                  command=lambda e=event: self._on_skip(e),
                  width=8).pack(side="left")

    def _on_attend(self, event):
        cbrs.update_from_interaction(
            self.profile, event,
            self.interest_index, self.interest_dim,
            self.context_index,  self.context_dim,
            interaction_type="attended"
        )
        self._refresh()

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

# =============================================================================
# ADD EVENT SCREEN
# =============================================================================

class AddEventScreen(tk.Frame):

    def __init__(self, parent, schema, events_path, on_done):
        super().__init__(parent)
        self.schema      = schema
        self.events_path = events_path
        self.on_done     = on_done
        self._build()

    def _build(self):
        tk.Label(self, text="Add New Event",
                 font=("TkDefaultFont", 14, "bold")).pack(pady=10)

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

        # ── Text fields ──────────────────────────────────────────────────────
        text_frame = tk.LabelFrame(inner, text="Event details", padx=6, pady=6)
        text_frame.pack(fill="x", padx=10, pady=6)

        self.name_var     = self._labeled_entry(text_frame, "Event name *", 0, width=40)
        #self.event_id_var = self._labeled_entry(text_frame, "Event ID *",   1, width=20)
        #tk.Label(text_frame, text="(e.g. evt_031 — must be unique)",
        #         font=("TkDefaultFont", 8), fg="gray").grid(row=1, column=2, sticky="w", padx=4)

        # ── Single-select dropdowns ───────────────────────────────────────────
        dd_frame = tk.LabelFrame(inner, text="Single-value fields", padx=6, pady=6)
        dd_frame.pack(fill="x", padx=10, pady=6)

        self.energy_var     = self._dropdown(dd_frame, "Energy level *",     self.schema["energy_level"],     0)
        self.cost_var       = self._dropdown(dd_frame, "Cost *",             self.schema["cost"],             1)
        self.setting_var    = self._dropdown(dd_frame, "Setting *",          self.schema["setting"],          2)
        self.location_var   = self._dropdown(dd_frame, "Location *",         self.schema["location"],         3)
        self.social_var     = self._dropdown(dd_frame, "Social intensity *",  self.schema["social_intensity"], 4)
        self.commitment_var = self._dropdown(dd_frame, "Commitment level *",  self.schema["commitment_level"], 5)
        self.skill_var      = self._dropdown(dd_frame, "Skill barrier *",    self.schema["skill_barrier"],    6)
        self.start_var      = self._dropdown(dd_frame, "Start time *",       self.schema["start_time"],       7)

        # ── Checkbox groups ───────────────────────────────────────────────────
        self.day_vars  = self._checkbox_group(inner, "Days of week *",
                                              self.schema["day_of_week"], columns=7)
        self.mood_vars = self._checkbox_group(inner, "Mood *",
                                              self.schema["mood"], columns=6)

        # Categories with dynamic subcategory reveal
        categories = list(self.schema["category_hierarchy"].keys())
        self.cat_vars = self._checkbox_group(
            inner, "Primary categories *", categories, columns=4,
            on_change=self._refresh_subcategories)

        self.sub_frame = tk.LabelFrame(inner, text="Subcategories", padx=6, pady=6)
        self.sub_frame.pack(fill="x", padx=10, pady=6)
        self.sub_vars  = {}
        self._refresh_subcategories()

        # ── Error label + submit ──────────────────────────────────────────────
        self.error_var = tk.StringVar()
        tk.Label(inner, textvariable=self.error_var,
                 fg="red", wraplength=500).pack(padx=10, anchor="w")

        btn_row = tk.Frame(inner)
        btn_row.pack(pady=12)
        tk.Button(btn_row, text="Save Event", command=self._submit,
                  width=16).pack(side="left", padx=6)
        tk.Button(btn_row, text="Cancel",     command=self.on_done,
                  width=10).pack(side="left")


    # ── Widget helpers ────────────────────────────────────────────────────────

    def _labeled_entry(self, parent, label, row, width=30):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=3)
        var = tk.StringVar()
        tk.Entry(parent, textvariable=var, width=width).grid(
            row=row, column=1, sticky="w", padx=6, pady=3)
        return var

    def _dropdown(self, parent, label, options, row):
        tk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=2)
        var = tk.StringVar(value=options[0])
        ttk.Combobox(parent, textvariable=var, values=options,
                     state="readonly", width=18).grid(
            row=row, column=1, sticky="w", padx=6, pady=2)
        return var

    def _checkbox_group(self, parent, label, options, columns=4, on_change=None):
        frame = tk.LabelFrame(parent, text=label, padx=6, pady=6)
        frame.pack(fill="x", padx=10, pady=6)
        vars_dict = {}
        for i, opt in enumerate(options):
            var = tk.BooleanVar()
            if on_change:
                var.trace_add("write", on_change)
            vars_dict[opt] = var
            tk.Checkbutton(frame, text=opt.replace("_", " "),
                           variable=var).grid(row=i // columns, column=i % columns,
                                              sticky="w", padx=4, pady=1)
        return vars_dict

    def _refresh_subcategories(self, *args):
        for widget in self.sub_frame.winfo_children():
            widget.destroy()

        hierarchy     = self.schema["category_hierarchy"]
        selected_cats = [k for k, v in self.cat_vars.items() if v.get()]
        visible_subs  = []
        for cat in selected_cats:
            for sub in hierarchy.get(cat, []):
                if sub not in visible_subs:
                    visible_subs.append(sub)

        if not visible_subs:
            tk.Label(self.sub_frame,
                     text="Select a category above to see subcategories.",
                     fg="gray").grid(row=0, column=0, sticky="w")
            self.sub_vars = {}
            return

        new_sub_vars = {}
        columns = 5
        for i, sub in enumerate(visible_subs):
            var = self.sub_vars.get(sub, tk.BooleanVar())
            new_sub_vars[sub] = var
            tk.Checkbutton(self.sub_frame, text=sub.replace("_", " "),
                           variable=var).grid(row=i // columns, column=i % columns,
                                              sticky="w", padx=4, pady=1)
        self.sub_vars = new_sub_vars

    # ── Validation + save ─────────────────────────────────────────────────────

    def _validate(self):
        errors = []
        if not self.name_var.get().strip():
            errors.append("Event name is required.")
        #if not self.event_id_var.get().strip():
        #    errors.append("Event ID is required.")
        if not any(v.get() for v in self.day_vars.values()):
            errors.append("Select at least one day of the week.")
        if not any(v.get() for v in self.mood_vars.values()):
            errors.append("Select at least one mood.")
        if not any(v.get() for v in self.cat_vars.values()):
            errors.append("Select at least one primary category.")
        return errors

    def _submit(self):
        errors = self._validate()
        if errors:
            self.error_var.set("⚠ " + "  |  ".join(errors))
            return
        self.error_var.set("")

        with open(self.events_path, "r") as f:
            events = json.load(f)
        
        existing_nums = []
        for e in events:
            try: 
                existing_nums.append(int(e.get("event_id", "")))
            except ValueError:
                pass

        new_id = str(max(existing_nums, default=0) + 1)

        new_event = {
            #"event_id":         self.event_id_var.get().strip(),
            "event_id":         new_id,
            "name":             self.name_var.get().strip(),
            "start_time":       self.start_var.get(),
            "day_of_week":      [k for k, v in self.day_vars.items()  if v.get()],
            "cost":             self.cost_var.get(),
            "setting":          self.setting_var.get(),
            "location":         self.location_var.get(),
            "primary_category": [k for k, v in self.cat_vars.items()  if v.get()],
            "subcategory":      [k for k, v in self.sub_vars.items()   if v.get()],
            "energy_level":     self.energy_var.get(),
            "social_intensity": self.social_var.get(),
            "commitment_level": self.commitment_var.get(),
            "skill_barrier":    self.skill_var.get(),
            "mood":             [k for k, v in self.mood_vars.items()  if v.get()],
        }

        # Load → append → write back
        with open(self.events_path, "r") as f:
            events = json.load(f)

        # Duplicate ID check
        existing_ids = {e["event_id"] for e in events}
        if new_event["event_id"] in existing_ids:
            self.error_var.set(f"⚠ Event ID '{new_event['event_id']}' already exists.")
            return

        events.append(new_event)
        with open(self.events_path, "w") as f:
            json.dump(events, f, indent=2)

        self.on_done(new_event)


