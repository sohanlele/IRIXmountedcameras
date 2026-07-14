"""Set/session-level fatigue analysis, aggregated from per-rep signals
this repo already computes (irix.rep_counting, irix.barbell.rpe,
irix.form.scoring) into the structured context irix-mvp-app's AI uses to
shape a member's next set (target weight/reps) -- expanding on the
per-rep numbers docs/ARCHITECTURE.md previously described as "this repo
only supplies the numbers, the fatigue judgment is the app's job".

That boundary still holds in spirit: nothing here decides what a member
should do next (no "reduce the weight" or "stop the set" instruction) --
it aggregates and classifies (velocity-loss %, which VL-zone the set
landed in, tempo drift, form trend) rather than prescribes. The app's AI
is still the thing that turns "this set showed VL22 with a rising elbow-
drift trend" into an actual coaching decision; this module just does the
arithmetic so the app doesn't have to re-derive it from a raw rep stream
every time.
"""
