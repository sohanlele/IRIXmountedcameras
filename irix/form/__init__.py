"""Rule-based rep form scoring.

Populates ``RepCompletedEvent.form_score`` / ``form_faults`` (defined in
``irix.pipeline.schema``, previously always ``None``/empty -- no code
anywhere in this repo produced them).

Approach and provenance: rather than the deep-learning per-exercise
classifiers several prior open-source projects use (frame sequences ->
LSTM/BiLSTM -> good/bad form label -- see
github.com/chrisprasanna/Exercise_Recognition_AI,
github.com/RiccardoRiccio/Fitness-AI-Trainer-With-Automatic-Exercise-Recognition-and-Counting),
this module scores form with direct joint-angle/keypoint geometry, in
keeping with this repo's existing pattern (``irix.rep_counting``,
``irix.barbell.rpe``): pure math over already-available pose keypoints,
fully unit-testable with synthetic fixtures, no training data or model
weights required. The specific fault taxonomy (which faults are worth
checking per exercise) is grounded in prior art:

- github.com/chrisprasanna/Exercise_Recognition_AI lists exactly this
  category of check on its roadmap: "detect poor form (e.g., leaning,
  fast eccentric motion, knees caving in, poor squat depth)".
- github.com/NgoQuocBao1010/Exercise-Correction trains one classifier per
  exercise per fault: bicep curl "lean back" error, lunge "knee over toe"
  error -- confirming per-exercise, per-fault scoring (rather than one
  generic "form score") is the right granularity.
- github.com/SravB/Computer-Vision-Weightlifting-Coach scores deadlift
  posture continuously in [0, 1] from joint positions, the same shape as
  this module's output.

See docs/ARCHITECTURE.md for the full citation list and the specific
geometric definition used for each fault.
"""
