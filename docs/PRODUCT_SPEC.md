# Product spec

## What this repo is

IRIX is an edge-computing system installed inside a commercial gym:
fixed, mounted cameras covering the workout floor, plus a BLE wristband
(IMU) issued to each member at check-in. This repository is the
**camera/edge software layer**: it turns camera + wristband data into
structured workout events (rep counted, set started/ended, station
handoff, weight confirmed, fatigue summary) and hands them off through a
versioned event API. See `irix/pipeline/schema.py` for the exact event
contract and `docs/API_SPEC.md` for the full reference.

It does **not** decide what to tell a member, generate coaching copy,
run any UI, do scheduling/nutrition, or run facial recognition. Those
are explicitly out of scope here -- see "Where this repo ends" below.

## Who it's for

Two audiences, indirectly:

- **Gym operators** (the actual buyer/pitch target -- see
  `IRIX x 24 HF.pdf` in the business workspace for the current pitch)
  get equipment-usage visibility and can offer members automated,
  accurate rep/set tracking without staff manually logging anything.
- **Members** get their workout automatically tracked -- reps, sets,
  rest, tempo, estimated load, fatigue trend -- without wearing a
  dedicated fitness watch or manually logging sets in an app. This repo
  computes that data; a separate app (`irix-mvp-app`) is what a member
  actually sees.

## Where this repo ends and other repos begin

- **`jeffreyjy/irix-mvp-app`** (FastAPI backend + iOS app) owns the
  member-facing UI and the AI-generated coaching copy/instructions. It
  consumes this repo's `CameraEvent` stream (see `docs/API_SPEC.md`) and
  decides what to say and show. As of the last check against its
  `api/v1` (see `docs/ARCHITECTURE.md`'s "Where this repo ends" section),
  it does not yet expose a live-camera-event ingestion endpoint --
  `HTTPCloudSync` in `irix/pipeline/cloud_sync.py` is a placeholder
  pointed at wherever that route ends up.
- **IRIX Studio** (separate project, not in this repo) is the front-desk
  check-in application: it assigns a member id, session id, and
  wristband id to a visit. This repo's responsibility starts once that
  assignment happens (`irix.identity.checkout.CheckoutRegistry` is the
  software-side record of it).
- **Wristband firmware and BLE radio hardware** are explicitly out of
  this repo's scope (pure software scaffold) -- see
  `docs/WRISTBAND_SYSTEM.md` for hardware recommendations and what's
  simulated vs. real today.

## Explicit non-goals (per the founding brief)

Do not build in this repo: a mobile app, an AI coach, workout
recommendations, conversational AI, scheduling, nutrition, or the IRIX
Studio check-in UI. Do not implement facial recognition or any
biometric-identifier extraction (face/hand geometry, iris, fingerprint,
voiceprint) -- identity here is always "which wristband," never "whose
face." Unknown is always preferred to a guessed/incorrect identity,
exercise, rep count, or weight -- see `docs/ARCHITECTURE.md` for the
specific places this shows up (e.g. crowded-station disambiguation
leaving an unresolved window unattributed rather than guessing).

## What the system produces (mapped to the current codebase)

| Capability | Status | Module |
|---|---|---|
| Which member is where | Real | `irix.identity`, `irix.topology.handoff` |
| What exercise they're doing | Real (fixed per-station config, not auto-classified) | `irix.rep_counting.exercises` |
| Set/rest start & end | Real | `irix.pipeline.events.RestGapSetBoundaryDetector` |
| Rep count | Real, camera+IMU fused | `irix.rep_counting`, `irix.fusion.rep_fusion` |
| Range of motion / tempo | Real (angle-based) | `irix.rep_counting.state_machine` |
| Velocity | Real (deg/s always; m/s when a barbell is tracked) | `irix.rep_counting`, `irix.barbell.tracker` |
| Fatigue metrics | Real | `irix.fatigue` |
| Exercise confidence / tracking confidence | Partial (`tracking_confidence` on reps; no per-exercise classifier confidence, since exercise is configured per station, not classified) | `irix.rep_counting.state_machine` |
| Movement quality (form) | Real, rule-based | `irix.form` |
| Estimated load | Partial (VLM classifier real; barbell/plate detector is an untrained stub -- see `docs/WRISTBAND_SYSTEM.md`/`ARCHITECTURE.md`) | `irix.weight_recognition` |

See `docs/IMPLEMENTATION_STATUS.md` for the full subsystem-by-subsystem
breakdown and `docs/ROADMAP.md` for what's next.
