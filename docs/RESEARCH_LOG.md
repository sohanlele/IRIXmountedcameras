# Research log

Chronological log of research that drove real architectural decisions in
this repo. Each entry: what was researched, what it changed, and the
source. Detailed reasoning for each decision lives in
`docs/ARCHITECTURE.md`; this is the index of *why*, dated.

## Pose estimation model choice

Generic single-RGB-camera human pose estimation was assessed as a
solved, commodity problem, not something worth training a custom model
for. Ultralytics YOLO-Pose (`yolov8n-pose.pt`, pretrained on COCO
17-keypoint layout) chosen as a real, freely available, auto-downloading
checkpoint requiring no gym-specific data collection. See
`docs/ARCHITECTURE.md`'s "Model weights" section.

## Weight recognition: VLM over QR stickers or OCR

Both QR-sticker-based and OCR-based plate reading were evaluated and
ruled out in favor of a VLM-based classifier (structured JSON output via
`google-genai`) -- see `docs/ARCHITECTURE.md`'s "Weight recognition" section
for the full tradeoff (QR requires a physical sticker per plate/dumbbell,
an operational burden this repo's design avoids; OCR on plate-printed
numbers is unreliable at camera angles/distances typical of a gym
floor).

## Real-time audio coaching layer: Vision Agents (Stream) evaluated, not adopted here

2026-07-14 survey of open-source tooling identified
[`GetStream/Vision-Agents`](https://github.com/GetStream/Vision-Agents)
(real-time pose tracking + LLM voice feedback over low-latency WebRTC,
with a runnable gym-coach tutorial) as a credible transport+voice layer
for whoever builds `irix-mvp-app`'s coaching layer. Deliberately **not**
added as a dependency of this repo (would blur the "no spoken text
originates here" boundary, and its dependency graph -- `aiortc`,
`onnxruntime`, `fastapi` -- is shaped for a standalone service, not this
repo's edge-inference library). Verified gotcha: `vision-agents` 0.6.6
fails to import on Python 3.10 (`from typing import Self`, 3.11+ only)
despite claiming 3.10 support in its own package classifiers -- confirmed
by actually installing and importing it. See
`docs/ARCHITECTURE.md`'s "Where this repo ends" section and
`irix_competitive_research.md` (business workspace) for the fuller
competitive survey this was part of.

## Privacy/legal positioning: BIPA scope

2026-07-14 review: under Illinois's BIPA and similar state biometric-
privacy laws, recording video is not the regulated act -- building a
faceprint from it is. This repo's identity model (wristband-resolved,
never a face-geometry embedding) sits outside BIPA's scope by
construction, not as a compliance retrofit -- a direct consequence of
choosing wristband-based identity for reliability reasons before privacy
law was a factor. [BIPA scope and biometric identifiers](https://www.biometricupdate.com/202407/scope-and-contours-of-bipa-biometric-identifiers-and-information).
GroeFit (a commercial-gym camera-analytics competitor) markets the same
"no facial recognition" property explicitly as a privacy differentiator
-- worth citing in pitch material, per `irix_competitive_research.md`.

## Motion-correlation identity disambiguation: prior art

`irix.identity.motion_correlation`'s wearable-IMU-to-tracked-body-motion
matching approach was arrived at independently, then confirmed as a
published, validated technique: "Person Re-Identification Using Deep
Modeling of Temporally Correlated Inertial Motion Patterns" (Sensors,
2020), validated across 86 subjects. [Paper](https://www.mdpi.com/1424-8220/20/3/949/htm).
Known limitation, stated honestly in that module's own docstring: no 3D
wrist-orientation estimation to strip gravity's changing projection out
of the raw accelerometer signal. A Versatile Quaternion Filter (VQF,
estimates orientation from accel+gyro, calibrated against a known static
pose) is the standard fix if correlation accuracy needs it in a real
crowded-station deployment -- not yet needed against synthetic
validation data. [VQF background](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12736794/).

## Velocity-based training thresholds

`irix.fatigue`'s velocity-loss zone boundaries were cross-checked against
published autoregulation literature: velocity-loss <=25% within a set
correlates with the best 1RM strength gains, and 20% velocity-loss is a
commonly used real-time fatigue readout in VBT coaching practice.
[Velocity-loss thresholds and autoregulation (2025)](https://journals.sagepub.com/doi/10.1177/17479541251339905).
GymAware (linear position transducer, considered the VBT gold standard)
explicitly corrects for the angle between its sensor cable and the true
bar path -- i.e. doesn't assume perfectly frontal mounting.
[GymAware buyers guide](https://gymaware.com/velocity-based-training-buyers-guide/).
`irix.barbell.tracker`'s camera-tilt correction (see
`docs/ARCHITECTURE.md`) follows this precedent. Separately: wrist/arm-
worn IMU VBT devices (e.g. the discontinued PUSH Band) are noted in the
literature as less reliable than camera- or LPT-based bar tracking for
velocity specifically -- supports this repo's design of camera bar-path
as the primary velocity signal, wristband IMU as a cross-check/rep-count
fusion input, never the sole velocity source.

## Barbell/plate detection dataset

COCO and other standard pretrained object-detection models have no
barbell/plate/dumbbell class. Roboflow's "Barbells Detector" dataset (92
labeled images, pretrained model available via their API) identified as
a starting point for `FreeWeightDetector` -- not yet used; fine-tuning
vs. a hosted inference API remains an open, real decision (account/cost/
accuracy tradeoffs). See `docs/IMPLEMENTATION_STATUS.md`.

## Competitive landscape survey (2026-07-14)

Full survey across consumer fitness-AI, commercial-gym analytics, sports
tracking, and open-source tooling in `irix_competitive_research.md`
(business workspace, not this repo). Highlights not already covered
above: Peloton IQ/Kemtai confirm the detect -> pose -> classify -> count
pipeline shape is the right one; Tempo's time-of-flight depth camera +
barbell tracking is the clearest hardware upgrade path for bar-velocity
accuracy this repo doesn't currently use (a real hardware lift, not a
near-term software change); SkillCorner/SoccerNet's multi-camera
handoff/re-ID architectures are conceptually the same shape as
`GymCoordinator`, and a SoccerNet-style "gym floor minimap" is flagged as
a cheap, high-visual-impact dashboard opportunity on top of state this
repo already tracks (`StationRegistry`/`GymCoordinator`).

## Software wristband + BLE gateway simulator

See `docs/ARCHITECTURE.md`'s "Software wristband + BLE gateway
simulator" section (2026-07-14) for the full design rationale --
summarized in `docs/WRISTBAND_SYSTEM.md`.
