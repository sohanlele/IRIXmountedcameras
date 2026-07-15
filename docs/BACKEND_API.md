# Backend API (Priority 11)

`irix.backend.studio_api.StudioBackendAPI` is the concrete surface this
repo exposes for a future IRIX Studio backend to call. **This repo does
not build Studio** -- no member-facing app, no staff console UI. That's
explicitly out of scope per the founding brief. What exists here is the
well-typed, tested set of operations Studio will eventually need,
implemented against the real state this repo already tracks
(`irix.identity.checkout.CheckoutRegistry`,
`irix.live.gym_runner.GymSessionRunner`,
`irix.live.station_runner.StationSessionRunner`).

## Construction

One `StudioBackendAPI` instance per gym deployment:

```python
from irix.backend.studio_api import StudioBackendAPI

api = StudioBackendAPI(
    checkout_registry=checkout_registry,   # required
    gym_session_runner=gym_session_runner, # optional -- see below
)
```

`gym_session_runner` is optional. Front-desk-only operations
(`assign_wristband`, `return_wristband`, `query_assignment`,
`query_battery`) work with just a `CheckoutRegistry` -- no live gym
loop required. Operations that need real-time session state
(`start_session`, `end_session`, `request_placement_change`) raise
`StudioAPIError` clearly if called without a `gym_session_runner`,
rather than silently no-op-ing.

## Operations

- **`assign_wristband(wristband_id, member_id, at_time)`** -- front-desk
  checkout. Delegates to `CheckoutRegistry.check_out`.
- **`return_wristband(wristband_id, at_time)`** -- physical hand-back.
  Checks the band in *and*, if a live `gym_session_runner` is
  configured, ends and forgets its `WorkoutStateMachine` too
  (`GymSessionRunner.record_wristband_returned`). Calling this on an
  already-returned band is not an error -- returns `was_active: False`.
- **`start_session(wristband_id)`** -- in this repo's presence-driven
  model, sessions auto-start from BLE presence rather than being
  explicitly commanded (see `irix.live.gym_runner`'s module docstring).
  So this is a **confirmation**, not a command: it reports whether a
  session for this band is currently active, and raises
  `StudioAPIError` only if the band has no assignment at all.
- **`end_session(wristband_id)`** -- a real command, distinct from
  `return_wristband`: ends this band's tracked workout
  (`GymSessionRunner.force_end_session`) without processing a physical
  return, for a Studio operator ending a member's workout early while
  the band is still on their wrist.
- **`query_battery(wristband_id)`** -- always returns
  `status: "unknown"`. No battery-level signal exists anywhere in this
  repo yet (see `docs/WRISTBAND_SYSTEM.md`); this never fabricates a
  number, consistent with this repo's "unknown over incorrect"
  principle applied everywhere else (load detection, identity
  resolution, etc).
- **`query_assignment(wristband_id)`** -- current member assignment, if
  any.
- **`query_wristband_status(wristband_id)`** -- everything this repo
  can honestly say about one band right now in one call: assignment,
  current station, workout phase + health flags, placement state,
  clock-sync confidence, and battery. Degrades gracefully (fields go
  `None`/`"unavailable"`, never raises) when no live
  `gym_session_runner` is configured -- "what do we know about this
  band" is a reasonable question to ask of an idle deployment.
- **`request_placement_change(wristband_id, to_side, at_time)`** --
  Priority 4's placement backend entry point, exposed at the
  Studio-facing layer. Resolves the band's currently-active station via
  `GymCoordinator` and delegates to that station's
  `StationSessionRunner.request_wristband_placement_change`.

## Event delivery

Structured workout events (`irix.pipeline.schema`) are delivered
separately from this request/response API, via the `on_events`/
`on_gym_events` callbacks already wired into
`StationSessionRunner`/`GymSessionRunner` (see `docs/API_SPEC.md`).
Every event now carries `schema_version` (`EVENT_SCHEMA_VERSION` in
`irix/pipeline/schema.py`) so a real Studio backend can version-check
incoming events independent of this request API's own versioning.

## What's still not real

- No actual network transport (REST/gRPC/etc) wraps
  `StudioBackendAPI` -- it's a plain Python class today, callable
  in-process. Wrapping it behind a real API server is future work once
  an actual Studio project exists to consume it.
- `query_battery` will need a real implementation once wristband
  hardware reports battery level (see `docs/WRISTBAND_SYSTEM.md`).
