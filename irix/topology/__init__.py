"""Multi-station gym-floor topology: what a 10-camera deployment needs
that a single-station scaffold doesn't -- a registry of which camera
covers which station, and per-member handoff tracking so a lifter
walking from the squat rack to the leg press doesn't get double-counted
by two cameras during the walk, or lost entirely between them.

Nothing here does computer vision -- it consumes the same BLE RSSI
proximity signal irix.identity.ble_pairing already resolves per-snapshot,
and adds the piece that was missing for a multi-station deployment:
state over time (which station is a member *currently* assigned to) and
hysteresis (don't flip a member between two adjacent stations on RSSI
jitter alone).
"""
