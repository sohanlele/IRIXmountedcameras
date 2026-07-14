"""Wristband checkout -- the front-desk step that turns a physical
wristband's BLE identifier into a real app account (Section 5.1).

Every other module in this repo (``irix.pipeline.schema``, ``run_demo``,
``run_upload``, and until now ``run_gym_demo``) treats ``member_id`` as a
plain string the caller already knows and passes in. That's fine for a
demo or an offline upload, but it isn't how a real deployment actually
learns who's wearing which band: a member checks a wristband out at the
front desk, that band gets associated with their app account for the
duration of their visit, and only from that point on does seeing that
band's BLE identifier at a station mean anything. Nothing in this repo
modeled that step before this module -- ``irix.identity.ble_pairing``
resolves *which station* a band is near, but has no concept of *whose
account* the band even belongs to.

This is deliberately just the software-side state machine (check a band
out, check it back in, resolve a band to an account) -- not a front-desk
kiosk UI, a payment/membership check, or the actual BLE-band-scan step
that would trigger a checkout in a real building (that's
hardware/product surface, out of scope here, same boundary
``ble_pairing.py``'s own docstring draws for the BLE radio stack).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class WristbandSession:
    """One checkout: this physical band, on this account, from
    ``checked_out_at`` until ``checked_in_at`` (``None`` while still out).
    """

    wristband_id: str
    member_id: str
    checked_out_at: float
    checked_in_at: Optional[float] = None

    @property
    def active(self) -> bool:
        return self.checked_in_at is None


class CheckoutRegistry:
    """Tracks which account each physical wristband is currently checked
    out to.

    One active checkout per band at a time -- a band has to be checked
    back in before the front desk can hand it to someone else, mirroring
    how a physical front-desk key/band cabinet actually works. This is
    the piece that makes ``member_id`` elsewhere in this repo a real,
    accountable identity instead of a free-text string a caller has to
    already know: ``irix.live.station_runner`` (and any future live
    pipeline) resolves a BLE-observed band id to a member id through
    here, rather than being told the member id up front.
    """

    def __init__(self):
        self._active: Dict[str, WristbandSession] = {}  # wristband_id -> current session
        self._history: Dict[str, list] = {}  # wristband_id -> past WristbandSessions

    def check_out(self, wristband_id: str, member_id: str, timestamp: float) -> WristbandSession:
        """Front desk hands ``wristband_id`` to ``member_id``'s account.

        Raises ``ValueError`` if the band is already checked out to
        someone -- it has to be checked back in first (a real front desk
        wouldn't hand out a band that's still on someone else's wrist).
        """
        existing = self._active.get(wristband_id)
        if existing is not None:
            raise ValueError(
                f"wristband {wristband_id!r} is already checked out to member "
                f"{existing.member_id!r} (since {existing.checked_out_at}) -- check it in first"
            )
        session = WristbandSession(wristband_id=wristband_id, member_id=member_id, checked_out_at=timestamp)
        self._active[wristband_id] = session
        return session

    def check_in(self, wristband_id: str, timestamp: float) -> Optional[WristbandSession]:
        """Front desk gets ``wristband_id`` back. Returns the closed-out
        session, or ``None`` if that band wasn't checked out to begin
        with (a no-op, not an error -- e.g. a band returned twice)."""
        session = self._active.pop(wristband_id, None)
        if session is None:
            return None
        session.checked_in_at = timestamp
        self._history.setdefault(wristband_id, []).append(session)
        return session

    def resolve_member(self, wristband_id: str) -> Optional[str]:
        """The account ``wristband_id`` is currently checked out to, or
        ``None`` if it isn't checked out to anyone right now. This is the
        lookup a live station runner does on every BLE-observed band id
        before it's willing to attribute any camera event to a member."""
        session = self._active.get(wristband_id)
        return session.member_id if session is not None else None

    def is_checked_out(self, wristband_id: str) -> bool:
        return wristband_id in self._active

    def active_session(self, wristband_id: str) -> Optional[WristbandSession]:
        return self._active.get(wristband_id)

    def history_for(self, wristband_id: str) -> list:
        return list(self._history.get(wristband_id, []))
