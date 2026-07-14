import pytest

from irix.identity.checkout import CheckoutRegistry


def test_check_out_then_resolve_member():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    assert registry.resolve_member("band-1") == "member-alice"
    assert registry.is_checked_out("band-1") is True


def test_unresolved_band_returns_none():
    registry = CheckoutRegistry()
    assert registry.resolve_member("band-1") is None
    assert registry.is_checked_out("band-1") is False


def test_check_out_already_checked_out_band_raises():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    with pytest.raises(ValueError, match="already checked out"):
        registry.check_out("band-1", "member-bob", timestamp=1.0)


def test_check_in_frees_the_band_for_reuse():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    registry.check_in("band-1", timestamp=10.0)
    assert registry.resolve_member("band-1") is None
    assert registry.is_checked_out("band-1") is False

    # now bob can check the same physical band out
    registry.check_out("band-1", "member-bob", timestamp=11.0)
    assert registry.resolve_member("band-1") == "member-bob"


def test_check_in_unknown_band_is_a_noop_not_an_error():
    registry = CheckoutRegistry()
    result = registry.check_in("band-1", timestamp=0.0)
    assert result is None


def test_active_session_reflects_current_checkout():
    registry = CheckoutRegistry()
    session = registry.check_out("band-1", "member-alice", timestamp=5.0)
    assert session.wristband_id == "band-1"
    assert session.member_id == "member-alice"
    assert session.checked_out_at == 5.0
    assert session.active is True

    closed = registry.check_in("band-1", timestamp=15.0)
    assert closed.active is False
    assert closed.checked_in_at == 15.0
    assert registry.active_session("band-1") is None


def test_history_accumulates_past_checkouts():
    registry = CheckoutRegistry()
    registry.check_out("band-1", "member-alice", timestamp=0.0)
    registry.check_in("band-1", timestamp=5.0)
    registry.check_out("band-1", "member-bob", timestamp=6.0)
    registry.check_in("band-1", timestamp=10.0)

    history = registry.history_for("band-1")
    assert len(history) == 2
    assert [s.member_id for s in history] == ["member-alice", "member-bob"]
