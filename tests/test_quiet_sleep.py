"""Tests for data.quiet_sleep (sleep derived from a room tracker and house quiet)."""

from datetime import UTC, datetime, time, timedelta

import pytest

from custom_components.area_occupancy.data.quiet_sleep import QuietSleep, in_window

T0 = datetime(2026, 1, 10, 23, 0, tzinfo=UTC)


def tracker() -> QuietSleep:
    return QuietSleep(
        room_state="Bedroom",
        room_delay=timedelta(minutes=5),
        quiet=timedelta(minutes=20),
        start=time(21),
        end=time(11),
    )


def at(minutes: float) -> datetime:
    return T0 + timedelta(minutes=minutes)


def asleep(q: QuietSleep, minutes: float, local: time = time(1)) -> bool:
    return q.asleep(at(minutes), local)


@pytest.mark.parametrize(
    ("local", "expected"),
    [(time(20, 59), False), (time(21), True), (time(3), True), (time(11), False)],
)
def test_window_wraps_midnight(local: time, expected: bool) -> None:
    assert in_window(local, time(21), time(11)) is expected


def test_window_without_wrap() -> None:
    assert in_window(time(14), time(13), time(15))
    assert not in_window(time(16), time(13), time(15))


def test_asleep_after_room_delay_and_quiet() -> None:
    q = tracker()
    q.activity_changed("light.bed", at(0), "on", "off")
    q.room_changed(at(1), "Bedroom")
    assert not asleep(q, 19)  # quiet time not over
    assert asleep(q, 20)
    assert not asleep(q, 20, local=time(12))  # outside the night window


def test_room_delay_counts_from_entering() -> None:
    q = tracker()
    q.room_changed(at(0), "Bedroom")
    assert not asleep(q, 4)
    assert asleep(q, 5)


def test_short_dropout_keeps_the_stay() -> None:
    q = tracker()
    q.room_changed(at(0), "Bedroom")
    q.room_changed(at(30), "Office")  # scan gap, one hop
    assert asleep(q, 31)
    q.room_changed(at(31.5), "Bedroom")
    assert asleep(q, 32)
    assert q.room_since == at(0)


def test_long_absence_restarts_the_room_delay() -> None:
    q = tracker()
    q.room_changed(at(0), "Bedroom")
    q.room_changed(at(30), "Office")
    assert not asleep(q, 33)  # gone for more than two minutes
    q.room_changed(at(40), "Bedroom")
    assert not asleep(q, 44)
    assert asleep(q, 45)


def test_mislocation_while_active_is_no_sleep() -> None:
    """A phone mislocated to the bedroom while its owner works in the office."""
    q = tracker()
    q.activity_changed("binary_sensor.pc_used", at(0), "off", "on")
    q.room_changed(at(1), "Bedroom")
    assert not asleep(q, 30)
    q.activity_changed("binary_sensor.pc_used", at(40), "on", "off")
    assert not asleep(q, 59)
    assert asleep(q, 60)


def test_any_change_counts_as_activity() -> None:
    q = tracker()
    q.room_changed(at(0), "Bedroom")
    q.activity_changed("sensor.phone_battery", at(30), "Charging", "Not Charging")
    assert not asleep(q, 49)
    assert asleep(q, 50)


@pytest.mark.parametrize(
    ("old", "new"), [("unavailable", "off"), ("off", "unavailable")]
)
def test_unavailable_transitions_are_no_activity(old: str, new: str) -> None:
    q = tracker()
    q.room_changed(at(0), "Bedroom")
    q.activity_changed("light.bed", at(10), old, new)
    assert asleep(q, 11)


def test_unknown_room_state_changes_nothing() -> None:
    q = tracker()
    q.room_changed(at(0), "Bedroom")
    q.room_changed(at(10), "unknown")
    assert asleep(q, 20)
