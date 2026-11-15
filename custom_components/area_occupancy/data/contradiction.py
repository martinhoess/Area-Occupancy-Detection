"""Sensors the only person at home contradicts by being somewhere else.

A sensor stuck on (a PIR that missed its "off", a TV left paused overnight)
holds its area occupied until the hourly health check flags it, and its
fixed thresholds run to 8 or 12 hours. A person's room tracker settles it
much sooner: when the only person at home has moved to another room, that
room sees them there, and a sensor was already on before they moved and
has stayed on since, that sensor cannot be them.

The tracker alone is not trusted. Bermuda places a person in the nearest
room it knows, so someone in a room without a scanner shows up next door,
and a phone left behind keeps reporting the room it lies in. The tracked
room must therefore confirm with fresh motion of its own, and only a
stretch that began before the tracker moved can be contradicted: whatever
came on after that may well be the person.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

from homeassistant.const import STATE_HOME, STATE_UNAVAILABLE, STATE_UNKNOWN
from homeassistant.core import HomeAssistant

from .entity_type import InputType

if TYPE_CHECKING:
    from .config import PersonConfig
    from .entity import Entity

# How long the tracker must have placed the person in the other room.
TRACKER_SETTLED = timedelta(minutes=20)
# How long the tracker must be back in a flagged room to release it, so a
# few seconds of Bermuda flicker do not.
RETURN_SETTLED = timedelta(minutes=2)
# How long a sensor must have been on without a break before it can be
# contradicted, and how recent motion in the tracked room must be to
# confirm the tracker. A real person moving in a room makes a PIR pulse
# well within this; the 28-day history here has kitchen and hall PIRs at
# a few minutes at most.
ACTIVE_UNBROKEN = timedelta(minutes=30)
# Only inputs that claim someone is in the room. Doors, windows and the
# room tracker itself say nothing about who is where.
CONTRADICTABLE_TYPES = frozenset(
    {InputType.MOTION, InputType.MEDIA, InputType.APPLIANCE}
)


@dataclass(frozen=True)
class TrackedRoom:
    """Where the tracker puts the only person at home, and since when.

    Both None while the person is home alone but the tracker reports nothing
    usable: a few seconds out of BLE range must not drop existing flags.
    """

    area_id: str | None
    since: datetime | None


def tracked_room(
    hass: HomeAssistant, people: Iterable[PersonConfig]
) -> TrackedRoom | None:
    """The room the tracker reports for the only person at home.

    Other person entities that are home only through a device tracker they
    share with the tracked person do not count: that is the same human.

    Args:
        hass: Home Assistant instance
        people: The configured people

    Returns:
        The tracker's ``area_id`` attribute and last change (both None when
        the tracker is unusable), or None when anyone else is home or the
        person is not.
    """
    persons = {state.entity_id: state for state in hass.states.async_all("person")}
    for person in people:
        state = persons.get(person.person_entity)
        if person.room_sensor is None or state is None or state.state != STATE_HOME:
            continue
        own = set(state.attributes.get("device_trackers") or [])
        others = [
            other
            for entity_id, other in persons.items()
            if entity_id != person.person_entity
            and other.state == STATE_HOME
            and not own & set(other.attributes.get("device_trackers") or [])
        ]
        if others:
            return None
        tracker = hass.states.get(person.room_sensor)
        area_id = tracker.attributes.get("area_id") if tracker is not None else None
        if (
            tracker is None
            or tracker.state in (STATE_UNKNOWN, STATE_UNAVAILABLE)
            or not isinstance(area_id, str)
            or not area_id
        ):
            return TrackedRoom(area_id=None, since=None)
        return TrackedRoom(area_id=area_id, since=tracker.last_changed)
    return None


def confirms_presence(entities: Iterable[Entity], now: datetime) -> bool:
    """Whether a room shows fresh motion of its own.

    Only motion that is on and came on within ``ACTIVE_UNBROKEN``. Not a
    fading tail, which runs on for most of an hour after someone has left;
    not a long stretch, which may be the very kind of stuck sensor this
    module looks for; and not sleep, which can be derived from the same
    tracker it would confirm.
    """
    for entity in entities:
        if entity.is_stuck or entity.type.input_type != InputType.MOTION:
            continue
        since = entity.active_since
        if (
            entity.evidence is True
            and since is not None
            and now - since < ACTIVE_UNBROKEN
        ):
            return True
    return False


def contradicted_since(
    entity: Entity,
    area_id: str,
    room: TrackedRoom | None,
    confirmed: bool,
    present_here: bool,
    now: datetime,
) -> datetime | None:
    """The new ``contradicted_since`` for one entity.

    Args:
        entity: The sensor to decide for
        area_id: The HA area the sensor belongs to
        room: Where the tracker puts the only person at home, if anywhere
        confirmed: Whether that room shows fresh motion of its own
        present_here: Whether the sensor's own room does: with one person
            at home, that disproves them being elsewhere, phone or not
        now: The current time

    Returns:
        The start of the stretch the contradiction covers, or None.
    """
    if room is None or present_here:
        return None
    since = entity.active_since
    held = (
        entity.contradicted_since is not None
        and since is not None
        and since <= entity.contradicted_since
    )
    if room.area_id is None or room.since is None:
        # Tracker unusable for now: keep what holds, set nothing new.
        return entity.contradicted_since if held else None
    if room.area_id == area_id:
        # Back in this room: release once the tracker has settled here.
        if held and now - room.since < RETURN_SETTLED:
            return entity.contradicted_since
        return None
    if held:
        # The same stretch is still on: confirmation was only needed to set it.
        return entity.contradicted_since
    if (
        not confirmed
        or now - room.since < TRACKER_SETTLED
        or entity.type.input_type not in CONTRADICTABLE_TYPES
        or entity.evidence is not True
        or since is None
        or now - since < ACTIVE_UNBROKEN
        or since >= room.since
    ):
        return None
    return since
