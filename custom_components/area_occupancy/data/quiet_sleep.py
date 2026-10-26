"""Sleep derived from quiet: a room tracker reports the sleep room while the house is quiet at night.

For people without a sleep sensor of their own. A room tracker (e.g. a Bermuda area sensor
reporting ``Bedroom``) alone is too noisy: it mislocates a phone for minutes at a time. Requiring
that nothing in the house has been active for a while filters every mislocation that happens while
the person is up and about, and the night window keeps naps and empty daytime rooms out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta

# Short drop-outs of the room tracker (a scan gap, a hop to the next room) do not end the stay.
ROOM_GAP = timedelta(minutes=2)
ACTIVE_STATES = frozenset({"on", "playing", "paused", "buffering"})
IGNORED_STATES = frozenset({"unknown", "unavailable", ""})


def in_window(local: time, start: time, end: time) -> bool:
    """Whether ``local`` lies in [start, end), which may wrap past midnight."""
    if start <= end:
        return start <= local < end
    return local >= start or local < end


@dataclass
class QuietSleep:
    """Tracks room stay and house activity for one person; pure, the caller feeds events."""

    room_state: str
    room_delay: timedelta
    quiet: timedelta
    start: time
    end: time
    room_since: datetime | None = None
    room_left: datetime | None = None
    last_activity: datetime | None = None
    active: set[str] = field(default_factory=set)

    def room_changed(self, when: datetime, state: str | None) -> None:
        """Room tracker reported ``state`` at ``when``.

        Unknown or unavailable counts as leaving: a short scan gap is covered by
        ROOM_GAP, a tracker that lost the phone for good ends the stay.
        """
        if state == self.room_state:
            if self.room_since is None or (
                self.room_left is not None and when - self.room_left > ROOM_GAP
            ):
                self.room_since = when
            self.room_left = None
        elif self.room_since is not None and self.room_left is None:
            self.room_left = when

    def activity_changed(
        self, entity_id: str, when: datetime, old: str | None, new: str | None
    ) -> None:
        """An activity entity changed; any real change counts as activity."""
        if new in IGNORED_STATES or new is None:
            # A TV cut at the power strip goes playing -> unavailable: no longer active.
            self.active.discard(entity_id)
            return
        if new in ACTIVE_STATES:
            self.active.add(entity_id)
        else:
            self.active.discard(entity_id)
        # Coming back from unavailable (restart, radio drop) is no sign of life.
        if old is not None and old not in IGNORED_STATES and old != new:
            self.last_activity = max(self.last_activity or when, when)

    def assume_asleep(self, now: datetime) -> None:
        """Restore a sleep that was on before a restart: room delay and quiet count as passed.

        Every entity's last_changed is the start time after a restart; without this
        the sensor would drop to off for the quiet time in the middle of the night.
        """
        self.room_since = now - self.room_delay
        self.room_left = None
        self.last_activity = None

    def in_room(self, now: datetime) -> bool:
        """In the sleep room now; a drop-out up to ROOM_GAP still counts."""
        if self.room_since is None:
            return False
        return self.room_left is None or now - self.room_left <= ROOM_GAP

    def asleep(self, now: datetime, local: time) -> bool:
        """Asleep: in the room for ``room_delay``, nothing active for ``quiet``, inside the window."""
        if not in_window(local, self.start, self.end) or self.active:
            return False
        if self.room_since is None or not self.in_room(now):
            return False
        if now - self.room_since < self.room_delay:
            return False
        return self.last_activity is None or now - self.last_activity >= self.quiet
