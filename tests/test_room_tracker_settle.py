"""Tests for a person's room tracker counting only once its room has held."""

from datetime import timedelta
from unittest.mock import PropertyMock, patch

from freezegun.api import FrozenDateTimeFactory
import pytest

from custom_components.area_occupancy.const import ROOM_TRACKER_SETTLE_SECONDS
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import PersonConfig
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from homeassistant.core import HomeAssistant
from homeassistant.util import dt as dt_util
from tests.conftest import create_test_area

# ruff: noqa: SLF001

TRACKER = "sensor.phone_area"
PERSON = PersonConfig(
    person_entity="person.resident",
    sleep_sensors=[],
    sleep_area_id="bedroom",
    room_sensor=TRACKER,
    room_state="Bedroom",
)
SETTLED = timedelta(seconds=ROOM_TRACKER_SETTLE_SECONDS + 1)


def _custom(hass: HomeAssistant, entity_id: str) -> Entity:
    """A custom binary sensor active in ``Bedroom``, as configured for Bermuda."""
    return Entity(
        entity_id=entity_id,
        type=EntityType(
            input_type=InputType.CUSTOM_BINARY,
            weight=1.0,
            prob_given_true=0.9,
            prob_given_false=0.1,
            active_states=["Bedroom"],
        ),
        prob_given_true=0.9,
        prob_given_false=0.1,
        decay=Decay(half_life=600.0),
        hass=hass,
    )


class TestRoomTrackerSettle:
    """Jumps of a few seconds into the room count nothing; a stay counts."""

    @pytest.fixture
    def tracker(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> Entity:
        bedroom = create_test_area(coordinator, area_name="Bedroom")
        hass.states.async_set(TRACKER, "Livingroom")
        entity = _custom(hass, TRACKER)
        entity.has_new_evidence()
        bedroom.entities._entities = {entity.entity_id: entity}
        return entity

    def _refresh(self, coordinator: AreaOccupancyCoordinator) -> None:
        with patch.object(
            type(coordinator.integration_config),
            "people",
            new_callable=PropertyMock,
            return_value=[PERSON],
        ):
            coordinator._settle_room_trackers()

    def _move(self, hass: HomeAssistant, entity: Entity, room: str) -> None:
        hass.states.async_set(TRACKER, room)
        entity.has_new_evidence()  # the coordinator's state listener

    def test_short_jump_counts_nothing(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(timedelta(seconds=3))
        self._refresh(coordinator)
        assert tracker.evidence is False
        self._move(hass, tracker, "Livingroom")
        self._refresh(coordinator)
        assert not tracker.active  # no decay tail either

    def test_stay_counts_after_settling(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        assert tracker.evidence is True
        assert tracker.previous_evidence is True
        self._move(hass, tracker, "Livingroom")
        assert tracker.decay.is_decaying

    def test_flicker_during_a_stay_keeps_counting(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        """In bed, the phone jumps out for 3 s every minute: the stay goes on."""
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        for _ in range(20):
            self._move(hass, tracker, "Livingroom")
            freezer.tick(timedelta(seconds=3))
            self._move(hass, tracker, "Bedroom")
            self._refresh(coordinator)
            assert tracker.evidence is True
            assert not tracker.decay.is_decaying
            freezer.tick(timedelta(seconds=57))
            self._refresh(coordinator)

    def test_jump_long_after_a_stay_waits(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        """The bridge covers ROOM_GAP after a counted stay, not a later jump."""
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        self._move(hass, tracker, "Livingroom")
        freezer.tick(timedelta(minutes=5))
        self._move(hass, tracker, "Bedroom")
        self._refresh(coordinator)
        assert tracker.evidence is False

    def test_restart_keeps_a_counted_stay(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        tracker: Entity,
    ) -> None:
        """After a restart last_changed is the start time; the stay still counts.

        Reconcile sets the settle delay, so the restored stay flips to false
        once and sets ``counted_at``; the state began before that, so the
        bridge counts it again at once and stops the decay.
        """
        hass.states.async_set(TRACKER, "Bedroom")  # state written at start
        tracker.previous_evidence = True  # restored from the database
        with patch.object(
            type(coordinator.integration_config),
            "people",
            new_callable=PropertyMock,
            return_value=[PERSON],
        ):
            coordinator._reconcile_entity_state()
            coordinator._settle_room_trackers()  # first refresh
        assert tracker.evidence is True
        assert not tracker.decay.is_decaying

    def test_flicker_through_unavailable_keeps_counting(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        """The tracker loses the phone for 3 s: the stay goes on."""
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        self._move(hass, tracker, "unavailable")
        freezer.tick(timedelta(seconds=3))
        self._move(hass, tracker, "Bedroom")
        assert tracker.evidence is True
        assert not tracker.decay.is_decaying

    @pytest.mark.parametrize("via", [None, "before", "after"])
    def test_noise_after_leaving_does_not_chain(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
        via: str | None,
    ) -> None:
        """After a stay, the phone next door jumps in for 3 s every minute.

        Only a jump within ROOM_GAP of the stay counts; a 3 s jump does not
        move ``counted_at`` on, so the bridge ends and the room empties. The
        same holds when a jump starts or ends through ``unavailable``.
        """
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        self._move(hass, tracker, "Livingroom")
        left = dt_util.utcnow()
        counted = 0
        for _ in range(30):
            if via == "before":
                freezer.tick(timedelta(seconds=56))
                self._move(hass, tracker, "unavailable")
                freezer.tick(timedelta(seconds=1))
            else:
                freezer.tick(timedelta(seconds=57))
            self._move(hass, tracker, "Bedroom")
            counted += tracker.evidence is True
            freezer.tick(timedelta(seconds=3))
            if via == "after":
                self._move(hass, tracker, "unavailable")
            self._move(hass, tracker, "Livingroom")
            self._refresh(coordinator)
        assert counted == 2  # jumps at 57 s and 117 s, then no more
        assert tracker.decay.decay_start <= left + timedelta(seconds=120)

    def test_rebuild_keeps_settle_and_bridge(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        """A settings edit rebuilds the entities mid-gap: the bridge holds."""
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        self._move(hass, tracker, "Livingroom")
        manager = coordinator.areas["Bedroom"].entities
        with patch.object(
            manager._factory,
            "create_all_from_config",
            return_value={TRACKER: _custom(hass, TRACKER)},
        ):
            manager.refresh_from_config()
        rebuilt = manager.entities[TRACKER]
        freezer.tick(timedelta(seconds=3))
        hass.states.async_set(TRACKER, "Bedroom")
        assert rebuilt.evidence is True  # bridged, as before the rebuild
        hass.states.async_set(TRACKER, "Livingroom")
        rebuilt.has_new_evidence()
        freezer.tick(timedelta(minutes=5))
        hass.states.async_set(TRACKER, "Bedroom")
        assert rebuilt.evidence is False  # a later jump still waits

    async def test_refresh_ignores_a_short_jump(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        """Through the coordinator's refresh: a 3 s jump leaves the area as it was."""
        area = coordinator.areas["Bedroom"]
        with patch.object(
            type(coordinator.integration_config),
            "people",
            new_callable=PropertyMock,
            return_value=[PERSON],
        ):
            await coordinator.async_refresh()
            before = area.probability()
            self._move(hass, tracker, "Bedroom")
            freezer.tick(timedelta(seconds=3))
            await coordinator.async_refresh()
            assert area.probability() == pytest.approx(before)
            self._move(hass, tracker, "Livingroom")
            await coordinator.async_refresh()
            assert area.probability() == pytest.approx(before)
            self._move(hass, tracker, "Bedroom")
            freezer.tick(SETTLED)
            await coordinator.async_refresh()  # a decay tick, no state change
            assert area.probability() > before

    def test_other_sensors_count_at_once(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        tracker: Entity,
    ) -> None:
        other = _custom(hass, "sensor.other_area")
        coordinator.areas["Bedroom"].entities._entities[other.entity_id] = other
        hass.states.async_set(other.entity_id, "Bedroom")
        self._refresh(coordinator)
        assert other.settle_delay == 0.0
        assert other.evidence is True
