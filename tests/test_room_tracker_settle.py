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

    def test_flicker_during_a_stay_keeps_the_fade(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        freezer: FrozenDateTimeFactory,
        tracker: Entity,
    ) -> None:
        """In bed, the phone jumps out and back: the fade bridges the gap."""
        self._refresh(coordinator)
        self._move(hass, tracker, "Bedroom")
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        self._move(hass, tracker, "Livingroom")
        freezer.tick(timedelta(seconds=3))
        self._move(hass, tracker, "Bedroom")
        self._refresh(coordinator)
        assert tracker.evidence is False
        assert tracker.decay.is_decaying
        freezer.tick(SETTLED)
        self._refresh(coordinator)
        assert tracker.evidence is True
        assert not tracker.decay.is_decaying

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
