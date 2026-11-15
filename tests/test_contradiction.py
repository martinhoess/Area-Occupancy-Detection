"""Tests for sensors contradicted by the only person at home being elsewhere."""

from datetime import timedelta
from unittest.mock import PropertyMock, patch

import pytest

from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.config import PersonConfig
from custom_components.area_occupancy.data.contradiction import (
    ACTIVE_UNBROKEN,
    RETURN_SETTLED,
    TRACKER_SETTLED,
    TrackedRoom,
    confirms_presence,
    contradicted_since,
    tracked_room,
)
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from homeassistant.const import STATE_OFF, STATE_ON
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
LATER = TRACKER_SETTLED + timedelta(minutes=1)


def _sensor(
    hass: HomeAssistant,
    entity_id: str,
    state: str,
    input_type: InputType = InputType.MOTION,
    on_for: timedelta = timedelta(0),
) -> Entity:
    """A real entity in ``state`` that changed ``on_for`` ago."""
    hass.states.async_set(entity_id, state)
    return Entity(
        entity_id=entity_id,
        type=EntityType(
            input_type=input_type,
            weight=1.0,
            prob_given_true=0.9,
            prob_given_false=0.05,
            active_states=[STATE_ON],
        ),
        prob_given_true=0.9,
        prob_given_false=0.05,
        decay=Decay(half_life=300.0),
        hass=hass,
        last_updated=dt_util.utcnow() - on_for,
        previous_evidence=state == STATE_ON,
    )


def _at_home(hass: HomeAssistant, *people: str) -> None:
    """The resident, an admin sharing their phone, and an independent guest."""
    trackers = {
        "person.resident": ["device_tracker.resident", "device_tracker.phone"],
        "person.admin": ["device_tracker.laptop", "device_tracker.phone"],
        "person.guest": ["device_tracker.guest"],
    }
    for person, own in trackers.items():
        hass.states.async_set(
            person,
            "home" if person in people else "not_home",
            {"device_trackers": own},
        )


def _room(area_id: str, ago: timedelta) -> TrackedRoom:
    return TrackedRoom(area_id=area_id, since=dt_util.utcnow() - ago)


class TestTrackedRoom:
    """Where the tracker puts the only person at home."""

    def test_names_area_and_change(self, hass: HomeAssistant) -> None:
        _at_home(hass, "person.resident")
        hass.states.async_set(TRACKER, "Office", {"area_id": "office"})
        room = tracked_room(hass, [PERSON])
        assert room is not None
        assert room.area_id == "office"
        assert room.since == hass.states.get(TRACKER).last_changed

    def test_person_sharing_the_phone_is_the_same_human(
        self, hass: HomeAssistant
    ) -> None:
        _at_home(hass, "person.resident", "person.admin")
        hass.states.async_set(TRACKER, "Office", {"area_id": "office"})
        assert tracked_room(hass, [PERSON]) is not None

    def test_guest_home_decides_nothing(self, hass: HomeAssistant) -> None:
        _at_home(hass, "person.resident", "person.guest")
        hass.states.async_set(TRACKER, "Office", {"area_id": "office"})
        assert tracked_room(hass, [PERSON]) is None

    def test_tracked_person_away_decides_nothing(self, hass: HomeAssistant) -> None:
        _at_home(hass, "person.guest")
        hass.states.async_set(TRACKER, "Office", {"area_id": "office"})
        assert tracked_room(hass, [PERSON]) is None

    @pytest.mark.parametrize(
        ("state", "attributes"),
        [
            ("unavailable", {"area_id": "office"}),
            ("unknown", {}),
            ("Office", {}),
        ],
    )
    def test_unusable_tracker_reports_an_empty_room(
        self, hass: HomeAssistant, state: str, attributes: dict
    ) -> None:
        _at_home(hass, "person.resident")
        hass.states.async_set(TRACKER, state, attributes)
        assert tracked_room(hass, [PERSON]) == TrackedRoom(area_id=None, since=None)

    def test_person_without_tracker_decides_nothing(self, hass: HomeAssistant) -> None:
        _at_home(hass, "person.resident")
        untracked = PersonConfig(
            person_entity="person.resident", sleep_sensors=[], sleep_area_id="bedroom"
        )
        assert tracked_room(hass, [untracked]) is None


class TestConfirmsPresence:
    """Only motion that just came on confirms the tracker."""

    def test_motion_that_just_came_on(self, hass: HomeAssistant) -> None:
        motion = _sensor(hass, "binary_sensor.desk", STATE_ON)
        assert confirms_presence([motion], dt_util.utcnow())

    def test_motion_on_for_long_proves_nothing(self, hass: HomeAssistant) -> None:
        motion = _sensor(hass, "binary_sensor.desk", STATE_ON, on_for=ACTIVE_UNBROKEN)
        assert not confirms_presence([motion], dt_util.utcnow())

    def test_fading_tail_proves_nothing(self, hass: HomeAssistant) -> None:
        """It keeps fading for most of an hour after someone left."""
        motion = _sensor(hass, "binary_sensor.desk", STATE_OFF)
        motion.decay.start_decay()
        assert not confirms_presence([motion], dt_util.utcnow())

    @pytest.mark.parametrize("input_type", [InputType.SLEEP, InputType.MEDIA])
    def test_other_inputs_prove_nothing(
        self, hass: HomeAssistant, input_type: InputType
    ) -> None:
        sensor = _sensor(hass, "binary_sensor.x", STATE_ON, input_type)
        assert not confirms_presence([sensor], dt_util.utcnow())

    def test_stuck_motion_proves_nothing(self, hass: HomeAssistant) -> None:
        motion = _sensor(hass, "binary_sensor.desk", STATE_ON)
        motion.stuck_since = motion.active_since
        assert not confirms_presence([motion], dt_util.utcnow())


class TestContradictedSince:
    """The decision for one sensor in the living room."""

    def _tv(self, hass: HomeAssistant, **kwargs: object) -> Entity:
        defaults = {"input_type": InputType.MEDIA, "on_for": timedelta(hours=15)}
        defaults.update(kwargs)
        return _sensor(hass, "media_player.tv", STATE_ON, **defaults)  # type: ignore[arg-type]

    def _decide(
        self,
        tv: Entity,
        room: TrackedRoom | None,
        confirmed: bool = True,
        present_here: bool = False,
    ) -> object:
        return contradicted_since(
            tv, "living_room", room, confirmed, present_here, dt_util.utcnow()
        )

    def test_stretch_from_before_the_move(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        assert self._decide(tv, _room("office", LATER)) == tv.active_since

    @pytest.mark.parametrize(
        "input_type", [InputType.MOTION, InputType.MEDIA, InputType.APPLIANCE]
    )
    def test_contradictable_inputs(
        self, hass: HomeAssistant, input_type: InputType
    ) -> None:
        tv = self._tv(hass, input_type=input_type)
        assert self._decide(tv, _room("office", LATER)) is not None

    @pytest.mark.parametrize(
        "input_type", [InputType.DOOR, InputType.SLEEP, InputType.CUSTOM_BINARY]
    )
    def test_other_inputs(self, hass: HomeAssistant, input_type: InputType) -> None:
        tv = self._tv(hass, input_type=input_type)
        assert self._decide(tv, _room("office", LATER)) is None

    def test_stretch_from_after_the_move(self, hass: HomeAssistant) -> None:
        """Phone left in the office: whatever came on since may be the person."""
        tv = self._tv(hass, on_for=ACTIVE_UNBROKEN + timedelta(minutes=5))
        assert self._decide(tv, _room("office", timedelta(hours=2))) is None

    def test_short_stretch(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass, on_for=timedelta(minutes=5))
        assert self._decide(tv, _room("office", LATER)) is None

    def test_tracker_not_settled(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        assert self._decide(tv, _room("office", timedelta(minutes=5))) is None

    def test_unconfirmed(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        assert self._decide(tv, _room("office", LATER), confirmed=False) is None

    def test_off(self, hass: HomeAssistant) -> None:
        tv = _sensor(hass, "media_player.tv", STATE_OFF, on_for=timedelta(hours=3))
        assert self._decide(tv, _room("office", LATER)) is None

    def test_held_without_confirmation(self, hass: HomeAssistant) -> None:
        """Confirmation sets a flag; keeping it only needs the same stretch."""
        tv = self._tv(hass)
        tv.contradicted_since = tv.active_since
        assert self._decide(tv, _room("bedroom", timedelta(seconds=5)), False) == (
            tv.active_since
        )

    def test_flicker_into_the_room_keeps_it(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        tv.contradicted_since = tv.active_since
        room = _room("living_room", timedelta(seconds=10))
        assert self._decide(tv, room) == tv.active_since

    def test_settled_return_releases_it(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        tv.contradicted_since = tv.active_since
        room = _room("living_room", RETURN_SETTLED)
        assert self._decide(tv, room) is None

    def test_fresh_motion_here_releases_it(self, hass: HomeAssistant) -> None:
        """Back on the sofa, phone left on the desk."""
        tv = self._tv(hass)
        tv.contradicted_since = tv.active_since
        assert self._decide(tv, _room("office", LATER), present_here=True) is None

    def test_fresh_motion_here_prevents_it(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        assert self._decide(tv, _room("office", LATER), present_here=True) is None

    def test_unusable_tracker_keeps_it(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        tv.contradicted_since = tv.active_since
        empty = TrackedRoom(area_id=None, since=None)
        assert self._decide(tv, empty, confirmed=False) == tv.active_since

    def test_unusable_tracker_sets_nothing(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        assert self._decide(tv, TrackedRoom(area_id=None, since=None)) is None

    def test_no_tracked_room_releases_it(self, hass: HomeAssistant) -> None:
        tv = self._tv(hass)
        tv.contradicted_since = tv.active_since
        assert self._decide(tv, None) is None


class TestCoordinator:
    """The refresh flags, holds and releases sensors in the other rooms."""

    @pytest.fixture
    def rooms(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> tuple[Entity, Entity]:
        """An office with fresh motion and a living room with a long-paused TV."""
        office = create_test_area(coordinator, area_name="Office")
        living = create_test_area(coordinator, area_name="Living Room")
        office.config.area_id = "office"
        living.config.area_id = "living_room"
        tv = _sensor(
            hass,
            "media_player.tv",
            STATE_ON,
            InputType.MEDIA,
            on_for=timedelta(hours=15),
        )
        _at_home(hass, "person.resident", "person.admin")
        hass.states.async_set(TRACKER, "Office", {"area_id": "office"})
        desk = _sensor(hass, "binary_sensor.desk", STATE_ON)
        office.entities._entities = {desk.entity_id: desk}
        living.entities._entities = {tv.entity_id: tv}
        return desk, tv

    def _refresh(
        self, coordinator: AreaOccupancyCoordinator, after: timedelta = LATER
    ) -> None:
        with patch.object(
            type(coordinator.integration_config),
            "people",
            new_callable=PropertyMock,
            return_value=[PERSON],
        ):
            coordinator._flag_contradicted_sensors(dt_util.utcnow() + after)

    def test_tv_in_the_other_room_stops_counting(
        self, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        desk, tv = rooms
        self._refresh(coordinator)
        assert tv.is_stuck
        assert not desk.is_stuck

    def test_silent_tracked_room_leaves_everything(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        """Somewhere without a scanner: the room Bermuda names is silent."""
        desk, tv = rooms
        hass.states.async_set(desk.entity_id, STATE_OFF)
        desk.has_new_evidence()
        self._refresh(coordinator)
        assert not tv.is_stuck

    def test_flag_holds_when_confirmation_goes(
        self, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        """Sitting still: the desk sensor stays on past the freshness window."""
        _, tv = rooms
        self._refresh(coordinator)
        self._refresh(coordinator, after=timedelta(hours=2))
        assert tv.is_stuck

    def test_coming_back_counts_again(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        _, tv = rooms
        self._refresh(coordinator)
        hass.states.async_set(TRACKER, "Livingroom", {"area_id": "living_room"})
        self._refresh(coordinator, after=timedelta(0))
        assert tv.is_stuck  # a flicker
        self._refresh(coordinator, after=RETURN_SETTLED)
        assert not tv.is_stuck

    def test_guest_arriving_counts_again(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        _, tv = rooms
        self._refresh(coordinator)
        _at_home(hass, "person.resident", "person.guest")
        self._refresh(coordinator)
        assert not tv.is_stuck

    def test_tracker_dropout_keeps_the_flag(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        """A few seconds out of BLE range while sitting still at the desk."""
        _, tv = rooms
        self._refresh(coordinator)
        hass.states.async_set(TRACKER, "unavailable")
        self._refresh(coordinator, after=timedelta(hours=1))
        hass.states.async_set(TRACKER, "Office", {"area_id": "office"})
        self._refresh(coordinator, after=timedelta(hours=1))
        assert tv.is_stuck

    def test_motion_in_the_flagged_room_counts_it_again(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        """Back on the sofa without the phone: the room's own PIR fires."""
        _, tv = rooms
        self._refresh(coordinator)
        sofa = _sensor(hass, "binary_sensor.sofa", STATE_OFF)
        living = coordinator.areas["Living Room"]
        living.entities._entities[sofa.entity_id] = sofa
        hass.states.async_set(sofa.entity_id, STATE_ON)
        sofa.has_new_evidence()
        self._refresh(coordinator, after=timedelta(0))
        assert not tv.is_stuck

    def test_end_of_a_contradicted_stretch_does_not_fade(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator, rooms: tuple
    ) -> None:
        """It never counted, so its end must not count it after all."""
        _, tv = rooms
        self._refresh(coordinator)
        hass.states.async_set(tv.entity_id, STATE_OFF)
        tv.has_new_evidence()
        assert not tv.decay.is_decaying
