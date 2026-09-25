"""Tests for the home-entity veto and the set_override service."""

from datetime import timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, PropertyMock, patch

import pytest
import voluptuous as vol

from custom_components.area_occupancy.area.area import Area
from custom_components.area_occupancy.const import (
    DOMAIN,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from custom_components.area_occupancy.coordinator import AreaOccupancyCoordinator
from custom_components.area_occupancy.data.activity import ActivityId
from custom_components.area_occupancy.data.decay import Decay
from custom_components.area_occupancy.data.entity import Entity
from custom_components.area_occupancy.data.entity_type import EntityType, InputType
from custom_components.area_occupancy.service import SET_OVERRIDE_SCHEMA, _set_override
from custom_components.area_occupancy.utils import nobody_home
from homeassistant.const import STATE_OFF, STATE_ON, STATE_UNAVAILABLE
from homeassistant.core import HomeAssistant, ServiceCall
from homeassistant.exceptions import ServiceValidationError
from homeassistant.util import dt as dt_util

# ruff: noqa: SLF001

HOME_ENTITY = "binary_sensor.anybody_home"


def _home_entity(coordinator: AreaOccupancyCoordinator, entity_id: str) -> Any:
    """Point the integration at a home entity for the duration of a test."""
    return patch.object(
        type(coordinator.integration_config),
        "home_entity",
        new_callable=PropertyMock,
        return_value=entity_id,
    )


def _call(**data: Any) -> Mock:
    """Build a service call carrying the given data."""
    call = Mock(spec=ServiceCall)
    call.data = data
    return call


class TestNobodyHome:
    """The state mapping that decides what counts as an empty home."""

    @pytest.mark.parametrize(
        ("entity_id", "state", "expected"),
        [
            # Plain on/off entities: on means someone is home.
            ("binary_sensor.anybody_home", "off", True),
            ("binary_sensor.anybody_home", "OFF", True),
            ("binary_sensor.anybody_home", "on", False),
            ("input_boolean.anybody_home", "off", True),
            # Trackers report a zone *name* while someone is in one.
            ("person.resident", "home", False),
            ("person.resident", "not_home", True),
            ("person.resident", "Town", True),
            ("person.resident", "Work", True),
            ("device_tracker.phone", "home", False),
            ("device_tracker.phone", "not_home", True),
            # zone.home carries a head count.
            ("zone.home", "0", True),
            ("zone.home", "0.0", True),
            ("zone.home", "1", False),
            ("zone.home", "3", False),
            # Any other zone says nothing about this house.
            ("zone.school", "0", False),
            # Head counts from a plain sensor still work.
            ("sensor.people_home", "0", True),
            ("sensor.people_home", "2", False),
        ],
    )
    def test_states(
        self, hass: HomeAssistant, entity_id: str, state: str, expected: bool
    ) -> None:
        """Each accepted entity kind maps onto the same answer."""
        hass.states.async_set(entity_id, state)
        assert nobody_home(hass, entity_id) is expected

    @pytest.mark.parametrize(
        "entity_id",
        ["binary_sensor.anybody_home", "person.resident", "zone.home"],
    )
    @pytest.mark.parametrize("state", ["unknown", "unavailable"])
    def test_undecided_states_force_nothing(
        self, hass: HomeAssistant, entity_id: str, state: str
    ) -> None:
        """A broken sensor must never black out the house."""
        hass.states.async_set(entity_id, state)
        assert nobody_home(hass, entity_id) is False

    def test_fails_open(self, hass: HomeAssistant) -> None:
        """No entity, no entry in the state machine: force nothing."""
        assert nobody_home(hass, "") is False
        assert nobody_home(hass, None) is False
        assert nobody_home(hass, "binary_sensor.never_existed") is False


class TestHomeVeto:
    """The veto sits above the sensor calculation."""

    def test_forces_area_clear(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """An empty home outvotes whatever the sensors say."""
        hass.states.async_set(HOME_ENTITY, "off")
        with _home_entity(coordinator, HOME_ENTITY):
            assert default_area.forced_state() is False
            assert default_area.forced_probability() == MIN_PROBABILITY
            assert default_area.probability() == MIN_PROBABILITY
            assert default_area.occupied() is False

    def test_covers_every_area(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """The veto is global, not a per-area setting."""
        hass.states.async_set(HOME_ENTITY, "off")
        with _home_entity(coordinator, HOME_ENTITY):
            assert coordinator.areas
            assert all(area.occupied() is False for area in coordinator.areas.values())

    @pytest.mark.parametrize("threshold", [0.01, 0.5, 1.0])
    def test_holds_at_every_threshold(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
        threshold: float,
    ) -> None:
        """A 1% threshold must not turn MIN_PROBABILITY back into occupied."""
        hass.states.async_set(HOME_ENTITY, "off")
        default_area.config.threshold = threshold
        with _home_entity(coordinator, HOME_ENTITY):
            assert default_area.occupied() is False
            default_area.override = True
            assert default_area.occupied() is True

    def test_someone_home_leaves_calculation_alone(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """With someone home the sensors decide as before."""
        hass.states.async_set(HOME_ENTITY, "on")
        with _home_entity(coordinator, HOME_ENTITY):
            assert default_area.forced_state() is None
            assert default_area.forced_probability() is None

    def test_unconfigured(
        self, coordinator: AreaOccupancyCoordinator, default_area: Area
    ) -> None:
        """Without a configured entity nothing changes at all."""
        with _home_entity(coordinator, ""):
            assert default_area.forced_state() is None

    def test_activity_follows_the_veto(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """The activity sensor must not keep reporting 'working' in an empty house."""
        hass.states.async_set(HOME_ENTITY, "off")
        with _home_entity(coordinator, HOME_ENTITY):
            activity = default_area.detected_activity()
        assert activity.activity_id is ActivityId.UNOCCUPIED
        assert activity.confidence == 0.0

    def test_entity_is_tracked(self, coordinator: AreaOccupancyCoordinator) -> None:
        """A veto nothing listens to would only apply at the next decay tick."""
        with _home_entity(coordinator, HOME_ENTITY):
            assert HOME_ENTITY in coordinator.tracked_entity_ids()
        with _home_entity(coordinator, ""):
            assert HOME_ENTITY not in coordinator.tracked_entity_ids()


class TestSetOverrideSchema:
    """The service schema is the first line of defence."""

    def test_accepts_valid_calls(self) -> None:
        """area_id plus a known state, with or without a duration."""
        assert SET_OVERRIDE_SCHEMA({"area_id": "office", "state": "clear"})
        assert SET_OVERRIDE_SCHEMA(
            {"area_id": "office", "state": "occupied", "duration": 60}
        )

    @pytest.mark.parametrize(
        "data",
        [
            {"state": "clear"},
            {"area_id": "office"},
            {"area_id": "", "state": "clear"},
            {"area_id": "office", "state": "bogus"},
            {"area_id": "office", "state": "clear", "duration": 0},
            {"area_id": "office", "state": "clear", "duration": 90000},
        ],
    )
    def test_rejects_invalid_calls(self, data: dict[str, Any]) -> None:
        """Missing, unknown or out-of-range values never reach the handler."""
        with pytest.raises(vol.Invalid):
            SET_OVERRIDE_SCHEMA(data)


class TestSetOverride:
    """The manual escape hatch."""

    @pytest.fixture(autouse=True)
    def _wire_coordinator(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Make the coordinator reachable and refreshes cheap."""
        hass.data[DOMAIN] = coordinator
        coordinator.async_refresh = AsyncMock()

    async def test_clear_and_back_to_auto(
        self, hass: HomeAssistant, default_area: Area
    ) -> None:
        """Clearing forces free, auto hands the area back."""
        area_id = default_area.config.area_id

        result = await _set_override(hass, _call(area_id=area_id, state="clear"))
        assert default_area.override is False
        assert result["occupied"] is False
        assert result["probability"] == MIN_PROBABILITY

        await _set_override(hass, _call(area_id=area_id, state="auto"))
        assert default_area.override is None
        assert default_area.forced_state() is None

    async def test_forces_occupied_with_nobody_home(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """A deliberate call is never silently undone by the home check."""
        hass.states.async_set(HOME_ENTITY, "off")
        with _home_entity(coordinator, HOME_ENTITY):
            await _set_override(
                hass, _call(area_id=default_area.config.area_id, state="occupied")
            )
            assert default_area.probability() == MAX_PROBABILITY
            assert default_area.occupied() is True

    async def test_forces_clear_with_someone_home(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """The override works in the other direction too."""
        hass.states.async_set(HOME_ENTITY, "on")
        with _home_entity(coordinator, HOME_ENTITY):
            await _set_override(
                hass, _call(area_id=default_area.config.area_id, state="clear")
            )
            assert default_area.occupied() is False

    async def test_unknown_area_is_rejected(self, hass: HomeAssistant) -> None:
        """An unknown area_id names the known ones rather than failing quietly."""
        with pytest.raises(ServiceValidationError, match="no_such_area"):
            await _set_override(hass, _call(area_id="no_such_area", state="clear"))

    async def test_duration_reverts_to_auto(
        self, hass: HomeAssistant, default_area: Area
    ) -> None:
        """The auto-revert is what keeps the escape hatch from becoming a trap."""
        with patch(
            "custom_components.area_occupancy.service.async_call_later"
        ) as call_later:
            call_later.return_value = Mock()
            result = await _set_override(
                hass,
                _call(area_id=default_area.config.area_id, state="clear", duration=60),
            )

        assert default_area.override is False
        assert default_area.override_cancel is not None
        assert result["expires_in"] == 60
        assert call_later.call_args[0][1] == 60

        revert = call_later.call_args[0][2]
        await revert(None)

        assert default_area.override is None
        assert default_area.override_cancel is None

    async def test_revert_survives_a_vanished_coordinator(
        self, hass: HomeAssistant, default_area: Area
    ) -> None:
        """A timer outliving its integration must not raise in the event loop."""
        with patch(
            "custom_components.area_occupancy.service.async_call_later"
        ) as call_later:
            call_later.return_value = Mock()
            await _set_override(
                hass,
                _call(area_id=default_area.config.area_id, state="clear", duration=60),
            )
        revert = call_later.call_args[0][2]

        del hass.data[DOMAIN]
        await revert(None)

        assert default_area.override is None

    async def test_second_call_cancels_pending_revert(
        self, hass: HomeAssistant, default_area: Area
    ) -> None:
        """Otherwise the first timer would later undo the second decision."""
        cancel = Mock()
        default_area.override = True
        default_area.override_cancel = cancel

        with patch(
            "custom_components.area_occupancy.service.async_call_later"
        ) as call_later:
            call_later.return_value = Mock()
            await _set_override(
                hass,
                _call(area_id=default_area.config.area_id, state="clear", duration=120),
            )

        cancel.assert_called_once_with()
        assert default_area.override is False
        assert call_later.call_args[0][1] == 120
        assert default_area.override_cancel is not None

    async def test_auto_never_schedules(
        self, hass: HomeAssistant, default_area: Area
    ) -> None:
        """Auto schedules nothing, whatever duration is passed."""
        with patch(
            "custom_components.area_occupancy.service.async_call_later"
        ) as call_later:
            result = await _set_override(
                hass,
                _call(area_id=default_area.config.area_id, state="auto", duration=60),
            )
        call_later.assert_not_called()
        assert default_area.override is None
        assert result["expires_in"] is None

    async def test_shutdown_cancels_pending_revert(self, default_area: Area) -> None:
        """A timer that outlives its area would fire against its replacement."""
        cancel = Mock()
        default_area.override = False
        default_area.override_cancel = cancel

        default_area.cancel_override_timer()

        cancel.assert_called_once_with()
        assert default_area.override_cancel is None


class TestAwayDelay:
    """The grace period that absorbs zone-edge flapping."""

    def _delay(self, coordinator: AreaOccupancyCoordinator, seconds: int) -> Any:
        """Set the configured away delay for the duration of a test."""
        return patch.object(
            type(coordinator.integration_config),
            "home_away_delay",
            new_callable=PropertyMock,
            return_value=seconds,
        )

    def test_zero_applies_at_once(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Without a delay the veto is immediate."""
        hass.states.async_set(HOME_ENTITY, "off")
        with _home_entity(coordinator, HOME_ENTITY), self._delay(coordinator, 0):
            assert coordinator.home_veto_active is True

    def test_delay_holds_the_veto_back(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Leaving is not enough; it has to last."""
        hass.states.async_set(HOME_ENTITY, "off")
        with _home_entity(coordinator, HOME_ENTITY), self._delay(coordinator, 120):
            assert coordinator.home_veto_active is False
            assert coordinator._home_away_since is not None
            coordinator._home_away_since = dt_util.utcnow() - timedelta(seconds=121)
            assert coordinator.home_veto_active is True

    def test_returning_resets_the_clock(
        self, hass: HomeAssistant, coordinator: AreaOccupancyCoordinator
    ) -> None:
        """Flapping at the zone edge must not accumulate towards the delay."""
        hass.states.async_set(HOME_ENTITY, "off")
        with _home_entity(coordinator, HOME_ENTITY), self._delay(coordinator, 120):
            assert coordinator.home_veto_active is False
            coordinator._home_away_since = dt_util.utcnow() - timedelta(seconds=119)
            hass.states.async_set(HOME_ENTITY, "on")
            assert coordinator.home_veto_active is False
            assert coordinator._home_away_since is None
            hass.states.async_set(HOME_ENTITY, "off")
            assert coordinator.home_veto_active is False


class TestStickyClear:
    """After a veto, stale evidence must not put the area straight back.

    Built on real entities with real decay on purpose: a fading entity still
    counts as active, and a mocked set of IDs hides exactly that.
    """

    def _sensors(
        self, hass: HomeAssistant, area: Area, **states: str
    ) -> dict[str, Entity]:
        """Give the area motion sensors in the given states, and nothing else."""
        sensors = {}
        for name, state in states.items():
            entity_id = f"binary_sensor.{name}"
            hass.states.async_set(entity_id, state)
            sensors[name] = Entity(
                entity_id=entity_id,
                type=EntityType(
                    input_type=InputType.MOTION,
                    weight=1.0,
                    prob_given_true=0.95,
                    prob_given_false=0.005,
                    active_states=[STATE_ON],
                ),
                prob_given_true=0.95,
                prob_given_false=0.005,
                decay=Decay(half_life=600.0),
                hass=hass,
                last_updated=dt_util.utcnow(),
                previous_evidence=state == STATE_ON,
            )
        area.entities._entities = {s.entity_id: s for s in sensors.values()}
        return sensors

    def _switch(self, hass: HomeAssistant, sensor: Entity, state: str) -> None:
        """Change a sensor the way the coordinator sees it, decay included."""
        hass.states.async_set(sensor.entity_id, state)
        sensor.has_new_evidence()

    def _leave_and_return(self, hass: HomeAssistant, area: Area) -> None:
        """Run one away/home cycle, snapshot included.

        The coordinator refreshes on the home entity's own state change, so
        the snapshot is taken at the return, before any sensor moves.
        """
        hass.states.async_set(HOME_ENTITY, "off")
        assert area.forced_state() is False
        hass.states.async_set(HOME_ENTITY, "on")
        area.forced_state()

    def test_leftover_sensor_does_not_restore_occupancy(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """The PC-in-use sensor that never went off proves nothing."""
        self._sensors(hass, default_area, pc_in_use=STATE_ON)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            assert default_area.forced_state() is False
            assert default_area.occupied() is False
            assert default_area.probability() == MIN_PROBABILITY

    def test_nothing_on_arms_no_latch(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """With every sensor off and settled there is nothing to suppress."""
        self._sensors(hass, default_area, pc_in_use=STATE_OFF, motion=STATE_OFF)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            assert default_area.forced_state() is None

    def test_a_new_sensor_releases_the_latch(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """Anything switching on that was not left over counts as arrival."""
        sensors = self._sensors(
            hass, default_area, pc_in_use=STATE_ON, motion=STATE_OFF
        )
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            assert default_area.forced_state() is False
            self._switch(hass, sensors["motion"], STATE_ON)
            assert default_area.forced_state() is None

    def test_leftover_back_on_while_fading_releases_the_latch(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """Dropping out and coming back is new, even inside its decay."""
        sensors = self._sensors(hass, default_area, pc_in_use=STATE_ON)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            self._switch(hass, sensors["pc_in_use"], STATE_OFF)
            assert sensors["pc_in_use"].decay.is_decaying
            assert default_area.forced_state() is False
            self._switch(hass, sensors["pc_in_use"], STATE_ON)
            assert default_area.forced_state() is None

    def test_sensor_fading_at_return_counts_when_it_switches_on(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """A sensor still fading from the departure fires again on arrival."""
        sensors = self._sensors(hass, default_area, motion=STATE_ON)
        self._switch(hass, sensors["motion"], STATE_OFF)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            assert default_area.forced_state() is False
            self._switch(hass, sensors["motion"], STATE_ON)
            assert default_area.forced_state() is None

    def test_latch_waits_for_a_leftover_to_fade(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """Ending while it still fades would flip the area to occupied."""
        sensors = self._sensors(hass, default_area, pc_in_use=STATE_ON)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            self._switch(hass, sensors["pc_in_use"], STATE_OFF)
            assert default_area.forced_state() is False
            sensors["pc_in_use"].decay.stop_decay()  # the decay has run out
            assert default_area.forced_state() is None

    def test_an_outage_is_not_an_off(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """A leftover back from unavailable is still the same leftover."""
        sensors = self._sensors(hass, default_area, door=STATE_ON)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            self._switch(hass, sensors["door"], STATE_UNAVAILABLE)
            assert default_area.forced_state() is False
            self._switch(hass, sensors["door"], STATE_ON)
            assert default_area.forced_state() is False

    def test_an_outage_outlasting_the_decay_keeps_the_leftover(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """The latch lets go meanwhile, but still knows the leftover."""
        sensors = self._sensors(hass, default_area, pc_in_use=STATE_ON)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            self._switch(hass, sensors["pc_in_use"], STATE_UNAVAILABLE)
            sensors["pc_in_use"].decay.decay_start = dt_util.utcnow() - timedelta(
                days=1
            )
            assert default_area.forced_state() is None
            self._switch(hass, sensors["pc_in_use"], STATE_ON)
            assert default_area.forced_state() is False

    def test_unavailable_at_return_is_a_leftover(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """An entity unreadable at the return reports a state from before."""
        sensors = self._sensors(
            hass, default_area, pc_in_use=STATE_ON, door=STATE_UNAVAILABLE
        )
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            assert default_area.forced_state() is False
            self._switch(hass, sensors["door"], STATE_ON)
            assert default_area.forced_state() is False

    def test_a_spent_decay_arms_no_latch(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """Without the decay tick is_decaying stays set; the factor decides."""
        sensors = self._sensors(hass, default_area, motion=STATE_ON)
        self._switch(hass, sensors["motion"], STATE_OFF)
        sensors["motion"].decay.decay_start = dt_util.utcnow() - timedelta(days=1)
        assert sensors["motion"].decay.is_decaying
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            assert default_area.forced_state() is None

    def test_one_leftover_still_holds_while_another_drops_out(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """Losing one leftover is not arrival; the other is still stale."""
        sensors = self._sensors(hass, default_area, pc_in_use=STATE_ON, mmwave=STATE_ON)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            self._switch(hass, sensors["mmwave"], STATE_OFF)
            assert default_area.forced_state() is False
            self._switch(hass, sensors["mmwave"], STATE_ON)
            assert default_area.forced_state() is None

    def test_override_clears_the_latch(
        self,
        hass: HomeAssistant,
        coordinator: AreaOccupancyCoordinator,
        default_area: Area,
    ) -> None:
        """A manual decision wins over a latch it knows nothing about."""
        self._sensors(hass, default_area, pc_in_use=STATE_ON)
        with _home_entity(coordinator, HOME_ENTITY):
            self._leave_and_return(hass, default_area)
            assert default_area.forced_state() is False
            default_area.override = True
            assert default_area.forced_state() is True
            default_area.override = None
            assert default_area.forced_state() is None
