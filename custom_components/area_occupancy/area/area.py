"""Area class for individual device areas.

The Area class encapsulates all area-specific behavior and components,
including configuration, entities, prior probability, and purpose management.
This represents a single device area in the multi-area architecture.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from homeassistant.helpers.device_registry import DeviceInfo

from ..const import (
    DEVICE_MANUFACTURER,
    DEVICE_MODEL,
    DEVICE_SW_VERSION,
    DOMAIN,
    MAX_PROBABILITY,
    MIN_PROBABILITY,
)
from ..data.activity import ActivityId, DetectedActivity, detect_activity
from ..data.adjacency import apply_logit_boost
from ..data.analysis import start_prior_analysis
from ..data.health import HealthMonitor
from ..utils import (
    apply_activity_boost,
    combined_probability as calc_combined,
    environmental_confidence as calc_env,
    presence_probability as calc_presence,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from ..coordinator import AreaOccupancyCoordinator
    from ..data.config import AreaConfig
    from ..data.entity import EntityFactory, EntityManager
    from ..data.prior import Prior
    from ..data.purpose import Purpose
else:
    from ..data.config import AreaConfig
    from ..data.entity import EntityFactory, EntityManager
    from ..data.prior import Prior
    from ..data.purpose import Purpose

_LOGGER = logging.getLogger(__name__)


class AreaDeviceHandle:
    """Lightweight reference to an Area instance that survives reloads."""

    def __init__(self, coordinator: AreaOccupancyCoordinator, area_name: str) -> None:
        """Initialize a new handle for the given area.

        Args:
            coordinator: Coordinator that manages the area.
            area_name: Name of the area represented by this handle.
        """
        self.coordinator = coordinator
        self.area_name = area_name
        self._area: Area | None = None

    def attach(self, area: Area | None) -> None:
        """Attach the latest Area implementation."""
        self._area = area

    @property
    def area(self) -> Area | None:
        """Return the currently attached Area."""
        return self._area

    def resolve(self) -> Area | None:
        """Return the current Area, refreshing from the coordinator."""
        area = self.coordinator.get_area(self.area_name)
        self._area = area
        return area

    def device_info(self) -> DeviceInfo | None:
        """Return DeviceInfo for the attached Area, if available."""
        area = self.resolve()
        if area is None:
            return None
        return area.device_info()


class Area:
    """Represents an individual device area in the multi-area architecture.

    The Area class encapsulates all area-specific components and behavior:
    - Configuration (sensors, weights, thresholds)
    - Entity management (tracking sensor states)
    - Prior probability calculation
    - Purpose management (area purpose and decay settings)

    This class is self-contained and handles all area-specific operations.
    """

    def __init__(
        self,
        coordinator: AreaOccupancyCoordinator,
        area_name: str,
        area_data: dict | None = None,
    ) -> None:
        """Initialize the Area instance.

        Args:
            coordinator: The coordinator instance managing this area
            area_name: Name/identifier for this area
            area_data: Optional area-specific configuration data

        Note:
            The area must be added to coordinator.areas BEFORE components
            are initialized. Components will be initialized lazily on first access
            to ensure the area exists in coordinator.areas first.
        """
        self.coordinator = coordinator
        self.area_name = area_name
        self.config = AreaConfig(coordinator, area_name=area_name, area_data=area_data)

        # Components will be initialized lazily after area is added to coordinator.areas
        # This avoids circular dependency issues during initialization
        self._factory: EntityFactory | None = None
        self._prior: Prior | None = None
        self._purpose: Purpose | None = None
        self._entities: EntityManager | None = None
        self._health_monitor: HealthMonitor | None = None

        # Entity IDs for platform entities (set by platform modules)
        self.occupancy_entity_id: str | None = None
        self.wasp_entity_id: str | None = None
        self.sleep_entity_id: str | None = None

        # Manual override from the set_override service; None means the
        # sensors decide. Deliberately not restored across restarts, so a
        # forgotten override cannot outlive a reboot.
        self.override: bool | None = None
        self.override_cancel: Callable[[], None] | None = None

        # Sticky clear after a veto: the entity IDs that have not been plainly
        # off since the veto lifted. They are stale by definition — they were
        # on, or unreadable, while the house was empty — so they must not put
        # the area straight back to occupied. None means no latch.
        self._stale_evidence: set[str] | None = None
        self._veto_was_active = False

        # Activity detection cache
        self._activity_cache: DetectedActivity | None = None
        self._activity_cache_key: tuple[frozenset[str], float] | None = None

    @property
    def factory(self) -> EntityFactory:
        """Get or create the EntityFactory for this area."""
        if self._factory is None:
            self._factory = EntityFactory(self.coordinator, area_name=self.area_name)
        return self._factory

    @property
    def prior(self) -> Prior:
        """Get or create the Prior instance for this area."""
        if self._prior is None:
            self._prior = Prior(
                self.coordinator, area_name=self.area_name, config=self.config
            )
        return self._prior

    @property
    def purpose(self) -> Purpose:
        """Get or create the Purpose for this area."""
        if self._purpose is None:
            purpose_value = getattr(self.config, "purpose", None)
            self._purpose = Purpose(purpose=purpose_value)
        return self._purpose

    @property
    def entities(self) -> EntityManager:
        """Get or create the EntityManager for this area."""
        if self._entities is None:
            self._entities = EntityManager(self.coordinator, area_name=self.area_name)
        return self._entities

    @property
    def health_monitor(self) -> HealthMonitor:
        """Get or create the HealthMonitor for this area."""
        if self._health_monitor is None:
            area_id = self.config.area_id or self.area_name
            self._health_monitor = HealthMonitor(
                self.area_name,
                area_id,
                self.coordinator.hass,
                purpose=self.purpose.purpose,
            )
        return self._health_monitor

    async def run_prior_analysis(self) -> None:
        """Run prior analysis for this area."""
        await start_prior_analysis(self.coordinator, self.area_name, self.prior)

    async def async_cleanup(self) -> None:
        """Clean up the area's resources.

        This should be called when the area is being removed or the
        integration is shutting down.
        """
        # Clear prior cache first to release cached data
        if self._prior is not None:
            self._prior.clear_cache()
        # Clean up health monitor repair issues to prevent orphans
        if self._health_monitor is not None:
            self._health_monitor.cleanup()
        await self.entities.cleanup()
        self.purpose.cleanup()

    def device_info(self) -> DeviceInfo:
        """Return device info for this area.

        Returns:
            DeviceInfo for this area
        """
        # Use area_id for device identifier (stable even if area is renamed)
        # Fallback to area_name if area_id is not available
        device_identifier = self.config.area_id or self.area_name
        return DeviceInfo(
            identifiers={(DOMAIN, device_identifier)},
            name=self.config.name,
            manufacturer=DEVICE_MANUFACTURER,
            model=DEVICE_MODEL,
            sw_version=DEVICE_SW_VERSION,
        )

    def _base_probability(self) -> float:
        """Calculate sensor-only occupancy probability (no activity boost).

        Combines presence probability (from strong binary indicators) with
        environmental confidence (from environmental sensors) as an additive
        update in logit space.

        This is the first phase of the two-phase probability calculation.
        Activity detection receives this value to avoid circular dependency.

        Returns:
            Probability value (0.0-1.0)
        """
        entities = self.entities.entities
        if not entities:
            return MIN_PROBABILITY

        presence = self.presence_probability()
        env = self.environmental_confidence()

        # Short-circuit on neutral environmental confidence. 0.5 is what
        # environmental_confidence() returns whenever the environmental channel
        # contributes nothing — no environmental sensors configured, or none of
        # them currently active. calc_combined() is continuous at that point
        # (logit(0.5) == 0, so it returns presence unchanged) — this just skips
        # a redundant logit/sigmoid round-trip.
        if env == 0.5:
            return presence

        return calc_combined(presence, env)

    def probability(self) -> float:
        """Calculate occupancy probability with activity- and adjacency-based boosts.

        Three-phase calculation:
        1. ``_base_probability()`` computes sensor-only probability.
        2. Activity detection runs against the base probability;
           a strong activity boosts in logit space.
        3. The adjacency boost (Phase 4) bends the result toward what
           the household's learned transitions suggest comes next given
           the recent trajectory.

        Returns:
            Probability value (0.0-1.0)
        """
        forced = self.forced_probability()
        if forced is not None:
            return forced

        base = self._base_probability()
        is_occupied = base >= self.config.threshold

        activity = detect_activity(self, base_probability=base, is_occupied=is_occupied)

        if activity.activity_id in (ActivityId.UNOCCUPIED, ActivityId.IDLE):
            result = base
        else:
            result = apply_activity_boost(
                base, activity.occupancy_boost, activity.confidence
            )

        boost = self.coordinator.adjacency_boost_for(self.area_name)
        if boost is not None:
            result = apply_logit_boost(result, boost)
        return result

    def forced_state(self) -> bool | None:
        """Return an overriding occupancy decision, or None for the sensors.

        Two overrides exist, service first: a ``set_override`` call wins, so
        a deliberate manual decision is never silently undone by the home
        check. Otherwise, while the configured home entity reports that
        nobody is home at all, the area is forced clear.

        Sits above the whole calculation on purpose. Nothing here reaches the
        learning pipeline, which reads sensor history rather than this
        integration's own output.

        Returns:
            True or False to force the decision, None to let the sensors decide
        """
        if self.override is not None:
            self._stale_evidence = None
            self._veto_was_active = False
            return self.override

        if self.coordinator.home_veto_active:
            self._veto_was_active = True
            self._stale_evidence = None
            return False

        if self._veto_was_active:
            # The veto just lifted. Whatever is on right now was already on
            # while the house was empty, so it proves nothing, and whatever is
            # still fading is left over from before as well. So is an entity
            # that is unavailable right now: what it reports on its return was
            # picked up while nobody was home.
            self._veto_was_active = False
            evidence, fading = self._latch_evidence()
            leftovers = {
                entity_id for entity_id, value in evidence.items() if value is not False
            }
            self._stale_evidence = leftovers if leftovers or fading else None

        if self._stale_evidence is not None:
            evidence, fading = self._latch_evidence()
            on = {entity_id for entity_id, value in evidence.items() if value is True}
            if on - self._stale_evidence:
                # Something is on that has not been left over since the veto
                # lifted, including a leftover that went off and came back
                # while still fading: real evidence.
                self._stale_evidence = None
                return None
            # Only a plain off clears a leftover, so switching back on counts as
            # new. An outage is not an off: the same state coming back after
            # one proves nothing. Hold while a leftover is on or anything still
            # fades: that contribution is stale too, and ending the latch before
            # it has faded would flip the area to occupied.
            self._stale_evidence -= {
                entity_id for entity_id, value in evidence.items() if value is False
            }
            if self._stale_evidence & on or fading:
                return False
            # Nothing left to suppress right now. Leftovers that are merely
            # unavailable stay remembered without forcing anything, so one
            # returning after an outage longer than its decay is still stale.
            if not self._stale_evidence:
                self._stale_evidence = None

        return None

    def _latch_evidence(self) -> tuple[dict[str, bool | None], bool]:
        """Return each entity's evidence, and whether any decay still contributes.

        Deliberately not ``active_entities``: that counts a fading entity as
        active, so a leftover that went off and came back within its decay
        would never register as new. Fading goes by the factor rather than
        ``is_decaying`` alone, because the decay tick only resets that flag in
        areas with decay enabled.

        Returns:
            Evidence by entity ID (None while unavailable), and True while a
            decay still adds to the probability; nothing when the manager is
            unavailable
        """
        try:
            entities = list(self.entities.entities.values())
        except (AttributeError, KeyError, TypeError):
            return {}, False
        evidence = {e.entity_id: e.evidence for e in entities}
        fading = any(e.decay.is_decaying and e.decay.decay_factor > 0 for e in entities)
        return evidence, fading

    def forced_probability(self) -> float | None:
        """Return the probability a forced decision reports, if any.

        Returns:
            A forced probability, or None to run the normal calculation
        """
        forced = self.forced_state()
        if forced is None:
            return None
        return MAX_PROBABILITY if forced else MIN_PROBABILITY

    def cancel_override_timer(self) -> None:
        """Drop a pending auto-revert for this area.

        Lives here rather than in the service so coordinator shutdown can
        reach it: a timer that outlives its area would later fire against a
        replaced one.
        """
        if self.override_cancel is not None:
            self.override_cancel()
            self.override_cancel = None

    def presence_probability(self) -> float:
        """Calculate presence probability from strong binary indicators.

        Uses motion, media, appliances, doors, windows, covers, power, and
        Wi-Fi client-count sensors to determine presence likelihood.

        Returns:
            Probability value (0.0-1.0)
        """
        entities = self.entities.entities
        if not entities:
            return MIN_PROBABILITY

        correlations = self._get_entity_correlations()

        return calc_presence(
            entities, prior=self.prior.value, correlations=correlations
        )

    def environmental_confidence(self) -> float:
        """Calculate environmental support confidence.

        Uses temperature, humidity, illuminance, CO2, and other environmental
        sensors to determine how much the environment supports occupancy.

        Returns:
            Confidence value (0.0-1.0), where 0.5 is neutral
        """
        entities = self.entities.entities
        if not entities:
            return 0.5  # Neutral when no entities

        correlations = self._get_entity_correlations()

        return calc_env(entities, correlations=correlations)

    def _get_entity_correlations(self) -> dict[str, float]:
        """Get cached correlation strengths for this area.

        Returns correlations loaded asynchronously by the coordinator.
        No DB calls are made in this method.

        Returns:
            Dict of entity_id -> correlation strength. Empty dict if no data.
        """
        return self.coordinator.get_cached_correlations(self.area_name)

    def area_prior(self) -> float:
        """Get the area's baseline occupancy prior from historical data.

        This returns the pure P(area occupied) without any sensor weighting.

        Returns:
            Prior probability (0.0-1.0)
        """
        return self.prior.value

    def decay(self) -> float:
        """Calculate the current decay probability (0.0-1.0) for this area.

        Returns:
            Decay probability (0.0-1.0)
        """
        entities = self.entities.entities
        if not entities:
            return 1.0

        decay_sum = sum(entity.decay.decay_factor for entity in entities.values())
        return decay_sum / len(entities)

    def tick_decay(self) -> None:
        """Tick all entity decays to update their state.

        This method should be called periodically (e.g., by the decay timer)
        to transition decay states when factors drop below threshold.
        """
        for entity in self.entities.entities.values():
            entity.decay.tick()

    def occupied(self) -> bool:
        """Return the current occupancy state (True/False) for this area.

        Returns:
            True if occupied, False otherwise
        """
        forced = self.forced_state()
        if forced is not None:
            # Not via probability(): thresholds reach 1 and 100, where
            # MIN_PROBABILITY >= threshold and MAX_PROBABILITY < threshold
            # would both invert the forced answer.
            return forced
        return self.probability() >= self.config.threshold

    def detected_activity(self) -> DetectedActivity:
        """Detect the current activity in this area.

        Results are cached and recomputed only when the set of active
        entity IDs or the base probability changes.

        Returns:
            DetectedActivity with activity_id, confidence, and matching indicators.
        """
        if self.forced_state() is False:
            # detect_activity() runs off the raw sensor probability, so
            # without this the activity sensor would still report "working"
            # in a house the veto just emptied.
            return DetectedActivity(activity_id=ActivityId.UNOCCUPIED, confidence=0.0)

        active_ids = frozenset(e.entity_id for e in self.entities.active_entities)
        base = self._base_probability()
        prob = round(base, 4)
        cache_key = (active_ids, prob)

        if self._activity_cache_key == cache_key and self._activity_cache is not None:
            return self._activity_cache

        result = detect_activity(
            self, base_probability=base, is_occupied=base >= self.config.threshold
        )
        self._activity_cache = result
        self._activity_cache_key = cache_key
        return result

    def threshold(self) -> float:
        """Return the current occupancy threshold (0.0-1.0) for this area.

        Returns:
            Threshold value (0.0-1.0)
        """
        return self.config.threshold
