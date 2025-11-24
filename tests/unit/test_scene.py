"""Unit tests for scene understanding module."""

import pytest
import numpy as np
import time
from unittest.mock import Mock

from omnisense.intelligence.scene import (
    SceneAnalyzer,
    Zone,
    ZoneType,
    OccupancyLevel,
    ZoneEvent,
    GroupFormation
)


class TestZone:
    """Test Zone class."""

    def test_zone_creation(self):
        """Test zone creation."""
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )

        assert zone.zone_id == "zone1"
        assert zone.name == "Test Zone"
        assert zone.zone_type == ZoneType.MONITORED
        assert zone.enabled

    def test_zone_contains(self):
        """Test zone containment check."""
        zone = Zone(
            zone_id="zone1",
            name="Test",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )

        # Point inside
        assert zone.contains(np.array([2.5, 2.5, 1.5]))

        # Point outside
        assert not zone.contains(np.array([10, 10, 10]))

        # Point on boundary
        assert zone.contains(np.array([5, 5, 3]))

    def test_zone_center(self):
        """Test zone center computation."""
        zone = Zone(
            zone_id="zone1",
            name="Test",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 10, 10, 2])
        )

        center = zone.get_center()
        expected = np.array([5, 5, 1])

        np.testing.assert_allclose(center, expected)

    def test_zone_volume(self):
        """Test zone volume computation."""
        zone = Zone(
            zone_id="zone1",
            name="Test",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 10, 5, 2])
        )

        volume = zone.get_volume()
        assert volume == 100.0  # 10 * 5 * 2


class TestSceneAnalyzer:
    """Test SceneAnalyzer class."""

    @pytest.fixture
    def analyzer(self):
        """Create scene analyzer instance."""
        return SceneAnalyzer()

    @pytest.fixture
    def mock_person(self):
        """Create mock person object."""
        def create_person(person_id, position, velocity=None):
            person = Mock()
            person.global_id = person_id
            person.position_3d = np.array(position)
            person.velocity_3d = velocity if velocity is not None else np.array([0, 0, 0])
            person.confidence = 0.9
            return person
        return create_person

    def test_initialization(self, analyzer):
        """Test analyzer initialization."""
        assert analyzer is not None
        assert len(analyzer.zones) == 0
        assert len(analyzer.zone_occupancy) == 0

    def test_add_zone(self, analyzer):
        """Test adding a zone."""
        zone = Zone(
            zone_id="zone1",
            name="Entry Zone",
            zone_type=ZoneType.ENTRY,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )

        analyzer.add_zone(zone)

        assert "zone1" in analyzer.zones
        assert "zone1" in analyzer.zone_occupancy
        assert analyzer.zone_occupancy["zone1"].occupancy_count == 0

    def test_remove_zone(self, analyzer):
        """Test removing a zone."""
        zone = Zone(
            zone_id="zone1",
            name="Test",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )

        analyzer.add_zone(zone)
        assert "zone1" in analyzer.zones

        analyzer.remove_zone("zone1")
        assert "zone1" not in analyzer.zones

    def test_zone_entry_detection(self, analyzer, mock_person):
        """Test zone entry event detection."""
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )
        analyzer.add_zone(zone)

        # Person outside zone
        persons = {1: mock_person(1, [10, 10, 1])}
        analyzer.update(persons)

        # Person enters zone
        persons = {1: mock_person(1, [2.5, 2.5, 1.5])}
        analyzer.update(persons)

        # Check entry event
        events = analyzer.get_zone_events(event_type="entry")
        assert len(events) > 0
        assert events[0].person_id == 1
        assert events[0].zone_id == "zone1"

    def test_zone_exit_detection(self, analyzer, mock_person):
        """Test zone exit event detection."""
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )
        analyzer.add_zone(zone)

        # Person inside zone
        persons = {1: mock_person(1, [2.5, 2.5, 1.5])}
        analyzer.update(persons)

        # Person exits zone
        persons = {1: mock_person(1, [10, 10, 1])}
        analyzer.update(persons)

        # Check exit event
        events = analyzer.get_zone_events(event_type="exit")
        assert len(events) > 0
        assert events[0].person_id == 1
        assert events[0].zone_id == "zone1"

    def test_restricted_zone_violation(self, analyzer, mock_person):
        """Test restricted zone violation detection."""
        zone = Zone(
            zone_id="restricted1",
            name="Restricted Area",
            zone_type=ZoneType.RESTRICTED,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )
        analyzer.add_zone(zone)

        # Person enters restricted zone
        persons = {1: mock_person(1, [2.5, 2.5, 1.5])}
        analyzer.update(persons)

        # Check violation event
        events = analyzer.get_zone_events(event_type="violation")
        assert len(events) > 0
        assert events[0].zone_id == "restricted1"

    def test_occupancy_counting(self, analyzer, mock_person):
        """Test zone occupancy counting."""
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 10, 10, 3])
        )
        analyzer.add_zone(zone)

        # Add multiple persons
        persons = {
            1: mock_person(1, [2, 2, 1]),
            2: mock_person(2, [5, 5, 1]),
            3: mock_person(3, [8, 8, 1])
        }
        analyzer.update(persons)

        occupancy = analyzer.get_zone_occupancy("zone1")
        assert occupancy.occupancy_count == 3
        assert 1 in occupancy.person_ids
        assert 2 in occupancy.person_ids
        assert 3 in occupancy.person_ids

    def test_density_computation(self, analyzer, mock_person):
        """Test person density computation."""
        # 10x10 zone = 100 m²
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 10, 10, 3])
        )
        analyzer.add_zone(zone)

        # Add 5 persons -> density = 0.05 persons/m²
        persons = {i: mock_person(i, [i, i, 1]) for i in range(5)}
        analyzer.update(persons)

        occupancy = analyzer.get_zone_occupancy("zone1")
        expected_density = 5 / 100.0
        assert abs(occupancy.density - expected_density) < 1e-6

    def test_occupancy_level_classification(self, analyzer, mock_person):
        """Test occupancy level classification."""
        # Small zone for easier testing
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 2, 2, 3]),  # 4 m²
            max_capacity=4
        )
        analyzer.add_zone(zone)

        # Empty
        analyzer.update({})
        occupancy = analyzer.get_zone_occupancy("zone1")
        assert occupancy.occupancy_level == OccupancyLevel.EMPTY

        # Low (1 person, 25% capacity)
        persons = {1: mock_person(1, [1, 1, 1])}
        analyzer.update(persons)
        occupancy = analyzer.get_zone_occupancy("zone1")
        assert occupancy.occupancy_level == OccupancyLevel.LOW

        # Medium (2 persons, 50% capacity)
        persons = {1: mock_person(1, [0.5, 0.5, 1]), 2: mock_person(2, [1.5, 1.5, 1])}
        analyzer.update(persons)
        occupancy = analyzer.get_zone_occupancy("zone1")
        assert occupancy.occupancy_level == OccupancyLevel.MEDIUM

        # High (3 persons, 75% capacity)
        persons.update({3: mock_person(3, [0.5, 1.5, 1])})
        analyzer.update(persons)
        occupancy = analyzer.get_zone_occupancy("zone1")
        assert occupancy.occupancy_level == OccupancyLevel.HIGH

        # Overcrowded (5 persons, 125% capacity)
        persons.update({4: mock_person(4, [1.5, 0.5, 1]), 5: mock_person(5, [1, 0.5, 1])})
        analyzer.update(persons)
        occupancy = analyzer.get_zone_occupancy("zone1")
        assert occupancy.occupancy_level == OccupancyLevel.OVERCROWDED

    def test_flow_vector_computation(self, analyzer, mock_person):
        """Test flow vector computation."""
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 10, 10, 3])
        )
        analyzer.add_zone(zone)

        # Add persons moving in same direction
        velocity = np.array([1, 0, 0])
        for i in range(10):
            persons = {
                1: mock_person(1, [2, 2, 1], velocity),
                2: mock_person(2, [5, 5, 1], velocity)
            }
            analyzer.update(persons)
            time.sleep(0.01)

        flow = analyzer.compute_flow_vector("zone1")

        assert flow is not None
        assert flow.zone_id == "zone1"
        assert flow.speed > 0
        # Direction should be primarily in X direction
        assert abs(flow.direction[0]) > abs(flow.direction[1])

    def test_crowding_detection(self, analyzer, mock_person):
        """Test crowding detection."""
        # Small zone
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 2, 2, 3])  # 4 m²
        )
        analyzer.add_zone(zone)

        # Add many persons to create high density
        persons = {}
        for i in range(10):
            x = (i % 2) + 0.5
            y = (i // 2) * 0.4 + 0.5
            persons[i] = mock_person(i, [x, y, 1])

        analyzer.update(persons)

        # Check crowding detection
        crowded = analyzer.detect_crowding()
        assert "zone1" in crowded

    def test_loitering_detection(self, analyzer, mock_person):
        """Test loitering detection."""
        zone = Zone(
            zone_id="zone1",
            name="Test Zone",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 5, 5, 3])
        )
        analyzer.add_zone(zone)

        # Person stays in zone for long time
        persons = {1: mock_person(1, [2.5, 2.5, 1.5])}

        # First update
        analyzer.update(persons, timestamp=0.0)

        # Second update after 400 seconds
        analyzer.update(persons, timestamp=400.0)

        # Check loitering (threshold is 300 seconds)
        loiterers = analyzer.detect_loitering(threshold_seconds=300)

        assert len(loiterers) > 0
        assert loiterers[0][0] == 1  # person_id
        assert loiterers[0][1] == "zone1"  # zone_id
        assert loiterers[0][2] >= 300  # dwell_time

    def test_multiple_zones_priority(self, analyzer, mock_person):
        """Test person in multiple overlapping zones."""
        zone1 = Zone(
            zone_id="zone1",
            name="Zone 1",
            zone_type=ZoneType.MONITORED,
            bounds_3d=np.array([0, 0, 0, 10, 10, 3]),
            priority=1
        )

        zone2 = Zone(
            zone_id="zone2",
            name="Zone 2",
            zone_type=ZoneType.RESTRICTED,
            bounds_3d=np.array([2, 2, 0, 8, 8, 3]),
            priority=2
        )

        analyzer.add_zone(zone1)
        analyzer.add_zone(zone2)

        # Person in overlapping area
        persons = {1: mock_person(1, [5, 5, 1.5])}
        analyzer.update(persons)

        # Person should be in both zones
        zones = analyzer.get_person_current_zones(1)
        assert "zone1" in zones
        assert "zone2" in zones


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
