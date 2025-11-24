"""Unit tests for activity recognition module."""

import pytest
import numpy as np
import time

from omnisense.intelligence.activity import (
    ActivityRecognizer,
    ActivityType,
    ActivityDetection
)


class TestActivityRecognizer:
    """Test activity recognition functionality."""

    @pytest.fixture
    def recognizer(self):
        """Create activity recognizer instance."""
        return ActivityRecognizer(use_ml_model=False)

    def test_initialization(self, recognizer):
        """Test recognizer initialization."""
        assert recognizer is not None
        assert not recognizer.use_ml_model
        assert len(recognizer.pose_sequences) == 0
        assert len(recognizer.current_activities) == 0

    def test_standing_detection(self, recognizer):
        """Test standing posture detection."""
        # Create keypoints for standing person
        keypoints = np.zeros((33, 3))

        # Set leg keypoints for standing (straight legs)
        # Hip, knee, ankle positions
        keypoints[23] = [0, 0, 1.0]  # Left hip
        keypoints[24] = [0.2, 0, 1.0]  # Right hip
        keypoints[25] = [0, 0, 0.5]  # Left knee
        keypoints[26] = [0.2, 0, 0.5]  # Right knee
        keypoints[27] = [0, 0, 0]  # Left ankle
        keypoints[28] = [0.2, 0, 0]  # Right ankle

        velocity = np.array([0, 0, 0])  # Stationary

        # Update multiple times to build history
        for _ in range(10):
            activity = recognizer.update(1, keypoints, velocity)
            time.sleep(0.01)

        assert activity.activity_type == ActivityType.STANDING
        assert activity.confidence > 0.5

    def test_sitting_detection(self, recognizer):
        """Test sitting posture detection."""
        # Create keypoints for sitting person
        keypoints = np.zeros((33, 3))

        # Set leg keypoints for sitting (bent knees)
        keypoints[23] = [0, 0, 0.8]  # Left hip
        keypoints[24] = [0.2, 0, 0.8]  # Right hip
        keypoints[25] = [0, 0.3, 0.5]  # Left knee (forward)
        keypoints[26] = [0.2, 0.3, 0.5]  # Right knee
        keypoints[27] = [0, 0.5, 0.1]  # Left ankle (below knee)
        keypoints[28] = [0.2, 0.5, 0.1]  # Right ankle

        velocity = np.array([0, 0, 0])

        for _ in range(10):
            activity = recognizer.update(1, keypoints, velocity)
            time.sleep(0.01)

        assert activity.activity_type == ActivityType.SITTING
        assert activity.confidence > 0.5

    def test_walking_detection(self, recognizer):
        """Test walking detection."""
        # Create keypoints for walking person
        keypoints = np.zeros((33, 3))

        # Standing pose
        keypoints[23] = [0, 0, 1.0]
        keypoints[24] = [0.2, 0, 1.0]
        keypoints[25] = [0, 0, 0.5]
        keypoints[26] = [0.2, 0, 0.5]
        keypoints[27] = [0, 0, 0]
        keypoints[28] = [0.2, 0, 0]

        # Walking velocity
        velocity = np.array([0.5, 0, 0])  # Walking speed

        # Simulate walking gait with alternating leg heights
        for i in range(15):
            # Alternate ankle heights to simulate steps
            keypoints[27, 2] = 0.1 * np.sin(i * 0.5)
            keypoints[28, 2] = 0.1 * np.cos(i * 0.5)

            activity = recognizer.update(1, keypoints, velocity)
            time.sleep(0.01)

        assert activity.activity_type == ActivityType.WALKING
        assert activity.confidence > 0.5

    def test_running_detection(self, recognizer):
        """Test running detection."""
        keypoints = np.zeros((33, 3))

        # Standing pose
        keypoints[23] = [0, 0, 1.0]
        keypoints[24] = [0.2, 0, 1.0]
        keypoints[25] = [0, 0, 0.5]
        keypoints[26] = [0.2, 0, 0.5]
        keypoints[27] = [0, 0, 0]
        keypoints[28] = [0.2, 0, 0]

        # Running velocity (fast)
        velocity = np.array([2.5, 0, 0])

        for _ in range(10):
            activity = recognizer.update(1, keypoints, velocity)
            time.sleep(0.01)

        assert activity.activity_type == ActivityType.RUNNING
        assert activity.confidence > 0.5

    def test_activity_transition(self, recognizer):
        """Test transition between activities."""
        keypoints_standing = np.zeros((33, 3))
        keypoints_standing[23] = [0, 0, 1.0]
        keypoints_standing[24] = [0.2, 0, 1.0]
        keypoints_standing[25] = [0, 0, 0.5]
        keypoints_standing[26] = [0.2, 0, 0.5]
        keypoints_standing[27] = [0, 0, 0]
        keypoints_standing[28] = [0.2, 0, 0]

        keypoints_sitting = np.zeros((33, 3))
        keypoints_sitting[23] = [0, 0, 0.8]
        keypoints_sitting[24] = [0.2, 0, 0.8]
        keypoints_sitting[25] = [0, 0.3, 0.5]
        keypoints_sitting[26] = [0.2, 0.3, 0.5]
        keypoints_sitting[27] = [0, 0.5, 0.1]
        keypoints_sitting[28] = [0.2, 0.5, 0.1]

        velocity = np.array([0, 0, 0])

        # Start standing
        for _ in range(10):
            recognizer.update(1, keypoints_standing, velocity)
            time.sleep(0.01)

        activity = recognizer.get_current_activity(1)
        assert activity.activity_type == ActivityType.STANDING

        # Transition to sitting
        for _ in range(10):
            recognizer.update(1, keypoints_sitting, velocity)
            time.sleep(0.01)

        activity = recognizer.get_current_activity(1)
        assert activity.activity_type == ActivityType.SITTING

        # Check history
        history = recognizer.get_activity_history(1)
        assert len(history) > 0
        assert history[0].activity_type == ActivityType.STANDING

    def test_multiple_persons(self, recognizer):
        """Test tracking multiple persons simultaneously."""
        keypoints = np.zeros((33, 3))
        keypoints[23] = [0, 0, 1.0]
        keypoints[24] = [0.2, 0, 1.0]
        keypoints[25] = [0, 0, 0.5]
        keypoints[26] = [0.2, 0, 0.5]
        keypoints[27] = [0, 0, 0]
        keypoints[28] = [0.2, 0, 0]

        # Different velocities for different activities
        vel_standing = np.array([0, 0, 0])
        vel_walking = np.array([0.5, 0, 0])
        vel_running = np.array([2.5, 0, 0])

        for _ in range(10):
            recognizer.update(1, keypoints, vel_standing)
            recognizer.update(2, keypoints, vel_walking)
            recognizer.update(3, keypoints, vel_running)
            time.sleep(0.01)

        # Check each person has different activity
        activity_1 = recognizer.get_current_activity(1)
        activity_2 = recognizer.get_current_activity(2)
        activity_3 = recognizer.get_current_activity(3)

        assert activity_1.activity_type == ActivityType.STANDING
        assert activity_2.activity_type in [ActivityType.WALKING, ActivityType.STANDING]
        assert activity_3.activity_type == ActivityType.RUNNING

    def test_activity_duration(self, recognizer):
        """Test activity duration tracking."""
        keypoints = np.zeros((33, 3))
        keypoints[23] = [0, 0, 1.0]
        keypoints[24] = [0.2, 0, 1.0]
        keypoints[25] = [0, 0, 0.5]
        keypoints[26] = [0.2, 0, 0.5]
        keypoints[27] = [0, 0, 0]
        keypoints[28] = [0.2, 0, 0]

        velocity = np.array([0, 0, 0])

        # Update for known duration
        start_time = time.time()
        for _ in range(30):
            recognizer.update(1, keypoints, velocity)
            time.sleep(0.03)  # 30ms per update

        activity = recognizer.get_current_activity(1)

        # Duration should be approximately 30 * 0.03 = 0.9 seconds
        assert activity.duration > 0.5  # At least 0.5 seconds
        assert activity.duration < 2.0  # Less than 2 seconds

    def test_joint_angle_computation(self, recognizer):
        """Test joint angle computation."""
        # Three points forming 90 degree angle
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 0, 0])  # Joint
        p3 = np.array([0, 1, 0])

        angle = recognizer._compute_joint_angle(p1, p2, p3)

        # Should be 90 degrees
        np.testing.assert_allclose(angle, 90.0, rtol=1e-5)

    def test_joint_angle_straight(self, recognizer):
        """Test joint angle for straight configuration."""
        # Three collinear points
        p1 = np.array([1, 0, 0])
        p2 = np.array([0, 0, 0])
        p3 = np.array([-1, 0, 0])

        angle = recognizer._compute_joint_angle(p1, p2, p3)

        # Should be 180 degrees
        np.testing.assert_allclose(angle, 180.0, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
