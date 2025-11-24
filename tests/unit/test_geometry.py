"""Unit tests for geometry utilities."""

import pytest
import numpy as np

from omnisense.utils.geometry import (
    triangulate_points,
    triangulate_multi_view,
    ray_intersection_3d,
    project_points,
    compute_fundamental_matrix,
    compute_epipolar_error,
    rotation_matrix_to_euler,
    euler_to_rotation_matrix,
    transform_points
)


class TestTriangulation:
    """Test triangulation functions."""

    def test_triangulate_points_basic(self):
        """Test basic two-view triangulation."""
        # Setup simple stereo configuration
        # Camera 1 at origin, Camera 2 at [1, 0, 0]
        proj1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        proj2 = np.array([
            [1, 0, 0, -1],
            [0, 1, 0, 0],
            [0, 0, 1, 0]
        ])

        # Point in 3D at [0.5, 0, 2]
        point_3d = np.array([0.5, 0, 2])

        # Project to both cameras
        point_2d_1 = project_points(point_3d.reshape(1, 3), proj1)[0]
        point_2d_2 = project_points(point_3d.reshape(1, 3), proj2)[0]

        # Triangulate back
        triangulated = triangulate_points(
            point_2d_1.reshape(1, 2),
            point_2d_2.reshape(1, 2),
            proj1,
            proj2
        )

        # Should recover original point
        assert triangulated.shape == (1, 3)
        np.testing.assert_allclose(triangulated[0], point_3d, rtol=1e-5, atol=1e-5)

    def test_triangulate_multi_view(self):
        """Test multi-view triangulation."""
        # Three camera configuration
        proj_matrices = [
            np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]]),
            np.array([[1, 0, 0, -1], [0, 1, 0, 0], [0, 0, 1, 0]]),
            np.array([[1, 0, 0, 0], [0, 1, 0, -1], [0, 0, 1, 0]])
        ]

        # 3D point
        point_3d = np.array([0.5, 0.5, 2])

        # Project to all cameras
        points_2d = [project_points(point_3d.reshape(1, 3), proj)[0]
                     for proj in proj_matrices]

        # Triangulate
        triangulated = triangulate_multi_view(points_2d, proj_matrices)

        # Verify
        assert triangulated.shape == (3,)
        np.testing.assert_allclose(triangulated, point_3d, rtol=1e-4, atol=1e-4)


class TestRayIntersection:
    """Test ray intersection function."""

    def test_ray_intersection_perfect(self):
        """Test ray intersection with perfectly intersecting rays."""
        # Two rays that intersect at [1, 1, 1]
        origins = np.array([
            [0, 0, 0],
            [2, 0, 0]
        ])

        directions = np.array([
            [1, 1, 1],
            [-1, 1, 1]
        ])

        # Normalize directions
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        point, error = ray_intersection_3d(origins, directions)

        assert point.shape == (3,)
        assert error < 1e-6
        np.testing.assert_allclose(point, [1, 1, 1], rtol=1e-5, atol=1e-5)

    def test_ray_intersection_skew(self):
        """Test ray intersection with skew rays."""
        # Two skew rays
        origins = np.array([
            [0, 0, 0],
            [1, 1, 0]
        ])

        directions = np.array([
            [1, 0, 0],
            [0, 1, 0]
        ])

        point, error = ray_intersection_3d(origins, directions)

        # Should find closest point of approach
        assert point.shape == (3,)
        assert error > 0  # Non-zero error for skew rays


class TestRotationConversion:
    """Test rotation matrix and Euler angle conversions."""

    def test_euler_to_rotation_matrix(self):
        """Test Euler to rotation matrix conversion."""
        # Identity rotation
        R = euler_to_rotation_matrix(0, 0, 0)
        np.testing.assert_allclose(R, np.eye(3), rtol=1e-10)

        # 90 degree rotation around Z
        R = euler_to_rotation_matrix(0, 0, 90)
        expected = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        np.testing.assert_allclose(R, expected, rtol=1e-10, atol=1e-10)

    def test_rotation_matrix_to_euler(self):
        """Test rotation matrix to Euler conversion."""
        # Round-trip test
        roll, pitch, yaw = 10, 20, 30

        R = euler_to_rotation_matrix(roll, pitch, yaw)
        roll_out, pitch_out, yaw_out = rotation_matrix_to_euler(R)

        np.testing.assert_allclose([roll_out, pitch_out, yaw_out],
                                  [roll, pitch, yaw],
                                  rtol=1e-6, atol=1e-6)

    def test_euler_roundtrip_special_cases(self):
        """Test Euler conversion round-trip with special cases."""
        test_cases = [
            (0, 0, 0),
            (90, 0, 0),
            (0, 90, 0),
            (0, 0, 90),
            (45, 45, 45),
            (-30, 60, -45)
        ]

        for roll, pitch, yaw in test_cases:
            R = euler_to_rotation_matrix(roll, pitch, yaw)
            roll_out, pitch_out, yaw_out = rotation_matrix_to_euler(R)

            # Reconstruct rotation matrix
            R_out = euler_to_rotation_matrix(roll_out, pitch_out, yaw_out)

            # Matrices should be equal
            np.testing.assert_allclose(R, R_out, rtol=1e-6, atol=1e-6)


class TestPointTransformation:
    """Test point transformation functions."""

    def test_transform_points_identity(self):
        """Test transformation with identity matrix."""
        points = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        T = np.eye(4)
        transformed = transform_points(points, T)

        np.testing.assert_allclose(transformed, points, rtol=1e-10)

    def test_transform_points_translation(self):
        """Test transformation with translation."""
        points = np.array([
            [1, 2, 3],
            [4, 5, 6]
        ])

        # Translation by [10, 20, 30]
        T = np.eye(4)
        T[:3, 3] = [10, 20, 30]

        transformed = transform_points(points, T)

        expected = points + np.array([10, 20, 30])
        np.testing.assert_allclose(transformed, expected, rtol=1e-10)

    def test_transform_points_rotation(self):
        """Test transformation with rotation."""
        points = np.array([[1, 0, 0]])

        # 90 degree rotation around Z
        R = euler_to_rotation_matrix(0, 0, 90)
        T = np.eye(4)
        T[:3, :3] = R

        transformed = transform_points(points, T)

        expected = np.array([[0, 1, 0]])
        np.testing.assert_allclose(transformed, expected, rtol=1e-10, atol=1e-10)


class TestFundamentalMatrix:
    """Test fundamental matrix computation."""

    def test_compute_fundamental_matrix(self):
        """Test fundamental matrix computation with known correspondences."""
        # Generate synthetic point correspondences
        # Using a simple stereo setup
        np.random.seed(42)

        # Random 3D points
        points_3d = np.random.rand(20, 3) * 10
        points_3d[:, 2] += 5  # Ensure points are in front of cameras

        # Camera matrices
        K = np.array([[1000, 0, 500], [0, 1000, 500], [0, 0, 1]])
        P1 = np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = np.hstack([np.eye(3), np.array([[1], [0], [0]])])  # Baseline of 1

        # Project points
        points1_h = (K @ P1 @ np.hstack([points_3d, np.ones((20, 1))]).T).T
        points1 = (points1_h[:, :2] / points1_h[:, 2:3])

        points2_h = (K @ P2 @ np.hstack([points_3d, np.ones((20, 1))]).T).T
        points2 = (points2_h[:, :2] / points2_h[:, 2:3])

        # Compute fundamental matrix
        F, inliers = compute_fundamental_matrix(points1, points2)

        assert F is not None
        assert F.shape == (3, 3)

        # Check epipolar constraint: p2^T F p1 = 0
        errors = compute_epipolar_error(points1, points2, F)
        assert np.mean(errors) < 1.0  # Should be close to zero


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
