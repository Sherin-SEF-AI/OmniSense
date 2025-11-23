"""
Geometric utilities for 3D vision and multi-view geometry.

Provides functions for coordinate transformations, triangulation, projection,
and other geometric operations needed for the spatial intelligence system.
"""

import numpy as np
from typing import Tuple, List, Optional
import cv2
from scipy.spatial.transform import Rotation


def project_3d_to_2d(
    points_3d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: Optional[np.ndarray] = None,
    tvec: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Project 3D points to 2D image coordinates.

    Args:
        points_3d: Nx3 array of 3D points
        camera_matrix: 3x3 camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector (3x1) - optional
        tvec: Translation vector (3x1) - optional

    Returns:
        Nx2 array of 2D image points
    """
    if rvec is None:
        rvec = np.zeros(3)
    if tvec is None:
        tvec = np.zeros(3)

    points_2d, _ = cv2.projectPoints(
        points_3d.reshape(-1, 1, 3),
        rvec, tvec,
        camera_matrix,
        dist_coeffs
    )
    return points_2d.reshape(-1, 2)


def triangulate_points(
    points1: np.ndarray,
    points2: np.ndarray,
    proj_matrix1: np.ndarray,
    proj_matrix2: np.ndarray
) -> np.ndarray:
    """
    Triangulate 3D points from two views using DLT.

    Args:
        points1: Nx2 array of points in first image
        points2: Nx2 array of points in second image
        proj_matrix1: 3x4 projection matrix for first camera
        proj_matrix2: 3x4 projection matrix for second camera

    Returns:
        Nx3 array of triangulated 3D points
    """
    points_4d = cv2.triangulatePoints(
        proj_matrix1,
        proj_matrix2,
        points1.T,
        points2.T
    )

    # Convert from homogeneous to 3D coordinates
    points_3d = points_4d[:3] / points_4d[3]
    return points_3d.T


def triangulate_multi_view(
    points_list: List[np.ndarray],
    proj_matrices: List[np.ndarray],
    method: str = 'dlt'
) -> np.ndarray:
    """
    Triangulate 3D points from multiple views.

    Args:
        points_list: List of Nx2 arrays, one per camera
        proj_matrices: List of 3x4 projection matrices
        method: Triangulation method ('dlt' or 'ransac')

    Returns:
        Nx3 array of triangulated 3D points
    """
    n_views = len(points_list)
    n_points = points_list[0].shape[0]

    points_3d = np.zeros((n_points, 3))

    for i in range(n_points):
        # Build linear system for DLT
        A = []
        for view_idx in range(n_views):
            if view_idx >= len(points_list):
                continue

            x, y = points_list[view_idx][i]
            P = proj_matrices[view_idx]

            A.append(y * P[2] - P[1])
            A.append(P[0] - x * P[2])

        A = np.array(A)

        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X = Vt[-1]
        X = X / X[3]  # Normalize homogeneous coordinate

        points_3d[i] = X[:3]

    return points_3d


def rotation_matrix_to_euler(R: np.ndarray) -> Tuple[float, float, float]:
    """
    Convert rotation matrix to Euler angles (roll, pitch, yaw).

    Args:
        R: 3x3 rotation matrix

    Returns:
        Tuple of (roll, pitch, yaw) in radians
    """
    rotation = Rotation.from_matrix(R)
    euler = rotation.as_euler('xyz', degrees=False)
    return tuple(euler)


def euler_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles to rotation matrix.

    Args:
        roll: Roll angle in radians
        pitch: Pitch angle in radians
        yaw: Yaw angle in radians

    Returns:
        3x3 rotation matrix
    """
    rotation = Rotation.from_euler('xyz', [roll, pitch, yaw], degrees=False)
    return rotation.as_matrix()


def transform_points(
    points: np.ndarray,
    R: np.ndarray,
    t: np.ndarray
) -> np.ndarray:
    """
    Apply rigid transformation to 3D points.

    Args:
        points: Nx3 array of 3D points
        R: 3x3 rotation matrix
        t: 3x1 translation vector

    Returns:
        Nx3 array of transformed points
    """
    return (R @ points.T).T + t.ravel()


def compute_projection_matrix(
    camera_matrix: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray
) -> np.ndarray:
    """
    Compute 3x4 projection matrix from camera parameters.

    Args:
        camera_matrix: 3x3 camera intrinsic matrix
        rvec: 3x1 rotation vector
        tvec: 3x1 translation vector

    Returns:
        3x4 projection matrix
    """
    R, _ = cv2.Rodrigues(rvec)
    Rt = np.hstack([R, tvec.reshape(-1, 1)])
    return camera_matrix @ Rt


def point_to_line_distance(
    point: np.ndarray,
    line_point: np.ndarray,
    line_direction: np.ndarray
) -> float:
    """
    Compute distance from a point to a line in 3D.

    Args:
        point: 3D point
        line_point: A point on the line
        line_direction: Direction vector of the line (normalized)

    Returns:
        Distance from point to line
    """
    line_direction = line_direction / np.linalg.norm(line_direction)
    point_to_line_point = point - line_point
    projection_length = np.dot(point_to_line_point, line_direction)
    projection = projection_length * line_direction
    perpendicular = point_to_line_point - projection
    return np.linalg.norm(perpendicular)


def ray_intersection_3d(
    origins: np.ndarray,
    directions: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Find the point of closest approach for multiple rays in 3D.

    Uses least squares to find the point that minimizes the sum of squared
    distances to all rays.

    Args:
        origins: Nx3 array of ray origins
        directions: Nx3 array of ray directions (should be normalized)

    Returns:
        Tuple of (intersection_point, residual_error)
    """
    n_rays = origins.shape[0]

    # Normalize directions
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    # Build linear system
    A = np.zeros((3, 3))
    b = np.zeros(3)

    for i in range(n_rays):
        d = directions[i]
        o = origins[i]

        # I - d*d^T (projection onto plane perpendicular to ray)
        M = np.eye(3) - np.outer(d, d)

        A += M
        b += M @ o

    # Solve for intersection point
    intersection = np.linalg.lstsq(A, b, rcond=None)[0]

    # Compute residual error
    error = 0
    for i in range(n_rays):
        dist = point_to_line_distance(intersection, origins[i], directions[i])
        error += dist ** 2

    return intersection, np.sqrt(error / n_rays)


def compute_reprojection_error(
    points_3d: np.ndarray,
    points_2d: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray
) -> float:
    """
    Compute average reprojection error.

    Args:
        points_3d: Nx3 array of 3D points
        points_2d: Nx2 array of observed 2D points
        camera_matrix: 3x3 camera matrix
        dist_coeffs: Distortion coefficients
        rvec: Rotation vector
        tvec: Translation vector

    Returns:
        Average reprojection error in pixels
    """
    projected, _ = cv2.projectPoints(
        points_3d, rvec, tvec, camera_matrix, dist_coeffs
    )
    projected = projected.reshape(-1, 2)
    error = np.linalg.norm(points_2d - projected, axis=1)
    return np.mean(error)


def homogeneous_to_euclidean(points: np.ndarray) -> np.ndarray:
    """Convert homogeneous coordinates to Euclidean coordinates."""
    return points[..., :-1] / points[..., -1:]


def euclidean_to_homogeneous(points: np.ndarray) -> np.ndarray:
    """Convert Euclidean coordinates to homogeneous coordinates."""
    ones = np.ones((*points.shape[:-1], 1))
    return np.concatenate([points, ones], axis=-1)


def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
    """
    Compute angle between two vectors in radians.

    Args:
        v1: First vector
        v2: Second vector

    Returns:
        Angle in radians [0, pi]
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
    return np.arccos(cos_angle)
