"""
Camera calibration module for OMNISENSE platform.

Implements intrinsic and extrinsic camera calibration using chessboard patterns,
including stereo calibration for computing spatial relationships between cameras.
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from dataclasses import dataclass, asdict

from omnisense.utils.logger import get_logger
from omnisense.utils.exceptions import CameraCalibrationError

logger = get_logger(__name__)


@dataclass
class IntrinsicCalibration:
    """Intrinsic camera calibration parameters."""
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    image_size: Tuple[int, int]
    optimal_camera_matrix: np.ndarray
    roi: Tuple[int, int, int, int]  # Region of interest after undistortion
    reprojection_error: float


@dataclass
class ExtrinsicCalibration:
    """Extrinsic calibration between two cameras."""
    camera_1_id: int
    camera_2_id: int
    rotation_matrix: np.ndarray
    translation_vector: np.ndarray
    essential_matrix: np.ndarray
    fundamental_matrix: np.ndarray
    reprojection_error: float


class CameraCalibrator:
    """
    Handles camera calibration operations.

    Supports intrinsic calibration, stereo calibration, and calibration persistence.
    """

    def __init__(
        self,
        chessboard_size: Tuple[int, int] = (9, 6),
        square_size: float = 0.025  # meters
    ):
        """
        Initialize calibrator.

        Args:
            chessboard_size: Number of internal corners (width, height)
            square_size: Physical size of chessboard square in meters
        """
        self.chessboard_size = chessboard_size
        self.square_size = square_size

        # Prepare object points
        self.obj_points_template = self._prepare_object_points()

        logger.info(
            f"CameraCalibrator initialized with {chessboard_size} chessboard, "
            f"{square_size}m squares"
        )

    def _prepare_object_points(self) -> np.ndarray:
        """Prepare 3D object points for chessboard."""
        objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[
            0:self.chessboard_size[0],
            0:self.chessboard_size[1]
        ].T.reshape(-1, 2)
        objp *= self.square_size
        return objp

    def calibrate_intrinsic(
        self,
        images: List[np.ndarray],
        show_corners: bool = False
    ) -> IntrinsicCalibration:
        """
        Perform intrinsic camera calibration.

        Args:
            images: List of calibration images
            show_corners: If True, display detected corners for debugging

        Returns:
            IntrinsicCalibration object

        Raises:
            CameraCalibrationError: If calibration fails
        """
        logger.info(f"Starting intrinsic calibration with {len(images)} images")

        # Arrays to store object points and image points
        obj_points = []  # 3D points in real world space
        img_points = []  # 2D points in image plane

        image_size = None

        for i, img in enumerate(images):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            image_size = gray.shape[::-1]

            # Find chessboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.chessboard_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if ret:
                # Refine corner positions
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria
                )

                obj_points.append(self.obj_points_template)
                img_points.append(corners_refined)

                logger.debug(f"Found chessboard in image {i+1}/{len(images)}")

                if show_corners:
                    img_display = img.copy()
                    cv2.drawChessboardCorners(
                        img_display,
                        self.chessboard_size,
                        corners_refined,
                        ret
                    )
                    cv2.imshow(f'Corners - Image {i}', img_display)
                    cv2.waitKey(500)
            else:
                logger.warning(f"Chessboard not found in image {i+1}/{len(images)}")

        if show_corners:
            cv2.destroyAllWindows()

        if len(obj_points) < 10:
            raise CameraCalibrationError(
                f"Insufficient valid images for calibration: {len(obj_points)}/10 minimum"
            )

        logger.info(f"Calibrating with {len(obj_points)} valid images")

        # Calibrate camera
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points,
            img_points,
            image_size,
            None,
            None
        )

        if not ret:
            raise CameraCalibrationError("Camera calibration failed")

        # Compute reprojection error
        total_error = 0
        for i in range(len(obj_points)):
            img_points_reprojected, _ = cv2.projectPoints(
                obj_points[i],
                rvecs[i],
                tvecs[i],
                camera_matrix,
                dist_coeffs
            )
            error = cv2.norm(img_points[i], img_points_reprojected, cv2.NORM_L2) / len(img_points_reprojected)
            total_error += error

        mean_error = total_error / len(obj_points)

        # Compute optimal camera matrix for undistortion
        optimal_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            camera_matrix,
            dist_coeffs,
            image_size,
            1,
            image_size
        )

        logger.info(f"Intrinsic calibration complete. Reprojection error: {mean_error:.4f} pixels")

        return IntrinsicCalibration(
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            image_size=image_size,
            optimal_camera_matrix=optimal_camera_matrix,
            roi=roi,
            reprojection_error=mean_error
        )

    def calibrate_stereo(
        self,
        images_left: List[np.ndarray],
        images_right: List[np.ndarray],
        intrinsic_left: IntrinsicCalibration,
        intrinsic_right: IntrinsicCalibration,
        camera_1_id: int = 0,
        camera_2_id: int = 1
    ) -> ExtrinsicCalibration:
        """
        Perform stereo calibration between two cameras.

        Args:
            images_left: Calibration images from left camera
            images_right: Calibration images from right camera (synchronized)
            intrinsic_left: Intrinsic calibration for left camera
            intrinsic_right: Intrinsic calibration for right camera
            camera_1_id: ID of first camera
            camera_2_id: ID of second camera

        Returns:
            ExtrinsicCalibration object

        Raises:
            CameraCalibrationError: If calibration fails
        """
        logger.info(f"Starting stereo calibration with {len(images_left)} image pairs")

        if len(images_left) != len(images_right):
            raise CameraCalibrationError(
                f"Image count mismatch: {len(images_left)} left, {len(images_right)} right"
            )

        # Find chessboard corners in all images
        obj_points = []
        img_points_left = []
        img_points_right = []

        for i, (img_l, img_r) in enumerate(zip(images_left, images_right)):
            gray_l = cv2.cvtColor(img_l, cv2.COLOR_BGR2GRAY) if len(img_l.shape) == 3 else img_l
            gray_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2GRAY) if len(img_r.shape) == 3 else img_r

            # Find corners in both images
            ret_l, corners_l = cv2.findChessboardCorners(gray_l, self.chessboard_size, None)
            ret_r, corners_r = cv2.findChessboardCorners(gray_r, self.chessboard_size, None)

            if ret_l and ret_r:
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners_l = cv2.cornerSubPix(gray_l, corners_l, (11, 11), (-1, -1), criteria)
                corners_r = cv2.cornerSubPix(gray_r, corners_r, (11, 11), (-1, -1), criteria)

                obj_points.append(self.obj_points_template)
                img_points_left.append(corners_l)
                img_points_right.append(corners_r)

                logger.debug(f"Found chessboard in pair {i+1}/{len(images_left)}")
            else:
                logger.warning(f"Chessboard not found in pair {i+1}/{len(images_left)}")

        if len(obj_points) < 10:
            raise CameraCalibrationError(
                f"Insufficient valid image pairs: {len(obj_points)}/10 minimum"
            )

        logger.info(f"Calibrating stereo with {len(obj_points)} valid pairs")

        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC

        ret, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points,
            img_points_left,
            img_points_right,
            intrinsic_left.camera_matrix,
            intrinsic_left.dist_coeffs,
            intrinsic_right.camera_matrix,
            intrinsic_right.dist_coeffs,
            intrinsic_left.image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            flags=flags
        )

        if not ret:
            raise CameraCalibrationError("Stereo calibration failed")

        logger.info(f"Stereo calibration complete. Reprojection error: {ret:.4f} pixels")

        return ExtrinsicCalibration(
            camera_1_id=camera_1_id,
            camera_2_id=camera_2_id,
            rotation_matrix=R,
            translation_vector=T,
            essential_matrix=E,
            fundamental_matrix=F,
            reprojection_error=ret
        )

    def save_intrinsic_calibration(
        self,
        calibration: IntrinsicCalibration,
        filepath: Path
    ):
        """Save intrinsic calibration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'camera_matrix': calibration.camera_matrix.tolist(),
            'dist_coeffs': calibration.dist_coeffs.tolist(),
            'image_size': list(calibration.image_size),
            'optimal_camera_matrix': calibration.optimal_camera_matrix.tolist(),
            'roi': list(calibration.roi),
            'reprojection_error': float(calibration.reprojection_error)
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved intrinsic calibration to {filepath}")

    def load_intrinsic_calibration(self, filepath: Path) -> IntrinsicCalibration:
        """Load intrinsic calibration from YAML file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise CameraCalibrationError(f"Calibration file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        logger.info(f"Loaded intrinsic calibration from {filepath}")

        return IntrinsicCalibration(
            camera_matrix=np.array(data['camera_matrix']),
            dist_coeffs=np.array(data['dist_coeffs']),
            image_size=tuple(data['image_size']),
            optimal_camera_matrix=np.array(data['optimal_camera_matrix']),
            roi=tuple(data['roi']),
            reprojection_error=data['reprojection_error']
        )

    def save_extrinsic_calibration(
        self,
        calibration: ExtrinsicCalibration,
        filepath: Path
    ):
        """Save extrinsic calibration to YAML file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = {
            'camera_1_id': calibration.camera_1_id,
            'camera_2_id': calibration.camera_2_id,
            'rotation_matrix': calibration.rotation_matrix.tolist(),
            'translation_vector': calibration.translation_vector.tolist(),
            'essential_matrix': calibration.essential_matrix.tolist(),
            'fundamental_matrix': calibration.fundamental_matrix.tolist(),
            'reprojection_error': float(calibration.reprojection_error)
        }

        with open(filepath, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

        logger.info(f"Saved extrinsic calibration to {filepath}")

    def load_extrinsic_calibration(self, filepath: Path) -> ExtrinsicCalibration:
        """Load extrinsic calibration from YAML file."""
        filepath = Path(filepath)

        if not filepath.exists():
            raise CameraCalibrationError(f"Calibration file not found: {filepath}")

        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)

        logger.info(f"Loaded extrinsic calibration from {filepath}")

        return ExtrinsicCalibration(
            camera_1_id=data['camera_1_id'],
            camera_2_id=data['camera_2_id'],
            rotation_matrix=np.array(data['rotation_matrix']),
            translation_vector=np.array(data['translation_vector']),
            essential_matrix=np.array(data['essential_matrix']),
            fundamental_matrix=np.array(data['fundamental_matrix']),
            reprojection_error=data['reprojection_error']
        )

    def undistort_image(
        self,
        image: np.ndarray,
        calibration: IntrinsicCalibration
    ) -> np.ndarray:
        """
        Undistort an image using calibration parameters.

        Args:
            image: Input image
            calibration: Intrinsic calibration

        Returns:
            Undistorted image
        """
        return cv2.undistort(
            image,
            calibration.camera_matrix,
            calibration.dist_coeffs,
            None,
            calibration.optimal_camera_matrix
        )


def calibrate_cli():
    """Command-line interface for camera calibration."""
    import argparse

    parser = argparse.ArgumentParser(description='OMNISENSE Camera Calibration Tool')
    parser.add_argument('mode', choices=['intrinsic', 'stereo'], help='Calibration mode')
    parser.add_argument('--images', required=True, help='Directory containing calibration images')
    parser.add_argument('--output', required=True, help='Output calibration file path')
    parser.add_argument('--chessboard', default='9,6', help='Chessboard size (width,height)')
    parser.add_argument('--square-size', type=float, default=0.025, help='Square size in meters')
    parser.add_argument('--show-corners', action='store_true', help='Display detected corners')

    args = parser.parse_args()

    # Parse chessboard size
    chessboard_size = tuple(map(int, args.chessboard.split(',')))

    calibrator = CameraCalibrator(chessboard_size, args.square_size)

    # Load images
    from glob import glob
    image_paths = sorted(glob(f"{args.images}/*.jpg") + glob(f"{args.images}/*.png"))

    if not image_paths:
        print(f"No images found in {args.images}")
        return

    print(f"Loading {len(image_paths)} images...")
    images = [cv2.imread(p) for p in image_paths]

    if args.mode == 'intrinsic':
        calibration = calibrator.calibrate_intrinsic(images, args.show_corners)
        calibrator.save_intrinsic_calibration(calibration, Path(args.output))
        print(f"Intrinsic calibration saved to {args.output}")
        print(f"Reprojection error: {calibration.reprojection_error:.4f} pixels")

    print("Calibration complete!")


if __name__ == '__main__':
    calibrate_cli()
