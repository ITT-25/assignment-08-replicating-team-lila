import numpy as np
import cv2

class MarkerDetection:
    """Handles ArUco marker detection and perspective transformation."""

    def detect(self, frame: np.ndarray) -> np.ndarray:
        """Detects ArUco markers in the given frame."""
        # Get markers
        pass

    def get_transform_matrix(self, markers: np.ndarray) -> np.ndarray:
        """Calculates the perspective transformation matrix based on detected markers."""
        pass

    def apply_transformation(self, frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Applies the perspective transformation to the frame."""
        return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))
