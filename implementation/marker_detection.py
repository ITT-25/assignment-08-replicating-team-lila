import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict
import cv2.aruco as aruco

class MarkerDetection:
    """Handles ArUco marker detection and perspective transformation."""
    def __init__(self, aruco_dict: Optional[aruco.Dictionary] = aruco.DICT_6X6_250):
        self.aruco_dict = aruco.getPredefinedDictionary(aruco_dict)
        self.detector = aruco.ArucoDetector(self.aruco_dict, aruco.DetectorParameters())
        self.marker_cache: Dict[int, Tuple[np.ndarray, float]] = {}

    
    def sort_markers(self, arr):
        arr_np = np.array(arr)

        # Sort points by their y-coordinate
        sort_y = sorted(arr_np, key=lambda x: x[1])

        # Split sorted points into top and bottom
        top_y = sort_y[:2]
        bottom_y = sort_y[2:]

        # Sort points by position
        top_left = min(top_y, key=lambda x: x[0])
        top_right = max(top_y, key=lambda x: x[0])
        bottom_left = min(bottom_y, key=lambda x: x[0])
        bottom_right = max(bottom_y, key=lambda x: x[0])

        return np.float32(np.array([top_left, top_right, bottom_right, bottom_left]))

        
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detects ArUco markers in the given frame and updates the marker cache."""
        # Get marker corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_markers, marker_ids, _ = self.detector.detectMarkers(gray)

        marker_corners = [marker[0] for marker in detected_markers] if detected_markers else []
        marker_centers = [np.mean(corners, axis=0) for corners in marker_corners]

        # Update marker cache with detected markers
        if marker_ids is not None:
            for i, marker_id in enumerate(marker_ids):
                self.marker_cache[int(marker_id[0])] = (marker_centers[i], marker_corners[i])

        # Reuse cached markers if not visible in the current frame
        marker_data = []
        for marker_id, (center, corners) in self.marker_cache.items():
            marker_data.append((marker_id, center))

        # Draw detected markers on the frame
        if marker_ids is not None:
            aruco.drawDetectedMarkers(frame, detected_markers, marker_ids)

        return marker_data


    def get_transform_matrix(self, markers: List[Tuple[int, int]], width: int, height: int) -> np.ndarray:
        """Calculates the perspective transformation matrix based on detected markers."""
        src_pts= np.float32(np.array([corners[1] for corners in markers]))

        src_pts = self.sort_markers(src_pts)

        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix

    def apply_transformation(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> np.ndarray:
        """Applies the perspective transformation to the frame."""
        if matrix is None:
            return frame  # If no valid matrix, return original frame
        return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))
