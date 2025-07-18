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
        
    def detect(self, frame: np.ndarray) -> List[Tuple[int, int]]:
        """Detects ArUco markers in the given frame."""
        # Get marker corners
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_markers, marker_ids, _ = self.detector.detectMarkers(gray)

        marker_corners = [marker[0] for marker in detected_markers] if detected_markers else []
        marker_centers = [np.mean(corners, axis=0) for corners in marker_corners]
        marker_data = list(zip(marker_centers, marker_corners))
        
        # Draw detected markers on the frame
        if marker_ids is not None:
            aruco.drawDetectedMarkers(frame, detected_markers, marker_ids)

        # Convert marker centers and IDs to a list of tuples
        marker_data = []
        if marker_ids is not None:
            for i, marker_id in enumerate(marker_ids):
                marker_data.append((int(marker_id[0]), marker_centers[i]))

        return marker_data


    def get_transform_matrix(self, markers: List[Tuple[int, int]], width: int, height: int) -> Optional[np.ndarray]:
        """Calculates the perspective transformation matrix based on detected markers."""
        if len(markers) < 4:
            print(f"{len(markers)}/4 markers")
            return None
        
        marker_corners = [corners for _, corners in markers[:4]]
        src_pts= np.float32(np.array([corners[0] for corners in marker_corners]))

        dst_pts = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)

        matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
        return matrix

    def apply_transformation(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> np.ndarray:
        """Applies the perspective transformation to the frame."""
        if matrix is None:
            return frame  # If no valid matrix, return original frame
        return cv2.warpPerspective(frame, matrix, (frame.shape[1], frame.shape[0]))
