from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np

@dataclass
class Fingertip:
    """Represents a fingertip with its position and pressing status."""
    position: Tuple[int, int]
    is_pressed: bool

class FingertipDetection:
    """Handles fingertip position detection and pressing status."""

    def detect(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> List[Fingertip]:
        """Detects fingertips in the given frame."""
        # Detect Hands with mediapipe and return fingertip coordinates relative to the cropped frame
        # - Also detect if a finger is pressed down
        # - Pressing status might require the hand detection to run before the perspective transformation,
        #   which would require the perspective transformation to be applied to the fingertip positions as well (I think mediapipe can do 3D coordinates to a certain extent)
        pass
    