from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from mediapipe.python.solutions.hands import Hands

@dataclass
class Fingertip:
    """Represents a fingertip with its position and pressing status."""
    position: Tuple[int, int]
    is_pressed: bool
    id: int

class FingertipDetection:
    """Handles fingertip position detection and pressing status."""
    def __init__(self):
        self.hands = Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4
        )
        
    def detect(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> List[Fingertip]:
        """Detects fingertips in the given frame."""
        result = self.hands.process(frame)
        if not result or not result.multi_hand_landmarks:
            return []  # Return empty list if no hands detected

        fingertips: List[Fingertip] = []

        # The landmark indices for fingertips in MediaPipe hand model
        # 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
        fingertip_indices = [4, 8, 12, 16, 20]

        for hand_idx, (landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_handedness)):
            if not landmarks or not landmarks.landmark:
                continue

            for finger_idx, fingertip_idx in enumerate(fingertip_indices):
                if fingertip_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[fingertip_idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    is_pressed = True

                    # Create consistent ID based on hand index and fingertip index
                    # Left hand: IDs 0-4, Right hand: IDs 5-9
                    hand_offset = 0 if handedness.classification[0].label == "Left" else 5
                    unique_id = hand_offset + finger_idx
                    
                    fingertips.append(Fingertip(
                        position=(x, y),
                        is_pressed=is_pressed,
                        id=unique_id
                    ))

        return fingertips
