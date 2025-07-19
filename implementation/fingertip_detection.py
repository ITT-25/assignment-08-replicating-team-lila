from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import numpy as np
from mediapipe.python.solutions.hands import Hands
import config as cfg
from scipy.stats import linregress

@dataclass
class Fingertip:
    """Represents a fingertip with its position and pressing status."""
    position: Tuple[int, int, float]
    is_pressed: bool
    id: int

class FingertipDetection:
    """Handles fingertip position detection and pressing status."""
    def __init__(self):
        self.hands = Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4
        )
        self.fingertip_history: Deque[List[Fingertip]] = Deque(maxlen=int(cfg.SAMPLING_RATE / 2))


    def detect(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> List[Fingertip]:
        """Detects fingertips in the given frame."""
        result = self.hands.process(frame)
        if not result or not result.multi_hand_landmarks:
            return []  # Return empty list if no hands detected

        fingertips: List[Fingertip] = []

        # The landmark indices for fingertips in MediaPipe hand model
        # 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
        fingertip_indices = [4, 8, 12, 16, 20]

        for hand_idx, (landmarks, world_landmarks, handedness) in enumerate(zip(result.multi_hand_landmarks, result.multi_hand_world_landmarks, result.multi_handedness)):
            if not landmarks or not landmarks.landmark:
                continue

            for finger_idx, fingertip_idx in enumerate(fingertip_indices):
                if fingertip_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[fingertip_idx]
                    world_landmark = world_landmarks.landmark[fingertip_idx]

                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = world_landmark.z

                    # Create consistent ID based on hand index and fingertip index
                    # Left hand: IDs 0-4, Right hand: IDs 5-9
                    hand_offset = 0 if handedness.classification[0].label == "Left" else 5
                    unique_id = hand_offset + finger_idx
                    
                    fingertips.append(Fingertip(
                        position=(x, y, z),
                        is_pressed=False,
                        id=unique_id
                    ))
        self.fingertip_history.append(fingertips)
        for fingertip in fingertips:
            fingertip.is_pressed = self._detect_pressing_status(fingertip.id)
        return fingertips
    
    def _detect_pressing_status(self, fingertip_id: int) -> bool:
        """Detects fingertip pressing status based on patterns in z coordinate history."""
        # Create a list of z coordinates for the specified fingertip ID from the history
        z_history: List[float] = []
        
        for fingertips in self.fingertip_history:
            for fingertip in fingertips:
                if fingertip.id == fingertip_id:
                    z_history.append(fingertip.position[2])
        
        if len(z_history) < self.fingertip_history.maxlen:
            return False
        
        # TODO properly detect pressing motion from z_history

        x = np.arange(len(z_history))
        slope, intercept, r_value, p_value, std_err = linregress(x, z_history) #Calculate upward or downward tendencies

        if slope > 0.0001:
            #print("Die Zahlen steigen tendenziell.")
            print("True")
            return True
        elif slope < -0.0001:
            #print("Die Zahlen fallen tendenziell.")
            print("False")
            return False
        else:
            print("Keine Tendenz erkennbar.")

        return False

