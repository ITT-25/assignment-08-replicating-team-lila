from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.gesture_recognizer import GestureRecognizer, GestureRecognizerOptions

@dataclass
class Fingertip:
    """Represents a fingertip with its position and pressing status."""
    position: Tuple[int, int]
    is_pressed: bool
    id: int

class FingertipDetection:
    """Handles fingertip position detection and pressing status."""
    def __init__(self, model_path: str = "pointing_input/gesture_recognizer.task"):
        base_options = python.BaseOptions(model_asset_buffer=open(model_path, "rb").read())
        options = GestureRecognizerOptions(base_options=base_options, num_hands=2)
        self.recognizer = GestureRecognizer.create_from_options(options)
        
    def detect(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> List[Fingertip]:
        """Detects fingertips in the given frame."""
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        result = self.recognizer.recognize(mp_image)
        
        fingertips: List[Fingertip] = []

        # The landmark indices for fingertips in MediaPipe hand model
        # 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
        fingertip_indices = [4, 8, 12, 16, 20]
        
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            if not landmarks:
                continue
            
            for i, fingertip_idx in enumerate(fingertip_indices):
                if fingertip_idx < len(landmarks):
                    landmark = landmarks[fingertip_idx]
                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    is_pressed = True
                    
                    # Create consistent ID based on hand index and fingertip index
                    # This gives each finger a unique ID (0-9 for two hands with 5 fingers each)
                    
                    fingertips.append(Fingertip(
                        position=(x, y),
                        is_pressed=is_pressed,
                        id=fingertip_idx if handedness.category_name == 'Left' else (fingertip_idx + 1)
                    ))

        return fingertips
