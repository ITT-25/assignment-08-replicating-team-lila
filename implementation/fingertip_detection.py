from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np
from mediapipe.python.solutions.hands import Hands
from mediapipe.python.solutions import drawing_utils
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
        self.fingertip_history: Deque[List[Fingertip]] = Deque(maxlen=cfg.SAMPLING_RATE // 4)
        


    def detect(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> List[Fingertip]:
        """Detects fingertips in the given frame and applies a perspective transform to their positions based on the provided matrix."""
        original_result = self.hands.process(frame)
        if not original_result or not original_result.multi_hand_landmarks:
            return []  # Return empty list if no hands detected

        fingertips: List[Fingertip] = []

        # The landmark indices for fingertips in MediaPipe hand model
        # 4=thumb, 8=index, 12=middle, 16=ring, 20=pinky
        fingertip_indices = [4, 8, 12, 16, 20]
        

        for hand_idx, (landmarks, world_landmarks, handedness) in enumerate(zip(original_result.multi_hand_landmarks, original_result.multi_hand_world_landmarks, original_result.multi_handedness)):
            if not landmarks or not landmarks.landmark or not world_landmarks or not world_landmarks.landmark:
                continue
            

            for finger_idx, fingertip_idx in enumerate(fingertip_indices):
                if fingertip_idx < len(landmarks.landmark):
                    landmark = landmarks.landmark[fingertip_idx]
                    world_landmark = world_landmarks.landmark[fingertip_idx]
            
                    drawing_utils.draw_landmarks(frame, landmarks)

                    # Convert normalized coordinates to pixel coordinates
                    x = int(landmark.x * frame.shape[1])
                    y = int(landmark.y * frame.shape[0])
                    z = world_landmark.z
                    
                    if matrix is not None:
                        # Apply perspective transformation to (x, y)
                        transformed_point = cv2.perspectiveTransform(
                            np.array([[[x, y]]], dtype=np.float32), matrix
                        )[0][0]
                        x, y = int(transformed_point[0]), int(transformed_point[1])


                    if fingertip_idx == 8:  # Index finger
                        print(f"Index finger position: ({z})")

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
        z_history: List[float] = []

        for fingertips in self.fingertip_history:
            for fingertip in fingertips:
                if fingertip.id == fingertip_id:
                    z_history.append(fingertip.position[2])

        if len(z_history) < self.fingertip_history.maxlen:
            return False

        # Smooth the z_history using a simple moving average
        smoothed_z_history = np.convolve(z_history, np.ones(3)/3, mode='valid')

        # positive z values indicate pressing (finger down), negative z values indicate hovering (finger up)
        # Find the min z value and perform a linear regression in both directions to detect a full pressing pattern
        min_index = np.argmin(smoothed_z_history)
        left_max_index = max(min_index - 1, 0)
        right_max_index = min(min_index + 1, len(smoothed_z_history) - 1)

        # Find local maxima to the left and right
        while left_max_index > 0 and smoothed_z_history[left_max_index - 1] >= smoothed_z_history[left_max_index]:
            left_max_index -= 1

        while right_max_index < len(smoothed_z_history) - 1 and smoothed_z_history[right_max_index + 1] >= smoothed_z_history[right_max_index]:
            right_max_index += 1

        # Ensure the segments have valid lengths
        if left_max_index >= min_index or right_max_index <= min_index:
            return False

        max_value = max(smoothed_z_history[left_max_index], smoothed_z_history[right_max_index])
        min_value = smoothed_z_history[min_index]

        # Check if the difference between maximum and minimum is significant
        if abs(max_value - smoothed_z_history[right_max_index]) > 0.01 or (max_value - min_value) < 0.01:
            return False

        # Perform linear regression on the two segments
        left_segment_x = np.arange(min_index - left_max_index + 1)
        left_segment_y = smoothed_z_history[left_max_index:min_index + 1]

        if len(left_segment_x) != len(left_segment_y):
            return False

        left_slope, _, _, _, _ = linregress(left_segment_x, left_segment_y)

        right_segment_x = np.arange(right_max_index - min_index + 1)
        right_segment_y = smoothed_z_history[min_index:right_max_index + 1]

        if len(right_segment_x) != len(right_segment_y):
            return False

        right_slope, _, _, _, _ = linregress(right_segment_x, right_segment_y)

        # Check for smooth downward slope followed by smooth upward slope
        # (negative left slope and positive right slope indicate pressing)
        if left_slope < -0.0001 and right_slope > 0.0001:
            return True

        return False

