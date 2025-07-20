from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple
import cv2
import numpy as np
from mediapipe.python.solutions.hands import Hands
from mediapipe.python.solutions import drawing_utils
import config as cfg


# The landmark indices for fingertips and their bases in MediaPipe hand model
# Each tuple: (base_idx, tip_idx)
finger_landmark_pairs = [(1, 4), (5, 8), (9, 12), (13, 16), (17, 20)]

@dataclass
class Fingertip:
    """Represents a fingertip with its position, base position, and pressing status."""
    position: Tuple[int, int, float]
    base_position: Tuple[int, int]
    is_pressed: bool
    id: int

class FingertipDetection:
    """Handles fingertip position detection and pressing status."""
    def __init__(self):
        self.hands = Hands(
            static_image_mode=False, max_num_hands=2, min_detection_confidence=0.4, min_tracking_confidence=0.4
        )
        self.fingertip_history: Deque[List[Fingertip]] = Deque(maxlen=cfg.SAMPLING_RATE // 4)
        self.hand_mask: Optional[np.ndarray] = None
        

    def detect(self, frame: np.ndarray, matrix: Optional[np.ndarray]) -> List[Fingertip]:
        """Detects fingertips in the given frame and applies a perspective transform to their positions based on the provided matrix."""
        # Process the image with MediaPipe
        original_result = self.hands.process(frame)
        
        # Initialize an empty hand mask
        self.hand_mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
        
        if not original_result or not original_result.multi_hand_landmarks:
            return []  # Return empty list if no hands detected

        fingertips: List[Fingertip] = []

        
        # Generate hand mask from landmarks
        self._generate_hand_mask(frame, original_result.multi_hand_landmarks)

        for hand_idx, (landmarks, world_landmarks, handedness) in enumerate(zip(original_result.multi_hand_landmarks, original_result.multi_hand_world_landmarks, original_result.multi_handedness)):
            if not landmarks or not landmarks.landmark or not world_landmarks or not world_landmarks.landmark:
                continue

            for finger_idx, (base_idx, tip_idx) in enumerate(finger_landmark_pairs):
                if tip_idx < len(landmarks.landmark) and base_idx < len(landmarks.landmark):
                    tip_landmark = landmarks.landmark[tip_idx]
                    tip_world_landmark = world_landmarks.landmark[tip_idx]
                    base_landmark = landmarks.landmark[base_idx]

                    drawing_utils.draw_landmarks(frame, landmarks)

                    # Convert normalized coordinates to pixel coordinates
                    x = int(tip_landmark.x * frame.shape[1])
                    y = int(tip_landmark.y * frame.shape[0])
                    z = tip_world_landmark.z
                    
                    # Get base position in pixel coordinates
                    base_x = int(base_landmark.x * frame.shape[1])
                    base_y = int(base_landmark.y * frame.shape[0])

                    if matrix is not None:
                        # Apply perspective transformation to (x, y)
                        transformed_point = cv2.perspectiveTransform(
                            np.array([[[x, y]]], dtype=np.float32), matrix
                        )[0][0]
                        x, y = int(transformed_point[0]), int(transformed_point[1])
                        
                        # Apply perspective transformation to base position
                        transformed_base = cv2.perspectiveTransform(
                            np.array([[[base_x, base_y]]], dtype=np.float32), matrix
                        )[0][0]
                        base_x, base_y = int(transformed_base[0]), int(transformed_base[1])
                        
                    # Flip the y based on the frame height
                    y = frame.shape[0] - y
                    base_y = frame.shape[0] - base_y

                    # Create consistent ID based on hand index and fingertip index
                    # Left hand: IDs 0-4, Right hand: IDs 5-9
                    hand_offset = 0 if handedness.classification[0].label == "Left" else 5
                    unique_id = hand_offset + finger_idx

                    fingertips.append(Fingertip(
                        position=(x, y, z),
                        base_position=(base_x, base_y),
                        is_pressed=False,
                        id=unique_id
                    ))

        self.fingertip_history.append(fingertips)
        for fingertip in fingertips:
            fingertip.is_pressed = self._detect_pressing_status(fingertip.id)
        return fingertips
    
    def _generate_hand_mask(self, frame: np.ndarray, multi_hand_landmarks: List) -> None:
        """Generates a mask for hands using MediaPipe hand landmarks."""
        height, width = frame.shape[:2]
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Define hand connections based on MediaPipe hand landmark model
        connections = [
            # Thumb
            [0, 1, 2, 3, 4, 3, 2, 1],
            # Index finger
            [0, 5, 6, 7, 8, 7, 6, 5],
            # Middle finger
            [5, 9, 10, 11, 12, 11, 10, 9],
            # Ring finger
            [9, 13, 14, 15, 16, 15, 14, 13],
            # Pinky
            [13, 17, 18, 19, 20, 19, 18, 17],
            # Palm
            [0, 1, 5, 9, 13, 17, 0]
        ]
        
        for landmarks in multi_hand_landmarks:
            # Create detailed masks for each part of the hand
            for connection in connections:
                pts = []
                for idx in connection:
                    landmark = landmarks.landmark[idx]
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    pts.append([x, y])
                if pts:
                    cv2.fillPoly(mask, [np.array(pts)], 255)
            
            # Create a convex hull around the palm landmarks for better coverage
            palm_landmarks = [0, 1, 5, 9, 13, 17]
            palm_pts = []
            for idx in palm_landmarks:
                landmark = landmarks.landmark[idx]
                x, y = int(landmark.x * width), int(landmark.y * height)
                palm_pts.append([x, y])
            if len(palm_pts) >= 3:  # Need at least 3 points for convex hull
                hull = cv2.convexHull(np.array(palm_pts))
                cv2.fillConvexPoly(mask, hull, 255)
            
            # Draw lines for fingers
            finger_connections = [
                (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
                (0, 5), (5, 6), (6, 7), (7, 8),  # Index
                (5, 9), (9, 10), (10, 11), (11, 12),  # Middle
                (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
                (13, 17), (17, 18), (18, 19), (19, 20)  # Pinky
            ]
            
            for start_idx, end_idx in finger_connections:
                start = landmarks.landmark[start_idx]
                end = landmarks.landmark[end_idx]
                start_x, start_y = int(start.x * width), int(start.y * height)
                end_x, end_y = int(end.x * width), int(end.y * height)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, thickness=8)
            
            # Additional connections between finger bases to improve palm coverage
            base_connections = [(1, 5), (5, 9), (9, 13), (13, 17)]
            for start_idx, end_idx in base_connections:
                start = landmarks.landmark[start_idx]
                end = landmarks.landmark[end_idx]
                start_x, start_y = int(start.x * width), int(start.y * height)
                end_x, end_y = int(end.x * width), int(end.y * height)
                cv2.line(mask, (start_x, start_y), (end_x, end_y), 255, thickness=8)
        
        # Apply morphological operations to make the mask thicker
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        self.hand_mask = mask
    
    def get_hand_mask(self) -> np.ndarray:
        """Returns the last generated hand mask."""
        if self.hand_mask is None:
            # Return empty mask if none has been generated yet
            return np.zeros((cfg.VIDEO_HEIGHT, cfg.VIDEO_WIDTH), dtype=np.uint8)
        return self.hand_mask
    
    def _detect_pressing_status(self, fingertip_id: int) -> bool:
        """Detects fingertip pressing status using the pattern of x,y distance changes between base and tip."""
        # Collect distance history for this fingertip
        distance_history: List[float] = []
        
        # Look through history to find this fingertip's past positions
        for frame_fingertips in self.fingertip_history:
            for fingertip in frame_fingertips:
                if fingertip.id == fingertip_id:
                    tip_x, tip_y, _ = fingertip.position
                    base_x, base_y = fingertip.base_position
                    distance = np.sqrt((tip_x - base_x)**2 + (tip_y - base_y)**2)
                    distance_history.append(distance)
        
        # Not enough history to make a determination
        if len(distance_history) < 2:
            # Get current distance and just use the static threshold
            if distance_history:
                current_distance = distance_history[-1]
                return current_distance > cfg.PRESS_DISTANCE_THRESHOLD
            return False
        
        # Apply smoothing to reduce noise (if we have enough samples)
        if len(distance_history) >= 3:
            smoothed_distances = np.convolve(distance_history, np.ones(3)/3, mode="valid")
        else:
            smoothed_distances = np.array(distance_history)
        
        # Calculate distance velocity (positive = finger extending quickly)
        # Make sure we have at least 2 elements before calculating velocity
        if len(smoothed_distances) >= 2:
            distance_velocity = smoothed_distances[-1] - smoothed_distances[-2]
            current_distance = smoothed_distances[-1]
            
            # Detect a press when:
            # 1. Current distance is above the threshold (finger is extended)
            # 2. Distance is increasing rapidly (finger is being thrust forward)
            return (current_distance > cfg.PRESS_DISTANCE_THRESHOLD and 
                    distance_velocity > cfg.DISTANCE_VELOCITY_THRESHOLD)
        else:
            # Fallback to just checking the threshold if we can't calculate velocity
            current_distance = smoothed_distances[-1]
            return current_distance > cfg.PRESS_DISTANCE_THRESHOLD

