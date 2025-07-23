import time
from typing import Optional, Tuple, List, Dict, Deque
from collections import defaultdict, deque
from dataclasses import dataclass
from fingertip_detection import Fingertip
import config

@dataclass
class Note:
    """Represents a musical note with pitch and key."""
    key: str
    octave: int
    pitch: float = 1.0
    last_activation: Optional[Tuple[float, Fingertip]] = None
    center: Tuple[int, int] = (0, 0)
    width: int = 70
    height: int = 0

class Piano:
    """Maps fingertip positions to piano keys."""
    
    def __init__(self, num_octaves: int = 2, pitch_range: Tuple[float, float] = (0.5, 2.0), pitch_bend_thresh: float = 150.0, vibrato_thresh_x: float = 20.0, vibrato_thresh_y: float = 50.0):
        self.num_octaves: int = num_octaves
        self.pitch_range: Tuple[float, float] = pitch_range

        self.pitch_bend_thresh: float = pitch_bend_thresh   # Minimum y-movement required for pitch bends
        self.vibrato_thresh_x: float = vibrato_thresh_x   # Minimum x-movement required for vibrato
        self.vibrato_thresh_y: float = vibrato_thresh_y   # Maximum y-movement allowed during vibrato
        self.vibrato_time_window = 0.3      # Temporal threshold between direction changes for vibrato (in seconds)
        self.vibrato_amplitude_window = 5.0     # Minimum x distance between reversals to register vibrato
        self.vibrato_history: Dict[int, Deque[Tuple[float, float]]] = defaultdict(lambda: deque(maxlen=15))    # Tracks timestamp and x-position per fingertip for vibrato detection

        config.WINDOW_WIDTH = Note.width * self.num_octaves * 7
        config.WINDOW_HEIGHT = int(config.WINDOW_WIDTH // 1.77)
        self.keys: List[Note] = self._generate_keys()

        self.sharp_keys = [key for key in self.keys if '#' in key.key]
        self.natural_keys = [key for key in self.keys if '#' not in key.key]

    def _generate_keys(self) -> List[Note]:
        """Generates virtual piano keys for the specified number of octaves."""
        keys = []
        num_of_octaves = self.num_octaves
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H']
        natural_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'H']
        
        # Sharp note positions relative to natural keys (between white keys)
        sharp_positions = {
            'C#': 1,  # Between C (0) and D (1)
            'D#': 2,  # Between D (1) and E (2)
            'F#': 4,  # Between F (3) and G (4)
            'G#': 5,  # Between G (4) and A (5)
            'A#': 6   # Between A (5) and H (6)
        }
        
        for octave in range(num_of_octaves):
            natural_key_index = 0
            
            for note in notes:
                if '#' in note:
                    # Sharp key positioning - positioned between natural keys
                    relative_pos = sharp_positions[note]
                    center_x = (octave * len(natural_notes) + relative_pos) * Note.width
                    # Sharp keys start from top and go down 40% of the height
                    sharp_center_y = config.WINDOW_HEIGHT * 0.2  # Center at 20% from top
                    keys.append(Note(
                        key=note,
                        octave=octave,
                        center=(center_x, sharp_center_y),
                        width=Note.width * 0.6,  # Sharp keys are narrower
                        height=config.WINDOW_HEIGHT * 0.4  # 40% of total height
                    ))
                else:
                    # Natural key positioning - each key positioned at its index * width + half width for center
                    center_x = (octave * len(natural_notes) + natural_key_index) * Note.width + Note.width / 2
                    keys.append(Note(
                        key=note,
                        octave=octave,
                        center=(center_x, config.WINDOW_HEIGHT / 2),
                        width=Note.width,
                        height=config.WINDOW_HEIGHT
                    ))
                    natural_key_index += 1
        
        return keys
    
    def update(self, fingertips: List[Fingertip]) -> None:
        """Updates the piano keys based on fingertip positions."""
        # First, clear all keys that are no longer being pressed by their activating fingertip
        for key in self.keys:
            if key.last_activation is not None:
                activating_fingertip = key.last_activation[1]
                # Check if the activating fingertip is still pressing this key
                fingertip_still_pressing = False
                for fingertip in fingertips:
                    if (fingertip.id == activating_fingertip.id):
                        is_overlapping, pitch = self._get_overlap_data(fingertip, key)
                        if is_overlapping:
                            fingertip_still_pressing = True
                            key.pitch = pitch
                        break
                
                # If the activating fingertip is no longer pressing, deactivate the key
                if not fingertip_still_pressing:
                    key.last_activation = None
        
        # check for new activations and prioritize the closest key for each fingertip
        for fingertip in fingertips:
            if not fingertip.is_pressed:
                continue
            
            # Find the closest key for this fingertip
            closest_key = None
            closest_distance = float('inf')
            closest_pitch = 1.0
            
            # First check sharp keys (priority since they're on top)
            for key in self.sharp_keys:
                if key.last_activation is None:  # Only consider keys that aren't already activated
                    is_overlapping, pitch = self._get_overlap_data(fingertip, key)
                    if is_overlapping:
                        # Calculate distance to key center
                        distance = ((fingertip.position[0] - key.center[0]) ** 2 + 
                                   (fingertip.position[1] - key.center[1]) ** 2) ** 0.5
                        
                        if distance < closest_distance:
                            closest_distance = distance
                            closest_key = key
                            closest_pitch = pitch
            
            # If no sharp key was found, check natural keys
            if closest_key is None:
                for key in self.natural_keys:
                    if key.last_activation is None:  # Only consider keys that aren't already activated
                        is_overlapping, pitch = self._get_overlap_data(fingertip, key)
                        if is_overlapping:
                            # Calculate distance to key center
                            distance = ((fingertip.position[0] - key.center[0]) ** 2 + 
                                      (fingertip.position[1] - key.center[1]) ** 2) ** 0.5
                            
                            if distance < closest_distance:
                                closest_distance = distance
                                closest_key = key
                                closest_pitch = pitch
            
            # Activate the closest key if one was found
            if closest_key is not None:
                closest_key.last_activation = (time.time(), fingertip)
                closest_key.pitch = closest_pitch

                
    def _get_overlap_data(self, fingertip: Fingertip, key: Note) -> Tuple[bool, float]:
        """Checks if a fingertip overlaps with a piano key."""
        # Calculate the distance from fingertip to key center
        dx = abs(fingertip.position[0] - key.center[0])
        dy = abs(fingertip.position[1] - key.center[1])
        
        # Check if the fingertip is within the key's boundaries
        overlap = (
            dx <= key.width // 2 and
            dy <= key.height // 2
        )

        if not overlap:
            return False, 1.0

        # Use the key's center as the reference point for pitch calculation
        relative_center = key.last_activation[1].position if key.last_activation else key.center
        now = time.time()

        y_delta = fingertip.position[1] - relative_center[1]
        x_delta = fingertip.position[0] - relative_center[0]

        pitch_direction_y = 1 if y_delta < 0 else -1
        pitch_direction_x = 1 if x_delta < 0 else -1

        y_delta = abs(y_delta)
        x_delta = abs(x_delta)

        pitch = 1.0  # Default pitch value
        
        # If it's a sharp key, decrease spatial thresholds (since they are narrower and shorter)
        if "#" in key.key:
            # Check for pitch bend
            if y_delta > self.pitch_bend_thresh * 0.6:
                # Map the y_delta to a pitch value
                pitch = self._map_delta_to_pitch(key.height, y_delta, pitch_direction_y, key)
                return True, pitch
            
            # Check for vibrato
            elif y_delta < self.vibrato_thresh_y * 0.4 and x_delta > self.vibrato_thresh_x * 0.6:
                # Get fingertip's vibrato history and add current timestamp and x-position
                hist = self.vibrato_history[fingertip.id]
                hist.append((now, fingertip.position[0]))
                crossings = []
                # Check history for direction reversals (oscillatory movements)
                for i in range(1, len(hist)):
                    t1, x1 = hist[i - 1]
                    t2, x2 = hist[i]
                    if i >= 2:
                        x0 = hist[i - 2][1]
                        # Check if direction reversed
                        if (x1 - x2) * (x0 - x1) < 0:
                            # Check if reversal is large and quick enough
                            if abs(x2 - x1) > self.vibrato_amplitude_window and (t2 - t1) <= self.vibrato_time_window:
                                crossings.append((t1, t2))

                # Apply vibrato if oscillations are detected
                if crossings:
                    pitch = self._map_delta_to_pitch(key.width, x_delta, pitch_direction_x, key)
                    return True, pitch
        else:
            if y_delta > self.pitch_bend_thresh:
                pitch = self._map_delta_to_pitch(key.height, y_delta, pitch_direction_y, key)
                return True, pitch

            elif y_delta < self.vibrato_thresh_y and x_delta > self.vibrato_thresh_x:
                hist = self.vibrato_history[fingertip.id]
                hist.append((now, fingertip.position[0]))
                crossings = []
                for i in range(1, len(hist)):
                    t1, x1 = hist[i - 1]
                    t2, x2 = hist[i]
                    if i >= 2:
                        x0 = hist[i - 2][1]
                        if (x1 - x2) * (x0 - x1) < 0:
                            if abs(x2 - x1) > self.vibrato_amplitude_window and (t2 - t1) <= self.vibrato_time_window:
                                crossings.append((t1, t2))

                if crossings:
                    pitch = self._map_delta_to_pitch(key.width, x_delta, pitch_direction_x, key)
                    return True, pitch

        return True, pitch
    

    def _map_delta_to_pitch(self, dimension, delta: int, direction: int, key: Note) -> float:
        """Maps the delta to a pitch value."""
        if direction > 0:
            return 1.0 + (delta / dimension)
        else:
            a = 1.0 - (delta / dimension)
            return a if a > 0 else 0.1        