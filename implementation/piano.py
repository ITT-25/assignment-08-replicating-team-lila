import time
from typing import Optional, Tuple, List
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
    
    def __init__(self, num_octaves: int = 2, pitch_range: Tuple[float, float] = (0.5, 2.0)):
        self.num_octaves: int = num_octaves
        self.pitch_range: Tuple[float, float] = pitch_range
        config.WINDOW_WIDTH = Note.width * self.num_octaves * 7
        config.WINDOW_HEIGHT = int(config.WINDOW_WIDTH // 1.77)
        self.keys: List[Note] = self._generate_keys()

    def _generate_keys(self) -> List[Note]:
        """Generates virtual piano keys for the specified number of octaves."""
        keys = []
        num_of_octaves = self.num_octaves
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H']
        natural_notes = ['C', 'D', 'E', 'F', 'G', 'A', 'H']
        
        # Sharp note positions relative to natural keys (between white keys)
        sharp_positions = {
            'C#': 0.5,  # Between C (0) and D (1)
            'D#': 1.5,  # Between D (1) and E (2)
            'F#': 3.5,  # Between F (3) and G (4)
            'G#': 4.5,  # Between G (4) and A (5)
            'A#': 5.5   # Between A (5) and H (6)
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
                    if (fingertip.id == activating_fingertip.id and 
                        fingertip.is_pressed):
                        is_overlapping, pitch = self._get_overlap_data(fingertip, key)
                        if is_overlapping:
                            fingertip_still_pressing = True
                            key.pitch = pitch
                            break
                
                # If the activating fingertip is no longer pressing, deactivate the key
                if not fingertip_still_pressing:
                    key.last_activation = None
        
        # Then, check for new activations
        for fingertip in fingertips:
            if fingertip.is_pressed:
                for key in self.keys:
                    is_overlapping, pitch = self._get_overlap_data(fingertip, key)
                    
                    if is_overlapping and key.last_activation is None:
                        key.last_activation = (time.time(), fingertip)
                        key.pitch = pitch
                
    def _get_overlap_data(self, fingertip: Fingertip, key: Note) -> Tuple[bool, float]:
        """Checks if a fingertip overlaps with a piano key."""
        overlap = (
            fingertip.position[0] >= key.center[0] - key.width // 2 and
            fingertip.position[0] <= key.center[0] + key.width // 2 and
            fingertip.position[1] >= key.center[1] - key.height // 2 and
            fingertip.position[1] <= key.center[1] + key.height // 2
        )

        if overlap:
            # Use the key's center as the reference point for pitch calculation
            relative_center = key.last_activation[1].position if key.last_activation else key.center
            y_delta = fingertip.position[1] - relative_center[1]
            pitch_direction = 1 if y_delta < 0 else -1
            y_delta = abs(y_delta)

            # Map the y_delta to a pitch value
            pitch = self._map_y_delta_to_pitch(y_delta, pitch_direction)
            return True, pitch

        return False, 1.0

    def _map_y_delta_to_pitch(self, y_delta: int, direction: int) -> float:
        """Maps the y_delta to a pitch value."""
        # Simple mapping: larger y_delta results in higher pitch
        return 1.0 + (y_delta / 100.0) * direction
