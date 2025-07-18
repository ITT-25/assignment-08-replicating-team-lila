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
    width: int = 0
    height: int = 0

class Piano:
    """Maps fingertip positions to piano keys."""
    
    def __init__(self, num_octaves: int = 2, pitch_range: Tuple[float, float] = (0.5, 2.0)):
        self.num_octaves: int = num_octaves
        self.pitch_range: Tuple[float, float] = pitch_range
        self.keys: List[Note] = self._generate_keys()

    def _generate_keys(self) -> List[Note]:
        """Generates virtual piano keys for the specified number of octaves."""
        keys = []
        num_of_octaves = self.num_octaves
        # Define all notes including sharps/flats
        notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'H']
        key_width = (config.WINDOW_WIDTH / len(notes)) * (1 / num_of_octaves)
        # properly set the width, height and center of each key relative to the total width/height stored in the cfg module
        for octave in range(len(num_of_octaves)):
            for note in len(notes):
                center_x = key_width * ((octave * len(notes)) + note + 0.5)
                keys.append(Note(
                    key=notes[note],
                    octave=octave,
                    center=(center_x, config.WINDOW_HEIGHT / 2),
                    width=key_width,
                    height=config.WINDOW_HEIGHT
                ))
        return keys
    
    def update(self, fingertips: List[Fingertip]) -> None:
        """Updates the piano keys based on fingertip positions."""
        for fingertip in fingertips:
            if fingertip.is_pressed:
                for key in self.keys:
                    is_overlapping, pitch = self._get_overlap_data(fingertip, key)
                    
                    if is_overlapping:
                        key.pitch = pitch
                        
                    if key.last_activation is None and is_overlapping:
                        key.last_activation = (time.time(), fingertip)
                        
                    elif key.last_activation is not None and not is_overlapping:
                        key.last_activation = None
                
    def _get_overlap_data(self, fingertip: Fingertip, key: Note) -> Tuple[bool, float]:
        """Checks if a fingertip overlaps with a piano key."""
        overlap = (
            fingertip.position[0] >= key.center[0] - key.width // 2 and
            fingertip.position[0] <= key.center[0] + key.width // 2 and
            fingertip.position[1] >= key.center[1] - key.height // 2 and
            fingertip.position[1] <= key.center[1] + key.height // 2
        )
        
        relative_center = key.last_activation[1].position if key.last_activation else fingertip.position

        if overlap:
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
