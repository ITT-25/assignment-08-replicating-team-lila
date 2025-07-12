import numpy as np
from dataclasses import dataclass

@dataclass
class Note:
    """Represents a musical note with pitch and name."""
    pitch: int
    name: str

class Piano:
    """Maps fingertip positions to piano keys."""

    def map_to_keys(self, fingertips: np.ndarray) -> list[Note]:
        """Matches fingertip positions with virtual piano keys and returns pressed notes."""
        pass
