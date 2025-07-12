import numpy as np
from piano import Note

# ? Might make more sense to store key states and start/stop sounds based on events?

class MediaPlayback:
    """Handles media playback and visualization."""

    def play_notes(self, notes: list[Note]) -> None:
        """Plays the given notes in a media player."""
        pass

    def visualize_keys(self, frame: np.ndarray, notes: list[Note]) -> None:
        """Visualizes the pressed keys in a cv2 window."""
        pass
