import numpy as np
from piano import Note, Piano

# ? Might make more sense to store key states and start/stop sounds based on events?

class MediaPlayback:
    """Handles media playback and visualization."""

    def __init__(self, piano: Piano):
        self.piano = piano
        # Initialize media player, e.g., pygame, soundfile, etc.
        # self.player = MediaPlayer()  # Placeholder for actual media player initialization

    def update(self, dt: float) -> None:
        """Updates the media playback state."""
        self.play_notes(self.piano.keys)
        self.visualize_keys(self.piano.keys)

    def play_notes(self, notes: list[Note]) -> None:
        """Plays the given notes in a media player."""
        pass

    def visualize_keys(self, frame: np.ndarray, notes: list[Note]) -> None:
        """Visualizes the pressed keys in a cv2 window."""
        pass
