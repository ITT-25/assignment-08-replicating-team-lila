import cv2
import numpy as np
from piano import Note, Piano
import fluidsynth


# ? Might make more sense to store key states and start/stop sounds based on events?

class MediaPlayback:
    """Handles media playback and visualization."""

    def __init__(self, piano: Piano):
        self.piano = piano
        self.fs = fluidsynth.Synth()
        self.fs.start()

        sfid = self.fs.sfload("FluidR3_GM.sf2")
        self.fs.program_select(0, sfid, 0, 0)

        # Initialize media player, e.g., pygame, soundfile, etc.
        # self.player = MediaPlayer()  # Placeholder for actual media player initialization

    def update(self, dt: float, frame: np.ndarray) -> None:
        """Updates the media playback state."""
        self.play_notes(self.piano.keys)
        self.visualize_keys(frame, self.piano.keys)

    def play_notes(self, notes: list[Note]) -> None:
        """Plays the given notes in a media player."""
        for note in notes:
            octave = note.octave + 1
            key = note.key
            midi_note = 12 * octave + {
                'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
                'E': 4, 'F': 5, 'F#': 6, 'G': 7,
                'G#': 8, 'A': 9, 'A#': 10, 'H': 11
            }[key]
            
            if note.last_activation is not None:
                self.fs.noteon(1, midi_note, 30)
            elif note.last_activation is None:
                self.fs.noteoff(1, midi_note)

    def visualize_keys(self, frame: np.ndarray, notes: list[Note]) -> None:
        """Visualizes the pressed keys in a cv2 window."""
        for note in notes:
            is_sharp = "#" in note.key
            color = (0, 0, 255, 170) if is_sharp else (255, 255, 255, 170)
            color = color if not note.last_activation else (0, 255, 0, 170)

            cv2.rectangle(
                frame,
                (int(note.center[0] - note.width // 2), int(note.center[1] - note.height // 2)),
                (int(note.center[0] + note.width // 2), int(note.center[1] + note.height // 2)),
                color
            )
        cv2.imshow("Piano Keys", frame)
        cv2.waitKey(1)
