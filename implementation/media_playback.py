import cv2
import numpy as np
from piano import Note, Piano
import fluidsynth


class MediaPlayback:
    """Handles media playback and visualization."""

    def __init__(self, piano: Piano):
        self.piano = piano
        self.fs = fluidsynth.Synth()
        self.fs.start()

        sfid = self.fs.sfload("FluidR3_GM.sf2")
        self.fs.program_select(0, sfid, 0, 0)

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
        natural_keys = [note for note in notes if "#" not in note.key]
        sharp_keys = [note for note in notes if "#" in note.key]

        # Draw natural keys first
        for note in natural_keys:
            top_left = (
                int(note.center[0] - note.width // 2),
                int(note.center[1] - note.height // 2)
            )
            bottom_right = (
                int(note.center[0] + note.width // 2),
                int(note.center[1] + note.height // 2)
            )

            color = (0, 255, 0) if note.last_activation else (255, 255, 255)
            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

            border_color = (200, 200, 200)
            cv2.rectangle(frame, top_left, bottom_right, border_color, 1)

        # Draw sharp keys on top of natural keys
        for note in sharp_keys:
            top_left = (
                int(note.center[0] - note.width // 2),
                int(note.center[1] - note.height // 2)
            )
            bottom_right = (
                int(note.center[0] + note.width // 2),
                int(note.center[1] + note.height // 2)
            )

            color = (0, 255, 0) if note.last_activation else (0, 0, 0)
            overlay = frame.copy()
            cv2.rectangle(overlay, top_left, bottom_right, color, -1)
            frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        cv2.imshow("Piano Keys", frame)
        cv2.waitKey(1)
