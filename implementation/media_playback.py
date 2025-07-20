import cv2
import numpy as np
from piano import Note, Piano
import fluidsynth


class MediaPlayback:
    """Handles media playback and visualization."""

    def __init__(self, piano: Piano):
        self.piano = piano
        self.active_notes = set()
        self.fs = fluidsynth.Synth()
        self.fs.start(driver="dsound", midi_driver=None)

        sfid = self.fs.sfload("implementation/steinway_concert_piano.sf2")
        self.fs.program_select(0, sfid, 0, 0)

    def update(self, dt: float) -> None:
        """Updates the media playback state."""
        self.play_notes(self.piano.keys)

    def play_notes(self, notes: list[Note]) -> None:
        """Plays the given notes in a media player."""
        for note in notes:
            octave = note.octave + 1
            key = note.key
            midi_note = 36 + 12 * octave + {
                'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
                'E': 4, 'F': 5, 'F#': 6, 'G': 7,
                'G#': 8, 'A': 9, 'A#': 10, 'H': 11
            }[key]
            
            if note.last_activation is not None:
                if midi_note not in self.active_notes:
                    self.fs.noteon(0, midi_note, 30)
                    self.active_notes.add(midi_note)
            elif note.last_activation is None:
                if midi_note in self.active_notes:
                    self.fs.noteoff(0, midi_note)
                    self.active_notes.remove(midi_note)

    def draw_keys(self, frame: np.ndarray, notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
        """Visualizes the pressed keys in a cv2 window and returns a mask of key locations."""
        natural_keys = [note for note in notes if "#" not in note.key]
        sharp_keys = [note for note in notes if "#" in note.key]

        key_mask = np.zeros(frame.shape[:2], dtype=np.uint8)

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

            color = (0, 255, 0) if note.last_activation else (230, 230, 210)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
            cv2.rectangle(key_mask, top_left, bottom_right, 1, -1)

            border_color = (200, 200, 200)
            cv2.rectangle(frame, top_left, bottom_right, border_color, 2)

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

            color = (0, 255, 0) if note.last_activation else (20, 20, 20)
            cv2.rectangle(frame, top_left, bottom_right, color, -1)
            cv2.rectangle(key_mask, top_left, bottom_right, 1, -1)

        return cv2.flip(frame, 0), cv2.flip(key_mask, 0)
