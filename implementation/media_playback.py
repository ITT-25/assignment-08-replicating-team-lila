import cv2
import numpy as np
from piano import Note, Piano
import fluidsynth
import math


class MediaPlayback:
    """Handles media playback and visualization."""

    def __init__(self, piano: Piano):
        self.piano = piano
        self.active_notes = set()
        self.note_channels = {}  # maps midi_note -> channel
        self.available_channels = list(range(16))  # fluidsynth default = 16 channels


        self.fs = fluidsynth.Synth()
        self.fs.start(driver="dsound", midi_driver=None)
        sfid = self.fs.sfload("implementation/steinway_concert_piano.sf2")
        
        # Set same instrument for all channels
        for ch in range(16):
            self.fs.program_select(ch, sfid, 0, 0)
            # self.fs.cc(ch, 101, 0)
            # self.fs.cc(ch, 100, 0)
            self.fs.cc(ch, 6, 24)  # +/- 12 semitones
            # self.fs.cc(ch, 38, 0)

    def update(self, dt: float) -> None:
        """Updates the media playback state."""
        self.play_notes(self.piano.keys)
        self.apply_pitch_bend(self.piano.keys)

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
                    # Assign an available channel
                    if not self.available_channels:
                        continue  # no free channel (ignore note or handle polyphony limit)
                    channel = self.available_channels.pop(0)
                    self.note_channels[midi_note] = channel

                    self.fs.noteon(channel, midi_note, 70)
                    self.active_notes.add(midi_note)
            elif note.last_activation is None:
                if midi_note in self.active_notes:
                    channel = self.note_channels.get(midi_note, 0)
                    self.fs.noteoff(channel, midi_note)

                    # Reset pitch bend and free channel
                    self.fs.pitch_bend(channel, 8192)
                    self.available_channels.append(channel)
                    del self.note_channels[midi_note]

                    self.active_notes.remove(midi_note)

    def apply_pitch_bend(self, notes: list[Note]) -> None:
        for note in notes:
            if note.last_activation is not None:
                octave = note.octave + 1
                key = note.key
                midi_note = 36 + 12 * octave + {
                    'C': 0, 'C#': 1, 'D': 2, 'D#': 3,
                    'E': 4, 'F': 5, 'F#': 6, 'G': 7,
                    'G#': 8, 'A': 9, 'A#': 10, 'H': 11
                }[key]

                if midi_note in self.note_channels:
                    channel = self.note_channels[midi_note]
                    bend_value = self.calculate_pitch_bend(note.pitch, bend_range=12)
                    bend_value -= 8192  # Convert to relative value
                    print(bend_value)
                    self.fs.pitch_bend(channel, bend_value)

    def calculate_pitch_bend(self, pitch: float, bend_range: float = 2.0) -> int:
        """
        Converts a pitch multiplier to a MIDI pitch bend value.
        :param pitch: The pitch multiplier (1.0 = normal, >1 higher, <1 lower).
        :param bend_range: Max pitch bend range in semitones (+/-).
        """
        # print(str(pitch))
        semitone_offset = 12 * math.log2(pitch)  # Convert ratio to semitones
        bend = int(8192 + (semitone_offset / bend_range) * 8192)
        return max(0, min(16383, bend))  # Clamp to valid MIDI range

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
