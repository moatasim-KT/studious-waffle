import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np
from collections import defaultdict
import sounddevice as sd
import soundfile as sf
import argparse
import os


class PianoVisualizer:
    def __init__(self, width=1600, height=500, mode="playback"):
        pygame.init()
        pygame.midi.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MIDI to Piano Keystrokes Visualizer")
        self.mode = mode

        self.white_key_width = 41
        self.white_key_height = 220
        self.black_key_width = 30
        self.black_key_height = 130

        self.piano_start_x = 20
        self.piano_start_y = 200

        self.active_notes = defaultdict(lambda: False)
        self.note_colors = defaultdict(lambda: (180, 180, 255))

        self.first_note = 24
        self.total_keys = 60

        self.font = pygame.font.SysFont("Arial", 24)
        self.key_font = pygame.font.SysFont("Arial", 16)

        self.note_names = [
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
            "A",
            "A#",
            "B",
        ]

        self.playing = False
        self.paused = False
        self.current_time = 0
        self.events_iterator = None
        self.tempo = 500000
        self.midi_events = []
        self.current_event_idx = 0

        self.highlighting_mode = False
        self.chord_sequence = []
        self.current_chord_idx = 0
        self.highlighted_notes = defaultdict(lambda: False)

        self.sounds = {}
        self.volume = 0.5
        self.sample_rate = 44100
        self.piano_samples = {}
        self.load_piano_samples()

        self.midi_input = None
        self.setup_midi_input()

    def play(self):
        self.playing = True
        self.paused = False

    def pause(self):
        self.paused = not self.paused
        if self.paused:
            pygame.mixer.stop()

    def stop(self):
        self.playing = False
        self.paused = True
        self.current_time = 0
        self.current_event_idx = 0
        self.active_notes.clear()
        self.note_colors.clear()

    def setup_midi_input(self):
        if not pygame.midi.get_init():
            pygame.midi.init()

        input_devices = [
            (i, info[1].decode("utf-8"))
            for i in range(pygame.midi.get_count())
            if (info := pygame.midi.get_device_info(i))[2]
        ]

        if input_devices:
            for device_id, name in input_devices:
                try:
                    self.midi_input = pygame.midi.Input(device_id)
                    print(f"Connected to MIDI device: {name}")
                    break
                except pygame.midi.MidiException as e:
                    print(f"Could not open MIDI device {device_id} ({name}): {e}")
            if not self.midi_input:
                print(
                    "No MIDI input devices could be opened. Using computer keyboard only."
                )
        else:
            print("No MIDI input devices found. Using computer keyboard only.")

    def load_piano_samples(self):
        try:
            samples_dir = "samples"
            os.makedirs(samples_dir, exist_ok=True)

            for note in range(self.first_note, self.first_note + self.total_keys):
                note_name = self.get_note_name(note)
                file_path = os.path.join(samples_dir, f"piano_{note_name}.wav")

                if os.path.isfile(file_path):
                    data, fs = sf.read(file_path)
                    if fs != self.sample_rate:
                        ratio = self.sample_rate / fs
                        data = np.interp(
                            np.linspace(0, len(data), int(len(data) * ratio)),
                            np.arange(len(data)),
                            data,
                        )
                    self.piano_samples[note] = data
                    print(f"Loaded sample for {note_name} ({note})")
                else:
                    print(f"Sample file not found: {file_path}")
        except Exception as e:
            print(f"Error loading piano samples: {e}")

    def generate_fallback_sound(self, note):
        """Generate a sine wave as a fallback sound for missing samples."""
        # Convert MIDI note to frequency (A440 tuning)
        frequency = 440.0 * 2 ** ((note - 69) / 12.0)
        duration = 1.0  # 1 second duration
        t = np.linspace(0, duration, int(self.sample_rate * duration), endpoint=False)
        # Generate sine wave
        return 0.5 * np.sin(2 * np.pi * frequency * t)

    def play_note(self, note, velocity=127):
        if note in self.piano_samples:
            try:
                if self.piano_samples[note] is None:
                    # Fallback to sine wave if sample is missing
                    print(f"Playing fallback sound for note {note}")
                    samples = self.generate_fallback_sound(note)
                else:
                    volume = (velocity / 127.0) * self.volume
                    samples = self.piano_samples[note] * volume
                sd.play(samples, self.sample_rate, blocking=False) #<-- Changed here
            except Exception as e:
                print(f"Error playing note {note}: {e}")
        else:
            print(f"No sample found for note {note}, playing fallback sound.")
            samples = self.generate_fallback_sound(note)
            sd.play(samples, self.sample_rate, blocking=False) #<-- Changed here

    def stop_note(self, note):
        """Stop a note from playing."""
        sd.stop() #<-- Changed here

    def is_black_key(self, note):
        return (note % 12) in [1, 3, 6, 8, 10]

    def get_note_name(self, note):
        note_name = self.note_names[note % 12]
        octave = (note - 12) // 12
        return f"{note_name}{octave}"

    def get_key_position(self, note):
        relative_note = note - self.first_note
        if relative_note < 0 or relative_note >= self.total_keys:
            return None

        white_key_count = sum(
            not self.is_black_key(n) for n in range(self.first_note, note)
        )

        if not self.is_black_key(note):
            return self.piano_start_x + (white_key_count * self.white_key_width)

        prev_white = note - 1
        while self.is_black_key(prev_white) and prev_white >= self.first_note:
            prev_white -= 1

        if prev_white < self.first_note:
            return None

        white_key_count_prev = len(
            [
                n
                for n in range(self.first_note, prev_white + 1)
                if not self.is_black_key(n)
            ]
        )

        return (
            self.piano_start_x
            + ((white_key_count_prev - 1) * self.white_key_width)
            + (self.white_key_width - self.black_key_width // 2)
        )

    def draw_piano(self):
        for note in range(self.first_note, self.first_note + self.total_keys):
            if not self.is_black_key(note):
                x = self.get_key_position(note)
                if x is not None:
                    color = (
                        (150, 150, 255) if self.active_notes[note] else (255, 255, 255)
                    )
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (
                            x,
                            self.piano_start_y,
                            self.white_key_width - 1,
                            self.white_key_height,
                        ),
                    )
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 0),
                        (
                            x,
                            self.piano_start_y,
                            self.white_key_width - 1,
                            self.white_key_height,
                        ),
                        1,
                    )
                    note_name = self.get_note_name(note)
                    name_text = self.key_font.render(note_name, True, (0, 0, 0))
                    text_x = x + (self.white_key_width - name_text.get_width()) // 2
                    text_y = (
                        self.piano_start_y
                        + self.white_key_height
                        - name_text.get_height()
                        - 5
                    )
                    self.screen.blit(name_text, (text_x, text_y))

        for note in range(self.first_note, self.first_note + self.total_keys):
            if self.is_black_key(note):
                x = self.get_key_position(note)
                if x is not None:
                    color = (100, 100, 255) if self.active_notes[note] else (0, 0, 0)
                    pygame.draw.rect(
                        self.screen,
                        color,
                        (
                            x,
                            self.piano_start_y,
                            self.black_key_width,
                            self.black_key_height,
                        ),
                    )

                    note_name = self.get_note_name(note)
                    name_text = self.key_font.render(note_name, True, (255, 255, 255))
                    text_x = x + (self.black_key_width - name_text.get_width()) // 2
                    text_y = (
                        self.piano_start_y
                        + self.black_key_height
                        - name_text.get_height()
                        - 5
                    )
                    self.screen.blit(name_text, (text_x, text_y))

    def load_midi(self, file_path):
        try:
            midi_file = mido.MidiFile(file_path)
            self.midi_events = []
            self.chord_sequence = []
            self.current_chord_idx = 0
            ticks_per_beat = midi_file.ticks_per_beat
            tempo = 500000
            events = []

            for track in midi_file.tracks:
                absolute_time = 0
                current_notes = set()
                for msg in track:
                    delta_ticks = msg.time
                    delta_seconds = mido.tick2second(delta_ticks, ticks_per_beat, tempo)
                    absolute_time += delta_seconds * 1000  # Convert to milliseconds

                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                        self.midi_events.append((absolute_time, "tempo", msg.tempo))
                    elif msg.type == "note_on" and msg.velocity > 0:
                        self.midi_events.append(
                            (absolute_time, "note_on", msg.note, msg.velocity)
                        )
                        current_notes.add(msg.note)
                        events.append((absolute_time, "chord", set(current_notes)))
                    elif msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                    ):
                        self.midi_events.append((absolute_time, "note_off", msg.note))
                        if msg.note in current_notes:
                            current_notes.remove(msg.note)

            self.midi_events.sort(key=lambda x: x[0])

            events.sort(key=lambda x: x[0])
            unique_chords = []
            prev_chord = set()

            for time, event_type, chord in events:
                if chord != prev_chord and len(chord) > 0:
                    unique_chords.append(chord)
                    prev_chord = chord

            self.chord_sequence = unique_chords

            print(
                f"Loaded {len(self.chord_sequence)} unique chords/notes from MIDI file"
            )
            return True
        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return False

    def update(self, dt):
        if not self.playing or self.paused or not self.midi_events:
            return

        self.current_time += dt * 1000
        while (
            self.current_event_idx < len(self.midi_events)
            and self.midi_events[self.current_event_idx][0] <= self.current_time
        ):
            event = self.midi_events[self.current_event_idx]

            if event[1] == "note_on":
                note = event[2]
                velocity = event[3]
                brightness = int(velocity * 2) + 50
                self.active_notes[note] = True
                self.note_colors[note] = (200, 200, brightness)
                self.play_note(note, velocity)

            elif event[1] == "note_off":
                note = event[2]
                self.active_notes[note] = False
                self.stop_note(note)

            elif event[1] == "tempo":
                self.tempo = event[2]

            self.current_event_idx += 1

            if self.current_event_idx >= len(self.midi_events):
                self.playing = False
                break

    def render_ui(self):
        status = (
            "Playing"
            if self.playing and not self.paused
            else "Paused" if self.paused else "Stopped"
        )
        status_text = self.font.render(f"Status: {status}", True, (255, 255, 255))
        self.screen.blit(status_text, (10, 10))

        time_text = self.font.render(
            f"Time: {self.current_time/1000:.2f}s", True, (255, 255, 255)
        )
        self.screen.blit(time_text, (10, 40))

        controls_text = self.font.render(
            "Controls: SPACE = Play/Pause, R = Reset, ESC = Quit, +/- = Volume",
            True,
            (255, 255, 255),
        )
        self.screen.blit(controls_text, (10, 70))

        volume_text = self.font.render(
            f"Volume: {int(self.volume * 100)}%", True, (255, 255, 255)
        )
        self.screen.blit(volume_text, (10, 100))

        range_text = self.font.render(
            f"Key Range: {self.get_note_name(self.first_note)} to {self.get_note_name(self.first_note + self.total_keys - 1)}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(range_text, (10, 130))

        if self.highlighting_mode and self.chord_sequence:
            progress_text = self.font.render(
                f"Chord: {self.current_chord_idx}/{len(self.chord_sequence)}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(progress_text, (10, 190))

        if self.highlighting_mode and self.highlighted_notes:
            highlighted = [
                self.get_note_name(note)
                for note in self.highlighted_notes
                if self.highlighted_notes[note]
            ]
            notes_str = "Highlighted: " + ", ".join(highlighted)
            notes_text = self.font.render(notes_str, True, (255, 255, 100))
            self.screen.blit(notes_text, (10, 220))

        active_notes = [
            name
            for note, is_active in self.active_notes.items()
            if is_active and (name := self.get_note_name(note))
        ]
        if active_notes:
            notes_str = "Playing: " + ", ".join(active_notes[:8])
            if len(active_notes) > 8:
                notes_str += f" +{len(active_notes) - 8} more"
            notes_text = self.font.render(notes_str, True, (255, 255, 255))
            self.screen.blit(notes_text, (10, 160))

    def run(self, midi_file):
        if not self.load_midi(midi_file):
            print(f"Failed to load MIDI file: {midi_file}")
            return

        self.play()
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE,):
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not self.playing:
                            self.play()
                        else:
                            self.pause()
                    elif event.key == pygame.K_r:
                        self.stop()
                    elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                        self.volume = min(1.0, self.volume + 0.05)
                    elif event.key == pygame.K_MINUS:
                        self.volume = max(0.0, self.volume - 0.05)

            dt = clock.tick(60) / 1000.0
            self.update(dt)
            self.screen.fill((40, 40, 40))
            self.draw_piano()
            self.render_ui()
            pygame.display.flip()

        pygame.quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI Piano Visualizer")
    parser.add_argument("midi_file", help="Path to the MIDI file")
    parser.add_argument(
        "--mode",
        choices=["playback", "play-along"],
        default="playback",
        help="Mode: playback or play-along",
    )
    args = parser.parse_args()

    visualizer = PianoVisualizer(mode=args.mode)
    visualizer.run(args.midi_file)
