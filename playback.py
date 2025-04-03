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
import threading
import queue
import contextlib
from scipy import signal


class PianoVisualizer:
    def __init__(self, width=1600, height=600, midi_file=None):
        pygame.init()
        pygame.midi.init()
        pygame.display.set_caption("MIDI Piano Visualizer")

        # Screen setup
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
        self.midi_file = midi_file

        # Piano key dimensions
        self.white_key_width = 40
        self.white_key_height = 220
        self.black_key_width = 28
        self.black_key_height = 140

        # Piano position
        self.piano_start_x = 50
        self.piano_start_y = 280

        # Note tracking
        self.active_notes = defaultdict(lambda: False)
        self.note_colors = defaultdict(lambda: (180, 180, 255))

        # Piano key range
        self.first_note = 21  # A0
        self.total_keys = 88

        # Fonts
        self.title_font = pygame.font.SysFont("Arial", 36, bold=True)
        self.font = pygame.font.SysFont("Arial", 24)
        self.key_font = pygame.font.SysFont("Arial", 16)

        # Note naming
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

        # Playback and timing
        self.playing = False
        self.paused = False
        self.current_time = 0
        self.tempo = 500000
        self.midi_events = []
        self.current_event_idx = 0

        # Audio playback
        self.volume = 0.7
        self.sample_rate = 44100
        self.piano_samples = {}
        self.active_sounds = {}
        self.sound_queue = queue.Queue()
        self.samples_loaded = False  # Flag to track sample loading status

        # UI elements
        self.show_note_names = True

        # Settings
        self.reverb_amount = 0.2

        # Status message for loading
        self.status_message = "Loading samples..."

        # Load samples asynchronously
        self.sample_loader_thread = threading.Thread(
            target=self.load_piano_samples_async, daemon=True
        )
        self.sample_loader_thread.start()

        # Audio engine setup
        self.stream = None
        self.audio_lock = threading.Lock()
        self.audio_thread = threading.Thread(
            target=self.audio_processing_thread, daemon=True
        )
        self.audio_thread.start()

        # Load the MIDI file if provided
        if self.midi_file:
            self.load_midi_file(self.midi_file)

        # Start the visualizer
        self.run()

    def get_note_name(self, note):
        """Get the name of a note including octave"""
        note_name = self.note_names[note % 12]
        octave = (note - 12) // 12
        return f"{note_name}{octave}"

    def load_midi_file(self, file_path):
        """Load and parse MIDI file"""
        try:
            self.status_message = f"Loading MIDI file: {file_path}..."
            midi_file = mido.MidiFile(file_path)
            self.midi_events = []  # Clear existing events

            # Parse events from MIDI file
            for track in midi_file.tracks:
                absolute_time = 0
                for msg in track:
                    absolute_time += msg.time
                    # Extract tempo if present
                    if msg.type == "set_tempo":
                        self.tempo = msg.tempo
                    if msg.type == "note_on" and msg.velocity > 0:
                        self.midi_events.append(
                            (absolute_time, "note_on", msg.note, msg.velocity)
                        )
                    elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                        self.midi_events.append((absolute_time, "note_off", msg.note))
            # Sort events by time
            self.midi_events.sort(key=lambda x: x[0])
            self.status_message = f"MIDI file loaded: {file_path}"
            print(f"MIDI file loaded: {file_path} with {len(self.midi_events)} events")
        except Exception as e:
            self.status_message = f"Error loading MIDI file: {e}"
            print(f"Error loading MIDI file {file_path}: {e}")

    def load_piano_samples_async(self):
        """Load piano samples asynchronously to avoid blocking the main thread"""
        try:
            samples_dir = "samples"

            # Check if samples directory exists
            if not os.path.exists(samples_dir):
                print(f"Samples directory not found. Creating directory: {samples_dir}")
                os.makedirs(samples_dir, exist_ok=True)
                return self._extracted_from_load_piano_samples_async_10()
            available_samples = [
                f
                for f in os.listdir(samples_dir)
                if f.startswith("piano_") and f.endswith(".wav")
            ]
            print(f"Found {len(available_samples)} piano samples")

            if not available_samples:
                print("No piano samples found. Using synthesized sounds.")
                return self._extracted_from_load_piano_samples_async_10()
            for note in range(self.first_note, self.first_note + self.total_keys):
                note_name = self.get_note_name(note)
                file_path = os.path.join(samples_dir, f"piano_{note_name}.wav")

                self.status_message = f"Loading sample: {note_name}"

                if os.path.isfile(file_path):
                    try:
                        data, fs = sf.read(file_path, dtype="float32")

                        if fs != self.sample_rate:
                            ratio = self.sample_rate / fs
                            data = signal.resample(data, int(len(data) * ratio))

                        if len(data.shape) == 1:
                            data = np.column_stack((data, data))
                        elif data.shape[1] == 1:
                            data = np.column_stack((data[:, 0], data[:, 0]))
                        elif data.shape[1] > 2:
                            data = data[:, :2]

                        self.piano_samples[note] = data
                        print(f"Loaded sample for {note_name} ({note})")
                    except Exception as e:
                        print(f"Error loading sample {file_path}: {e}")
                        # Use None to indicate missing sample
                        self.piano_samples[note] = None
                else:
                    # Use None to indicate missing sample
                    self.piano_samples[note] = None

            # If no valid samples were loaded, log a message
            if all(value is None for value in self.piano_samples.values()):
                print("No valid piano samples found. Using synthesized sounds.")
                self.status_message = "No valid samples. Using synthesized sounds."

        except Exception as e:
            print(f"Error loading piano samples: {e}")
            self.status_message = f"Error loading samples: {e}"

        # Mark samples as loaded regardless of success or failure
        self.samples_loaded = True
        self.status_message = "Ready to play!"

    # TODO Rename this here and in `load_piano_samples_async`
    def _extracted_from_load_piano_samples_async_10(self):
        self.samples_loaded = True  # Mark as loaded even without samples
        self.status_message = "No samples found. Using synthesized sounds."
        return

    def update_statistics(self):
        """Update note statistics"""
        self.current_polyphony = len(self.active_sounds)

    def play_note(self, note, velocity=127):
        """Play a note with better sound management"""
        # Activate the visual note
        self.active_notes[note] = True

        self.update_statistics()

        try:
            # Prepare the audio data
            if note in self.piano_samples and self.piano_samples[note] is not None:
                # Use recorded sample
                volume = (velocity / 127.0) ** 1.2 * self.volume
                samples = self.piano_samples[note] * volume
            else:
                # Use synthesized sound
                samples = self.generate_improved_sound(note, velocity)

            # Ensure correct data type
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)

            # Add to sound queue for playback
            self.sound_queue.put((note, samples))
        except Exception as e:
            print(f"Error playing note {note}: {e}")

    def generate_improved_sound(self, note, velocity):
        """Generate a sine wave sound for a given note"""
        frequency = 440.0 * (2.0 ** ((note - 69) / 12.0))
        duration = 1.0
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)

        # Use a mix of sine waves for richer sound
        sine1 = np.sin(2 * np.pi * frequency * t)
        sine2 = np.sin(2 * np.pi * frequency * 2 * t) * 0.3  # First harmonic
        sine3 = np.sin(2 * np.pi * frequency * 3 * t) * 0.15  # Second harmonic

        sine_wave = sine1 + sine2 + sine3
        sine_wave = sine_wave / np.max(np.abs(sine_wave))  # Normalize

        # Apply envelope
        attack = int(0.01 * self.sample_rate)
        decay = int(0.1 * self.sample_rate)
        release = int(0.5 * self.sample_rate)

        envelope = np.ones_like(sine_wave)
        envelope[:attack] = np.linspace(0, 1, attack)
        envelope[attack : attack + decay] = np.linspace(1, 0.7, decay)
        envelope[-release:] = np.linspace(0.7, 0, release)

        sine_wave *= envelope
        sine_wave *= (velocity / 127.0) ** 1.2 * self.volume
        return np.column_stack((sine_wave, sine_wave))

    def audio_processing_thread(self):
        """Background thread for audio processing"""
        # Wait for samples to be loaded before starting the audio stream
        max_wait_time = 30  # Maximum time to wait in seconds
        wait_start = time.time()

        while not self.samples_loaded:
            time.sleep(0.1)
            if time.time() - wait_start > max_wait_time:
                print("Timeout waiting for samples. Starting audio stream anyway.")
                break

        # Start the audio stream
        try:
            self.stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype="float32",
                callback=self.audio_callback,
            )
            self.stream.start()
            print("Audio stream started successfully")
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.status_message = f"Audio error: {e}"

        # Keep the thread alive
        while True:
            time.sleep(0.1)

    def audio_callback(self, outdata, frames, time, status):
        """Real-time audio callback"""
        if status:
            print(f"Audio callback status: {status}")

        output = np.zeros((frames, 2), dtype=np.float32)

        with self.audio_lock:
            self._process_active_sounds(output, frames)
            self._process_queued_sounds(output)
            self._apply_effects(output, frames)

        outdata[:] = output

    def _process_active_sounds(self, output, frames):
        notes_to_remove = []
        for note, (data, position) in list(self.active_sounds.items()):
            if position >= len(data):
                notes_to_remove.append(note)
                continue

            to_copy = min(frames, len(data) - position)
            output[:to_copy] += data[position : position + to_copy]

            new_position = position + to_copy
            if new_position >= len(data):
                notes_to_remove.append(note)
            else:
                self.active_sounds[note] = (data, new_position)

        for note in notes_to_remove:
            del self.active_sounds[note]
            # Also deactivate visual note when sound is finished
            self.active_notes[note] = False

    def _process_queued_sounds(self, output):
        try:
            while not self.sound_queue.empty():
                note, data = self.sound_queue.get_nowait()
                if note in self.active_sounds:
                    self._crossfade_sounds(note, data)
                self.active_sounds[note] = (data, 0)
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Error processing sound queue: {e}")

    def _crossfade_sounds(self, note, new_data):
        existing_data, position = self.active_sounds[note]
        remaining = len(existing_data) - position
        if remaining > 0:
            crossfade_len = min(1000, remaining)
            fade_in = np.linspace(0, 1, crossfade_len)
            fade_out = np.linspace(1, 0, crossfade_len)
            new_data[:crossfade_len] = new_data[:crossfade_len] * fade_in.reshape(
                -1, 1
            ) + existing_data[position : position + crossfade_len] * fade_out.reshape(
                -1, 1
            )

    def _apply_effects(self, output, frames):
        if self.reverb_amount > 0:
            reverb_tail = int(self.sample_rate * 0.3)
            decay = np.exp(-np.linspace(0, 5, reverb_tail))
            reverb = np.zeros((frames + reverb_tail, 2), dtype=np.float32)
            reverb[:frames] = output * self.reverb_amount
            for i in range(min(frames, reverb_tail)):
                output[i] += reverb[i] * decay[i]

        # Final volume adjustment and clipping
        output *= self.volume
        np.clip(output, -0.99, 0.99, out=output)

    def run(self):
        """Start the visualizer and handle events"""
        clock = pygame.time.Clock()

        running = True
        while running:
            dt = clock.tick(60) / 1000.0  # dt in seconds
            for event in pygame.event.get():
                running = self._handle_input(event)
                if not running:
                    break

                if event.type == pygame.VIDEORESIZE:
                    self.width, self.height = event.size
                    self.screen = pygame.display.set_mode((self.width, self.height), pygame.RESIZABLE)

            self._handle_midi_playback(dt)
            self.update()

        self._cleanup()


    def _handle_input(self, event):
        if event.type == pygame.QUIT:
            return False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                return False
            elif event.key == pygame.K_SPACE:
                self.paused = not self.paused
                self.status_message = "Paused" if self.paused else "Playing"
            elif event.key == pygame.K_n:
                self.show_note_names = not self.show_note_names
            # Test playing notes with keyboard (A-K for C major scale)
            elif event.key in range(pygame.K_a, pygame.K_l):  # Check for keys A-K
                note = 60 + (event.key - pygame.K_a)
                if note <= 72:  # Limit to C8
                    self.play_note(note, 100)
        return True

    def _handle_midi_playback(self, dt):
        if not self.midi_file or self.paused or not self.midi_events:
            return
        while (self.current_event_idx < len(self.midi_events) and
               self.midi_events[self.current_event_idx][0] <= self.current_time):
            _, event_type, note, *rest = self.midi_events[self.current_event_idx]

            if event_type == "note_on":
                velocity = rest[0] if rest else 0
                if velocity > 0:
                    self.play_note(note, velocity)
                else:  # Velocity 0 is equivalent to note_off
                    self.stop_note(note)
            elif event_type == "note_off":
                self.stop_note(note)

            self.current_event_idx += 1

        self.current_time += dt

    def _cleanup(self):
        if self.stream:
            with contextlib.suppress(Exception):  # Suppress potential exceptions during cleanup
                self.stream.stop()
                self.stream.close()
        pygame.quit()

    def stop_note(self, note):
        """Stop a note from playing"""
        self.active_notes[note] = False
        # We'll let the sound fade out naturally in the audio processing

    def update(self):
        """Update the game state and redraw the screen"""
        self.screen.fill((40, 40, 40))

        # Draw status information
        self.draw_status()

        # Draw piano keys
        self.draw_piano()

        # Update the display
        pygame.display.flip()

    def draw_status(self):
        """Draw status information"""
        # Draw title
        title_text = self.title_font.render(
            "MIDI Piano Visualizer", True, (255, 255, 255)
        )
        self.screen.blit(
            title_text, (self.width // 2 - title_text.get_width() // 2, 10)
        )

        # Draw status message
        status_text = self.font.render(self.status_message, True, (200, 200, 200))
        self.screen.blit(
            status_text, (self.width // 2 - status_text.get_width() // 2, 60)
        )

        # Draw controls
        controls_text = self.font.render(
            "Controls: Space = Pause/Play, N = Toggle Note Names, ESC = Exit",
            True,
            (200, 200, 200),
        )
        self.screen.blit(
            controls_text, (self.width // 2 - controls_text.get_width() // 2, 100)
        )

        # If MIDI file is loaded, show playback position
        if self.midi_file and len(self.midi_events) > 0:
            if self.current_event_idx < len(self.midi_events):
                progress = self.current_event_idx / len(self.midi_events) * 100
            else:
                progress = 100

            progress_text = self.font.render(
                f"Progress: {progress:.1f}% (Event {self.current_event_idx}/{len(self.midi_events)})",
                True,
                (200, 200, 200),
            )
            self.screen.blit(
                progress_text, (self.width // 2 - progress_text.get_width() // 2, 140)
            )

        # Also draw a test piano message
        test_text = self.font.render(
            "Test piano: A-K keys to play C major scale", True, (180, 180, 180)
        )
        self.screen.blit(test_text, (self.width // 2 - test_text.get_width() // 2, 180))

    def is_black_key(self, note):
        """Check if a MIDI note number is a black key"""
        return (note % 12) in [1, 3, 6, 8, 10]

    def get_key_position(self, note):
        """Get the x position of a key"""
        # Calculate relative position from first note
        note_index = note - self.first_note
        if note_index < 0 or note_index >= self.total_keys:
            return None

        # Count white keys before this note
        white_key_count = len([n for n in range(self.first_note, note) if not self.is_black_key(n)])


        if self.is_black_key(note):
            return self._extracted_from_get_key_position_14(note)
        else:
            # White key position
            return self.piano_start_x + white_key_count * self.white_key_width

    # TODO Rename this here and in `get_key_position`
    def _extracted_from_get_key_position_14(self, note):
        # For black keys, position is based on the previous white key
        prev_white = note - 1
        while self.is_black_key(prev_white) and prev_white >= self.first_note:
            prev_white -= 1

        # Calculate position of previous white key
        prev_white_index = prev_white - self.first_note
        prev_white_count = prev_white - self.first_note + 1 - sum(bool(self.is_black_key(n))
                                                              for n in range(self.first_note, prev_white + 1))

        # Black key position is offset from white key
        x = self.piano_start_x + (prev_white_count - 1) * self.white_key_width
        return x + self.white_key_width - self.black_key_width // 2

    def draw_piano(self):
        """Draw the piano keyboard"""
        # First draw all white keys
        for note in range(self.first_note, self.first_note + self.total_keys):
            if not self.is_black_key(note):
                x = self.get_key_position(note)
                if x is not None:
                    self._draw_key(
                        note,
                        x,
                        self.white_key_width - 1,
                        self.white_key_height,
                        (150, 150, 255),
                        (0, 0, 0),
                    )

        # Then draw all black keys (so they appear on top)
        for note in range(self.first_note, self.first_note + self.total_keys):
            if self.is_black_key(note):
                x = self.get_key_position(note)
                if x is not None:
                    self._draw_key(
                        note,
                        x,
                        self.black_key_width,
                        self.black_key_height,
                        (100, 100, 255),
                        (255, 255, 255),
                    )

    def _draw_key(self, note, x, width, height, active_color, text_color):
        """Draws a single key (black or white)"""
        is_white_key = height == self.white_key_height

        # Determine key color based on active state
        if self.active_notes[note]:
            color = active_color
        else:
            color = (255, 255, 255) if is_white_key else (0, 0, 0)

        # Draw key rectangle
        pygame.draw.rect(self.screen, color, (x, self.piano_start_y, width, height))

        # Draw outline
        if is_white_key:
            pygame.draw.rect(
                self.screen,
                (0, 0, 0),
                (x, self.piano_start_y, width, height),
                1,
            )

        # Draw note name if enabled
        if self.show_note_names:
            note_name = self.get_note_name(note)
            name_text = self.key_font.render(note_name, True, text_color)
            text_x = x + (width - name_text.get_width()) // 2
            text_y = self.piano_start_y + height - name_text.get_height() - 5
            self.screen.blit(name_text, (text_x, text_y))


# Main entry point
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="MIDI Piano Visualizer")
    parser.add_argument(
        "--midi_file", type=str, default=None, help="Path to MIDI file for playback"
    )
    args = parser.parse_args()

    # Initialize the visualizer
    visualizer = PianoVisualizer(midi_file=args.midi_file)
