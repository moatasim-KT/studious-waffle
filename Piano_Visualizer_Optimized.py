import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np
from collections import defaultdict


class PianoVisualizer:
    def __init__(self, width=1430, height=500):
        pygame.init()
        pygame.midi.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MIDI to Piano Keystrokes Visualizer")

        # Piano dimensions - Significantly increased key sizes
        self.white_key_width = 30  # Increased from 30
        self.white_key_height = 220  # Increased from 200
        self.black_key_width = 20  # Increased from 20
        self.black_key_height = 130  # Increased from 120

        # Starting position for piano
        self.piano_start_x = 20
        self.piano_start_y = 200

        # Note data
        self.active_notes = defaultdict(lambda: False)
        self.note_colors = defaultdict(lambda: (180, 180, 255))

        # Reduced number of keys (from 88 to 36 keys - 3 octaves)
        self.total_keys = 88  # Reduced from 88
        self.first_note = 21  # MIDI note number for C3 (instead of A0 which is 21)

        # Increased font sizes for better visibility
        self.font = pygame.font.SysFont("Arial", 20)  # Increased from 22
        self.key_font = pygame.font.SysFont("Arial", 14)  # Increased from 14

        # Note names
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

        # Playback variables
        self.playing = False
        self.paused = False
        self.current_time = 0
        self.events_iterator = None
        self.tempo = 500000  # Default tempo (microseconds per beat)
        self.midi_events = []
        self.current_event_idx = 0

        # Sound variables
        self.sounds = {}
        self.volume = 0.5
        self.generate_piano_sounds()

    def generate_piano_sounds(self):
        """Generate sine wave sounds for each piano note"""
        # Only generate sounds for the keys we're displaying
        sample_rate = 44100
        max_amplitude = 32767  # For 16-bit audio
        duration = 2.0  # seconds
        samples = int(duration * sample_rate)

        for note in range(self.first_note, self.first_note + self.total_keys):
            # Calculate frequency using the equal temperament formula: f = 2^(n-49)/12 * 440Hz
            frequency = 440.0 * (2.0 ** ((note - 69) / 12.0))
            # Generate sine wave
            t = np.linspace(0, duration, samples, False)
            sine_wave = np.sin(2 * np.pi * frequency * t)
            # Apply envelope to avoid clicks and make piano-like sound
            attack = int(0.01 * sample_rate)  # 10ms attack
            decay = int(0.1 * sample_rate)  # 100ms decay
            # Create envelope
            envelope = np.ones(samples)
            # Attack
            envelope[:attack] = np.linspace(0, 1, attack)
            # Decay
            decay_end = attack + decay
            envelope[attack:decay_end] = np.linspace(1, 0.7, decay)
            # Release
            release = int(1.5 * sample_rate)  # 1.5s release
            if decay_end < samples:
                release_start = min(samples - release, decay_end)
                envelope[release_start:] = np.linspace(
                    envelope[release_start], 0, samples - release_start
                )

            # Apply envelope to sine wave
            sine_wave = sine_wave * envelope

            # Convert to 16-bit PCM
            audio_data = (sine_wave * max_amplitude * 0.5).astype(np.int16)

            # Create stereo sound
            stereo_data = np.column_stack((audio_data, audio_data))

            # Create sound object
            sound = pygame.sndarray.make_sound(stereo_data)

            # Store in dictionary
            self.sounds[note] = sound

    def play_note(self, note, velocity=127):
        """Play a note with given velocity"""
        if note in self.sounds:
            # Set volume based on velocity (MIDI velocity is 0-127)
            volume = (velocity / 127.0) * self.volume
            self.sounds[note].set_volume(volume)
            self.sounds[note].play()

    def stop_note(self, note):
        """Stop a note from playing"""
        if note in self.sounds:
            # Fade out instead of immediate stop for smoother sound
            self.sounds[note].fadeout(50)  # 50ms fadeout

    def is_black_key(self, note):
        """Check if a MIDI note number is a black key"""
        return (note % 12) in [1, 3, 6, 8, 10]

    def get_note_name(self, note):
        """Get the name of a note (e.g., C4, F#5) from its MIDI number"""
        note_name = self.note_names[note % 12]
        octave = (note - 12) // 12  # MIDI octave system
        return f"{note_name}{octave}"

    def get_key_position(self, note):
        # sourcery skip: extract-method, sum-comprehension
        """Get the x position of a key based on its MIDI note number"""
        # Adjust note to our piano range
        relative_note = note - self.first_note
        if relative_note < 0 or relative_note >= self.total_keys:
            return None

        # Count white keys before this note
        white_key_count = len([n for n in range(self.first_note, note) if not self.is_black_key(n)])

        if self.is_black_key(note):
            # For black keys (positioned between white keys)
            # Find the white key to the left
            prev_white = note - 1
            while self.is_black_key(prev_white) and prev_white >= self.first_note:
                prev_white -= 1

            if prev_white < self.first_note:
                return None

            # Count white keys up to the previous white key
            white_key_count_prev = 0
            for n in range(self.first_note, prev_white + 1):
                if not self.is_black_key(n):
                    white_key_count_prev += 1

            white_key_x = self.piano_start_x + (
                (white_key_count_prev - 1) * self.white_key_width
            )
            return white_key_x + (self.white_key_width - self.black_key_width // 2)

        return self.piano_start_x + (white_key_count * self.white_key_width)

    def draw_piano(self):
        """Draw the piano keyboard with key names"""
        # First draw all white keys
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

                    # Draw note name at the bottom of the white key
                    note_name = self.get_note_name(note)
                    name_text = self.key_font.render(note_name, True, (0, 0, 0))
                    text_x = x + (self.white_key_width - name_text.get_width()) // 2
                    text_y = self.piano_start_y + self.white_key_height - name_text.get_height() - 5
                    self.screen.blit(name_text, (text_x, text_y))

        # Then draw all black keys on top
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

                    # Draw note name on the black key
                    note_name = self.get_note_name(note)
                    name_text = self.key_font.render(note_name, True, (255, 255, 255))
                    text_x = x + (self.black_key_width - name_text.get_width()) // 2
                    text_y = self.piano_start_y + self.black_key_height - name_text.get_height() - 5
                    self.screen.blit(name_text, (text_x, text_y))

    def load_midi(self, file_path):
        """Load a MIDI file"""
        try:
            midi_file = mido.MidiFile(file_path)
            # Extract all note events with absolute timing
            self.midi_events = []

            ticks_per_beat = midi_file.ticks_per_beat
            tempo = 500000  # Default tempo in microseconds per beat

            for track in midi_file.tracks:
                absolute_time = 0
                for msg in track:
                    # Convert delta time to absolute time in milliseconds
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
                    elif msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                    ):
                        self.midi_events.append((absolute_time, "note_off", msg.note))

            # Sort events by time
            self.midi_events.sort(key=lambda x: x[0])
            self.current_event_idx = 0
            self.current_time = 0
            return True
        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return False

    def play(self):
        """Start MIDI playback"""
        self.playing = True
        self.paused = False

    def pause(self):
        """Pause MIDI playback"""
        self.paused = not self.paused
        if self.paused:
            # Stop all sounds when paused
            pygame.mixer.stop()

    def stop(self):
        """Stop MIDI playback"""
        self.playing = False
        self.current_time = 0
        self.current_event_idx = 0
        self.active_notes.clear()
        # Stop all sounds
        pygame.mixer.stop()

    def update(self, dt):
        """Update playback state"""
        if not self.playing or self.paused or not self.midi_events:
            return

        # Advance time
        self.current_time += dt * 1000  # Convert to milliseconds to match MIDI time

        # Process events that should occur at or before the current time
        while self.current_event_idx < len(self.midi_events) and (
            self.midi_events[self.current_event_idx][0] <= self.current_time
        ):
            event = self.midi_events[self.current_event_idx]

            if event[1] == "note_on":
                note = event[2]
                velocity = event[3]
                # Map velocity to color brightness
                brightness = int(velocity * 2) + 50
                self.active_notes[note] = True
                self.note_colors[note] = (200, 200, brightness)
                # Play sound
                self.play_note(note, velocity)

            elif event[1] == "note_off":
                note = event[2]
                self.active_notes[note] = False
                # Stop sound
                self.stop_note(note)

            elif event[1] == "tempo":
                self.tempo = event[2]

            self.current_event_idx += 1

            # If we've processed all events, stop playback
            if self.current_event_idx >= len(self.midi_events):
                self.playing = False
                break

    def render_ui(self):
        """Render UI elements"""
        # Draw playback status
        status = (
            "Playing"
            if self.playing and not self.paused
            else "Paused" if self.paused else "Stopped"
        )
        status_text = self.font.render(f"Status: {status}", True, (255, 255, 255))
        self.screen.blit(status_text, (10, 10))

        # Draw current time
        time_text = self.font.render(
            f"Time: {self.current_time/1000:.2f}s", True, (255, 255, 255)
        )
        self.screen.blit(time_text, (10, 40))

        # Draw controls help
        controls_text = self.font.render(
            "Controls: SPACE = Play/Pause, R = Reset, ESC = Quit, +/- = Volume",
            True,
            (255, 255, 255),
        )
        self.screen.blit(controls_text, (10, 70))

        # Draw volume indicator
        volume_text = self.font.render(
            f"Volume: {int(self.volume * 100)}%", True, (255, 255, 255)
        )
        self.screen.blit(volume_text, (10, 100))

        # Display note range information
        range_text = self.font.render(
            f"Key Range: {self.get_note_name(self.first_note)} to {self.get_note_name(self.first_note + self.total_keys - 1)}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(range_text, (10, 130))

        # Draw currently playing notes
        if active_notes := [
            self.get_note_name(note)
            for note in self.active_notes
            if self.active_notes[note]
        ]:
            notes_str = "Playing: " + ", ".join(active_notes[:8])
            if len(active_notes) > 8:
                notes_str += f" +{len(active_notes) - 8} more"
            notes_text = self.font.render(notes_str, True, (255, 255, 255))
            self.screen.blit(notes_text, (10, 160))

    def run(self, midi_file):
        """Main application loop."""
        if not self.load_midi(midi_file):
            print(f"Failed to load MIDI file: {midi_file}")
            return

        self.playing = True  # Start playing immediately after loading MIDI
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    self.handle_keydown_event(event.key)

            dt = clock.tick(60) / 1000.0
            self.update(dt)

            self.screen.fill((40, 40, 40))
            self.draw_piano()
            self.render_ui()
            pygame.display.flip()

        pygame.quit()

    def handle_keydown_event(self, key):
        """Handles keydown events for controlling playback and volume."""
        if key == pygame.K_ESCAPE:
            self.playing = False # Stop the main loop and quit
        elif key == pygame.K_SPACE:
            self.pause()
        elif key == pygame.K_r:
            self.stop()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS):
            self.volume = min(1.0, self.volume + 0.05)
        elif key == pygame.K_MINUS:
            self.volume = max(0.0, self.volume - 0.05)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python midi_visualizer.py <midi_file>")
        sys.exit(1)

    midi_file = sys.argv[1]
    visualizer = PianoVisualizer()
    visualizer.run(midi_file)
