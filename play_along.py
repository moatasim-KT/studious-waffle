import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np
from collections import defaultdict


class PianoVisualizer:

    def __init__(self, width=1450, height=500):
        pygame.init()
        pygame.midi.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Interactive Piano Keystrokes Visualizer")

        # Piano dimensions
        self.white_key_width = 50  # Reduced from 60 to fit more keys
        self.white_key_height = 220
        self.black_key_width = 30  # Reduced from 40
        self.black_key_height = 130

        # Starting position for piano
        self.piano_start_x = 20
        self.piano_start_y = 200

        # Note data
        self.active_notes = defaultdict(lambda: False)  # Currently pressed notes
        self.highlighted_notes = defaultdict(
            lambda: False
        )  # Notes highlighted from MIDI
        self.note_colors = defaultdict(lambda: (180, 180, 255))

        # From C2 (36) to B5 (83) = 48 keys (4 octaves)
        self.first_note = 36  # MIDI note number for C2
        self.total_keys = 48  # 4 octaves

        # Font sizes
        self.font = pygame.font.SysFont("Arial", 24)
        self.key_font = pygame.font.SysFont("Arial", 16)  # For key labels

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

        # Key mapping (computer keyboard to MIDI notes)
        self.key_mapping = {
            pygame.K_z: 36,  # C2
            pygame.K_s: 37,  # C#2
            pygame.K_x: 38,  # D2
            pygame.K_d: 39,  # D#2
            pygame.K_c: 40,  # E2
            pygame.K_v: 41,  # F2
            pygame.K_g: 42,  # F#2
            pygame.K_b: 43,  # G2
            pygame.K_h: 44,  # G#2
            pygame.K_n: 45,  # A2
            pygame.K_j: 46,  # A#2
            pygame.K_m: 47,  # B2
            pygame.K_COMMA: 48,  # C3
            pygame.K_l: 49,  # C#3
            pygame.K_PERIOD: 50,  # D3
            pygame.K_SEMICOLON: 51,  # D#3
            pygame.K_SLASH: 52,  # E3
            # Upper row for another octave
            pygame.K_q: 48,  # C3
            pygame.K_2: 49,  # C#3
            pygame.K_w: 50,  # D3
            pygame.K_3: 51,  # D#3
            pygame.K_e: 52,  # E3
            pygame.K_r: 53,  # F3
            pygame.K_5: 54,  # F#3
            pygame.K_t: 55,  # G3
            pygame.K_6: 56,  # G#3
            pygame.K_y: 57,  # A3
            pygame.K_7: 58,  # A#3
            pygame.K_u: 59,  # B3
            pygame.K_i: 60,  # C4 (middle C)
            pygame.K_9: 61,  # C#4
            pygame.K_o: 62,  # D4
            pygame.K_0: 63,  # D#4
            pygame.K_p: 64,  # E4
            pygame.K_LEFTBRACKET: 65,  # F4
            pygame.K_EQUALS: 66,  # F#4
            pygame.K_RIGHTBRACKET: 67,  # G4
        }

        # Playback mode variables
        self.highlighting_mode = False
        self.midi_events = []
        self.current_chord = set()  # Set of notes in the current chord
        # Initialize chord_sequence and current_chord_idx
        self.chord_sequence = []
        self.current_chord_idx = 0

        # Sound variables
        self.sounds = {}
        self.volume = 0.5
        # self.generate_piano_sounds()

        # MIDI Input setup
        self.midi_input = None
        self.setup_midi_input()

    def setup_midi_input(self):
        """Set up MIDI input device"""
        # List available MIDI input devices
        input_devices = []
        for i in range(pygame.midi.get_count()):
            device_info = pygame.midi.get_device_info(i)
            is_input = device_info[2]
            if is_input:
                device_name = device_info[1].decode("utf-8")
                input_devices.append((i, device_name))
                print(f"MIDI Input #{i}: {device_name}")

        # If we have MIDI input devices available
        if input_devices:
            # Find first available input device
            for device_id, name in input_devices:
                try:
                    self.midi_input = pygame.midi.Input(device_id)
                    print(f"Connected to MIDI device: {name}")
                    break
                except pygame.midi.MidiException as e:
                    print(f"Could not open MIDI device {device_id} ({name}): {e}")

            if self.midi_input is None:
                print(
                    "No MIDI input devices could be opened. Using computer keyboard only."
                )
        else:
            print("No MIDI input devices found. Using computer keyboard only.")

    # def generate_piano_sounds(self):
    #     """Generate sine wave sounds for each piano note"""
    #     sample_rate = 44100
    #     max_amplitude = 32767  # For 16-bit audio

    #     # Only generate sounds for the keys we're displaying
    #     for note in range(self.first_note, self.first_note + self.total_keys):
    #         # Calculate frequency using the equal temperament formula: f = 2^(n-49)/12 * 440Hz
    #         frequency = 440.0 * (2.0 ** ((note - 69) / 12.0))

    #         # Generate 1 second of sound (can be shorter in practice)
    #         duration = 2.0  # seconds
    #         samples = int(duration * sample_rate)

    #         # Generate sine wave
    #         t = np.linspace(0, duration, samples, False)
    #         sine_wave = np.sin(2 * np.pi * frequency * t)

    #         # Apply envelope to avoid clicks and make piano-like sound
    #         attack = int(0.01 * sample_rate)  # 10ms attack
    #         decay = int(0.1 * sample_rate)  # 100ms decay
    #         release = int(1.5 * sample_rate)  # 1.5s release

    #         # Create envelope
    #         envelope = np.ones(samples)
    #         # Attack
    #         envelope[:attack] = np.linspace(0, 1, attack)
    #         # Decay
    #         decay_end = attack + decay
    #         envelope[attack:decay_end] = np.linspace(1, 0.7, decay)
    #         # Release
    #         if decay_end < samples:
    #             release_start = min(samples - release, decay_end)
    #             envelope[release_start:] = np.linspace(
    #                 envelope[release_start], 0, samples - release_start
    #             )

    #         # Apply envelope to sine wave
    #         sine_wave = sine_wave * envelope

    #         # Convert to 16-bit PCM
    #         audio_data = (sine_wave * max_amplitude * 0.5).astype(np.int16)

    #         # Create stereo sound
    #         stereo_data = np.column_stack((audio_data, audio_data))

    #         # Create sound object
    #         sound = pygame.sndarray.make_sound(stereo_data)

    #         # Store in dictionary
    #         self.sounds[note] = sound

    # def play_note(self, note, velocity=127):
    #     """Play a note with given velocity"""
    #     if note in self.sounds:
    #         # Set volume based on velocity (MIDI velocity is 0-127)
    #         volume = (velocity / 127.0) * self.volume
    #         self.sounds[note].set_volume(volume)
    #         self.sounds[note].play()

    # def stop_note(self, note):
    #     """Stop a note from playing"""
    #     if note in self.sounds:
    #         # Fade out instead of immediate stop for smoother sound
    #         self.sounds[note].fadeout(50)  # 50ms fadeout

    def is_black_key(self, note):
        """Check if a MIDI note number is a black key"""
        return (note % 12) in [1, 3, 6, 8, 10]

    def get_note_name(self, note):
        """Get the name of a note (e.g., C4, F#5) from its MIDI number"""
        note_name = self.note_names[note % 12]
        octave = (note - 12) // 12  # MIDI octave system
        return f"{note_name}{octave}"

    def get_key_position(self, note):
        """Get the x position of a key based on its MIDI note number"""
        # Adjust note to our piano range
        relative_note = note - self.first_note
        if relative_note < 0 or relative_note >= self.total_keys:
            return None

        white_key_count = self.count_white_keys_before(note)

        if not self.is_black_key(note):
            return self.piano_start_x + (white_key_count * self.white_key_width)
        else:
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

            white_key_x = self.piano_start_x + ((white_key_count_prev - 1) * self.white_key_width)
            return white_key_x + (self.white_key_width - self.black_key_width // 2)

    def draw_piano(self):
        """Draw the piano keyboard with key names"""
        # First draw all white keys
        for note in range(self.first_note, self.first_note + self.total_keys):
            if not self.is_black_key(note):
                x = self.get_key_position(note)
                if x is not None:
                    # Determine color based on state
                    if self.active_notes[note]:  # User is actively pressing
                        color = (100, 100, 255)  # Blue for active press
                    elif self.highlighted_notes[note]:  # Note is highlighted from MIDI
                        color = (255, 255, 100)  # Yellow for highlighted notes
                    else:
                        color = (255, 255, 255)  # Default white

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
                    text_y = (
                        self.piano_start_y
                        + self.white_key_height
                        - name_text.get_height()
                        - 5
                    )
                    self.screen.blit(name_text, (text_x, text_y))

        # Then draw all black keys on top
        for note in range(self.first_note, self.first_note + self.total_keys):
            if self.is_black_key(note):
                x = self.get_key_position(note)
                if x is not None:
                    # Determine color based on state
                    if self.active_notes[note]:  # User is actively pressing
                        color = (80, 80, 255)  # Blue for active press
                    elif self.highlighted_notes[note]:  # Note is highlighted from MIDI
                        color = (
                            200,
                            200,
                            80,
                        )  # Darker yellow for highlighted black keys
                    else:
                        color = (0, 0, 0)  # Default black

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
                    text_y = (
                        self.piano_start_y
                        + self.black_key_height
                        - name_text.get_height()
                        - 5
                    )
                    self.screen.blit(name_text, (text_x, text_y))

    def load_midi(self, file_path):
        """Load a MIDI file and extract notes/chords for highlighting"""
        try:
            midi_file = mido.MidiFile(file_path)

            # Extract chords (notes that play simultaneously)
            events = []
            for track in midi_file.tracks:
                abs_time = 0
                current_notes = set()

                for msg in track:
                    abs_time += msg.time

                    if msg.type == "note_on" and msg.velocity > 0:
                        current_notes.add(msg.note)
                        events.append((abs_time, "chord", set(current_notes)))
                    elif msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                    ):
                        if msg.note in current_notes:
                            current_notes.remove(msg.note)

            # Sort events by time and remove duplicates
            events.sort(key=lambda x: x[0])

            # Extract unique chords
            unique_chords = []
            prev_chord = set()

            for time, event_type, chord in events:
                if chord != prev_chord and len(chord) > 0:
                    unique_chords.append(chord)
                    prev_chord = chord

            self.chord_sequence = unique_chords
            self.current_chord_idx = 0

            print(
                f"Loaded {len(self.chord_sequence)} unique chords/notes from MIDI file"
            )
            return True

        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return False

    def next_chord(self):
        """Move to the next chord in the sequence"""
        if not self.chord_sequence:
            return

        # Clear previous highlights
        self.highlighted_notes.clear()

        # Move to next chord
        if self.current_chord_idx < len(self.chord_sequence):
            chord = self.chord_sequence[self.current_chord_idx]
            for note in chord:
                if self.first_note <= note < self.first_note + self.total_keys:
                    self.highlighted_notes[note] = True

            self.current_chord_idx += 1

            # Display chord info
            notes = [
                self.get_note_name(note)
                for note in chord
                if self.first_note <= note < self.first_note + self.total_keys
            ]
            print(
                f"Chord {self.current_chord_idx}/{len(self.chord_sequence)}: {', '.join(notes)}"
            )

    def prev_chord(self):
        """Move to the previous chord in the sequence"""
        if not self.chord_sequence:
            return

        # Clear previous highlights
        self.highlighted_notes.clear()

        # Move to previous chord
        self.current_chord_idx = max(0, self.current_chord_idx - 2)
        if 0 <= self.current_chord_idx < len(self.chord_sequence):
            chord = self.chord_sequence[self.current_chord_idx]
            for note in chord:
                if self.first_note <= note < self.first_note + self.total_keys:
                    self.highlighted_notes[note] = True

            self.current_chord_idx += 1

            # Display chord info
            notes = [
                self.get_note_name(note)
                for note in chord
                if self.first_note <= note < self.first_note + self.total_keys
            ]
            print(
                f"Chord {self.current_chord_idx}/{len(self.chord_sequence)}: {', '.join(notes)}"
            )

    def process_midi_input(self):
        """Process incoming MIDI messages from the external MIDI keyboard"""
        if self.midi_input is None:
            return

        # Check if there are any pending MIDI events
        if self.midi_input.poll():
            # Read all available MIDI events
            midi_events = self.midi_input.read(64)

            for event in midi_events:
                # event[0] contains the MIDI data as a list [status, data1, data2, 0]
                # For note events: status (144-159=note on, 128-143=note off), data1=note, data2=velocity
                status = event[0][0]
                note = event[0][1]
                velocity = event[0][2]

                # Check if the note is within our piano range
                if self.first_note <= note < self.first_note + self.total_keys:
                    # Note On event (in MIDI, channel 1 note on = 144, channel 2 = 145, etc.)
                    if 144 <= status <= 159 and velocity > 0:
                        self.active_notes[note] = True
                        # self.play_note(note, velocity)
                    # Note Off event (or Note On with velocity 0)
                    elif (128 <= status <= 143) or (
                        144 <= status <= 159 and velocity == 0
                    ):
                        self.active_notes[note] = False
                        # self.stop_note(note)

    def handle_keyboard_events(self, events):
        """Handle keyboard events for piano playing"""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # Signal to quit

                # Piano mode keys
                elif event.key in self.key_mapping:
                    note = self.key_mapping[event.key]
                    if self.first_note <= note < self.first_note + self.total_keys:
                        self.active_notes[note] = True
                        # self.play_note(note)

                # Navigation controls for highlight mode
                elif event.key == pygame.K_RIGHT:
                    self.next_chord()
                elif event.key == pygame.K_LEFT:
                    self.prev_chord()
                elif event.key == pygame.K_h:
                    self.highlighting_mode = not self.highlighting_mode
                    message = (
                        "Highlighting mode ON"
                        if self.highlighting_mode
                        else "Highlighting mode OFF"
                    )
                    print(message)

                # Volume controls
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    self.volume = min(1.0, self.volume + 0.05)
                elif event.key == pygame.K_MINUS:
                    self.volume = max(0.0, self.volume - 0.05)

            elif event.type == pygame.KEYUP:
                if event.key in self.key_mapping:
                    note = self.key_mapping[event.key]
                    if self.first_note <= note < self.first_note + self.total_keys:
                        self.active_notes[note] = False
                        # self.stop_note(note)

        return True  # Continue running

    def check_highlighted_played(self):
        """Check if the user played all the currently highlighted notes"""
        if not self.highlighting_mode or not self.highlighted_notes:
            return

        # Get current highlighted notes
        highlighted = set(
            note for note in self.highlighted_notes if self.highlighted_notes[note]
        )
        active = set(note for note in self.active_notes if self.active_notes[note])

        # If all highlighted notes are being played (and nothing extra)
        if highlighted == active and highlighted:
            # Flash confirmation and automatically move to next chord
            pygame.display.flip()
            pygame.time.delay(200)  # Brief flash
            self.next_chord()

    def render_ui(self):
        """Render UI elements"""
        # Draw mode information
        mode_text = self.font.render(
            f"Mode: {'Highlight' if self.highlighting_mode else 'Free Play'}",
            True,
            (255, 255, 255),
        )
        self.screen.blit(mode_text, (10, 10))

        # Draw MIDI input status
        midi_status = "MIDI: Connected" if self.midi_input else "MIDI: Not Connected"
        midi_text = self.font.render(midi_status, True, (255, 255, 255))
        self.screen.blit(midi_text, (300, 10))

        # Draw chord progress if in highlight mode
        if self.highlighting_mode and self.chord_sequence:
            progress_text = self.font.render(
                f"Chord: {self.current_chord_idx}/{len(self.chord_sequence)}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(progress_text, (10, 40))

            # Display currently highlighted notes
            if self.highlighted_notes:
                highlighted = [
                    self.get_note_name(note)
                    for note in self.highlighted_notes
                    if self.highlighted_notes[note]
                ]
                notes_str = "Highlighted: " + ", ".join(highlighted)
                notes_text = self.font.render(notes_str, True, (255, 255, 100))
                self.screen.blit(notes_text, (10, 70))

        # Draw volume indicator
        volume_text = self.font.render(
            f"Volume: {int(self.volume * 100)}%", True, (255, 255, 255)
        )
        self.screen.blit(volume_text, (10, 100))

        # Draw controls help
        controls1 = self.font.render(
            "Controls: H = Toggle Highlight Mode, ESC = Quit, +/- = Volume",
            True,
            (255, 255, 255),
        )
        controls2 = self.font.render(
            "In Highlight Mode: LEFT/RIGHT = Prev/Next Chord", True, (255, 255, 255)
        )
        self.screen.blit(controls1, (10, 130))
        self.screen.blit(controls2, (10, 160))

        # Display currently playing notes
        active_notes = [
            self.get_note_name(note)
            for note in self.active_notes
            if self.active_notes[note]
        ]
        if active_notes:
            playing_str = "Playing: " + ", ".join(active_notes[:8])
            if len(active_notes) > 8:
                playing_str += f" +{len(active_notes) - 8} more"
            playing_text = self.font.render(playing_str, True, (180, 180, 255))
            self.screen.blit(playing_text, (500, 10))

    def run(self, midi_file=None):
        """Main application loop"""
        # Load MIDI file if provided
        if midi_file:
            if not self.load_midi(midi_file):
                print(f"Failed to load MIDI file: {midi_file}")
                print("Running in free play mode only")
            else:
                self.highlighting_mode = True
                self.next_chord()  # Show first chord

        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle events
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    running = False

            # Process keyboard input
            if not self.handle_keyboard_events(events):
                running = False

            # Process MIDI input from external keyboard
            self.process_midi_input()

            # Check if highlighted notes are played correctly
            self.check_highlighted_played()

            # Render
            self.screen.fill((40, 40, 40))
            self.draw_piano()
            self.render_ui()
            pygame.display.flip()

            # Cap at 60 FPS
            clock.tick(60)

        # Clean up MIDI resources
        if self.midi_input:
            self.midi_input.close()
        pygame.midi.quit()
        pygame.quit()


if __name__ == "__main__":
    midi_file = None
    if len(sys.argv) > 1:
        midi_file = sys.argv[1]

    visualizer = PianoVisualizer()
    visualizer.run(midi_file)
