import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np
from collections import defaultdict
import os


class PianoVisualizer:

    def __init__(self, width=1450, height=700):  # Increased height from 500 to 700
        pygame.init()
        pygame.midi.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Interactive Piano Keystrokes Visualizer")

        # Piano dimensions
        self.white_key_width = 50
        self.white_key_height = 220
        self.black_key_width = 30
        self.black_key_height = 130

        # Starting position for piano - adjusted for better balance
        self.piano_start_x = 20
        self.piano_start_y = 300  # Changed from 400 to 300 to make more room for notes

        # Note data
        self.active_notes = defaultdict(lambda: False)  # Currently pressed notes
        self.highlighted_notes = defaultdict(
            lambda: False
        )  # Notes highlighted from MIDI
        self.note_colors = defaultdict(lambda: (180, 180, 255))

        # From C2 (36) to B5 (83) = 48 keys (4 octaves)
        self.first_note = 36  # MIDI note number for C2
        self.total_keys = 48  # 4 octaves
        self.last_note = self.first_note + self.total_keys - 1  # MIDI note number for highest supported note

        # Track out-of-range notes we've already warned about to avoid spam
        self.warned_notes = set()

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
        # Load piano sounds
        self.generate_piano_sounds()

        # MIDI Input setup
        self.midi_input = None
        self.setup_midi_input()
        
        # MIDI Output setup (for Ableton Live or other DAWs)
        self.midi_output = None
        self.midi_output_name = "No Output"
        self.setup_midi_output()

        # Note guidance system
        self.note_highway_height = 350  # Increased from 180 to 350
        self.note_fall_speed = 100  # Pixels per second
        self.timed_notes = []  # List of (time, notes, duration) tuples
        self.playback_start_time = 0
        self.is_playing = False
        self.hit_line_y = self.piano_start_y - 10  # Position of the hit line
        self.visible_note_time = 4.0  # How many seconds of notes to show in advance
        
        # Highway colors - changed from yellow to more distinct colors
        self.highway_bg_color = (30, 30, 40)  # Darker blue background
        self.white_note_color = (70, 130, 180)  # Steel blue for white key notes
        self.black_note_color = (100, 100, 150)  # Muted purple for black key notes
        self.hit_line_color = (220, 50, 50)  # Bright red for the hit line

        # UI Style - simplified for minimalist look
        self.ui_bg_color = (30, 30, 35)
        self.ui_text_color = (220, 220, 220)
        self.ui_accent_color = (100, 150, 220)

    def generate_piano_sounds(self):
        """Generate simple piano tone sounds for each note using sine waves"""
        print("Generating piano sound samples...")
        sample_rate = 44100  # Sample rate in Hz
        max_sample_length = 2.0  # Maximum sample length in seconds

        for note in range(self.first_note, self.first_note + self.total_keys):
            try:
                # Calculate frequency using the formula: f = 440 * 2^((n-69)/12)
                frequency = 440 * (2 ** ((note - 69) / 12))

                # Generate samples for a sine wave
                duration = min(max_sample_length, 4.0 - (note - self.first_note) / 48)
                samples = np.sin(2 * np.pi * np.arange(sample_rate * duration) * frequency / sample_rate)

                # Apply attack and decay
                attack = int(sample_rate * 0.01)  # 10ms attack
                decay = int(sample_rate * duration * 0.8)
                if attack > 0:
                    samples[:attack] *= np.linspace(0, 1, attack)
                if decay < len(samples):
                    samples[decay:] *= np.linspace(1, 0, len(samples) - decay)

                # Convert to stereo by duplicating the mono channel
                samples = (samples * 32767).astype(np.int16)
                samples = np.column_stack((samples, samples))  # Make it 2D for stereo

                # Create a Pygame sound object
                sound = pygame.sndarray.make_sound(samples)
                self.sounds[note] = sound
            except Exception as e:
                print(f"Error generating sound for note {note}: {e}")

        print(f"Generated {len(self.sounds)} piano sounds")

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

    def setup_midi_output(self):
        """Set up MIDI output device to send notes to Ableton Live or other DAWs"""
        # List available MIDI output devices
        output_devices = []
        for i in range(pygame.midi.get_count()):
            device_info = pygame.midi.get_device_info(i)
            is_output = device_info[3]  # Index 3 is output capability (True/False)
            if is_output:
                device_name = device_info[1].decode("utf-8")
                output_devices.append((i, device_name))
                print(f"MIDI Output #{i}: {device_name}")
                
        if output_devices:
            # Print available MIDI output devices and instructions for connecting to Ableton Live
            self.print_midi_output_instructions(output_devices)
            
            # Automatically select first available output
            try:
                self.midi_output = pygame.midi.Output(output_devices[0][0])
                self.midi_output_name = output_devices[0][1]
                print(f"\nAutomatically connected to MIDI output: {self.midi_output_name}")
            except pygame.midi.MidiException as e:
                print(f"Could not open MIDI output: {e}")
        else:
            print("No MIDI output devices found.")

    def print_midi_output_instructions(self, output_devices):
        """Print available MIDI output devices and instructions for connecting to Ableton Live"""
        print("\nAvailable MIDI output devices:")
        for i, (device_id, name) in enumerate(output_devices):
            print(f"{i+1}. {name} (ID: {device_id})")
            
        print("\nTo connect to Ableton Live:")
        print("1. Choose a MIDI output device from the list above")
        print("2. In Ableton Live, go to Preferences > Link/MIDI")
        print("3. Find the device in the 'MIDI Ports' section")
        print("4. Set 'Track' to ON for that device in the 'Input' column")
        print("5. Create a MIDI track in Ableton and set its input to the same device")
        print("6. Set that track's Monitor to 'In' or 'Auto'")

    def is_black_key(self, note):
        """Check if a MIDI note number is a black key"""
        return (note % 12) in [1, 3, 6, 8, 10]

    def count_white_keys_before(self, note):
        return sum(not self.is_black_key(n) for n in range(self.first_note, note))

    def _find_previous_white_key(self, note):
        """Find the previous white key for a given note for positioning black keys.
        
        For black keys (positioned between white keys), determine the white key immediately to the left.
        """
        prev_white = note - 1
        while prev_white >= self.first_note and self.is_black_key(prev_white):
            prev_white -= 1
        return prev_white if prev_white >= self.first_note else None

    def get_note_name(self, note):
        """Get the name of a note (e.g., C4, F#5) from its MIDI number"""
        note_name = self.note_names[note % 12]
        octave = (note - 12) // 12  # MIDI octave system
        return f"{note_name}{octave}"

    def get_key_position(self, note):
        """Get the x position of a key based on its MIDI note number"""
        # For black keys (positioned between white keys)
        prev_white = self._find_previous_white_key(note)
        if prev_white is None:
            return None

        # Count white keys up to the previous white key using existing method
        white_key_count_prev = self.count_white_keys_before(prev_white + 1)

        white_key_x = self.piano_start_x + (
            (white_key_count_prev - 1) * self.white_key_width
        )
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

    def draw_note_highway(self):
        """Draw the note highway with falling notes"""
        if not self.is_playing or not self.timed_notes:
            return

        # Draw highway background with subtle grid
        highway_rect = pygame.Rect(
            self.piano_start_x, 
            self.piano_start_y - self.note_highway_height, 
            self.white_key_width * self.count_white_keys_before(self.first_note + self.total_keys),
            self.note_highway_height
        )
        pygame.draw.rect(self.screen, self.highway_bg_color, highway_rect)
        
        # Add subtle grid lines for timing reference
        grid_spacing = 50  # pixels
        for y in range(self.piano_start_y - self.note_highway_height, self.piano_start_y, grid_spacing):
            pygame.draw.line(
                self.screen,
                (40, 45, 55),  # Subtle darker line
                (self.piano_start_x, y),
                (highway_rect.right, y),
                1
            )
            
        # Draw the hit line
        pygame.draw.line(
            self.screen,
            self.hit_line_color,
            (self.piano_start_x, self.hit_line_y),
            (
                self.piano_start_x
                + self.white_key_width
                * self.count_white_keys_before(self.first_note + self.total_keys),
                self.hit_line_y,
            ),
            3,
        )

        # Get the current playback time
        current_time = time.time() - self.playback_start_time

        # Find notes that should be visible
        visible_start = current_time - 0.5  # Show notes slightly before they're needed
        visible_end = current_time + self.visible_note_time

        for start_time, notes, duration in self.timed_notes:
            # Skip notes that are already past or too far in the future
            if start_time < visible_start - duration or start_time > visible_end:
                continue

            # Calculate the vertical position of the note
            time_diff = start_time - current_time
            y_pos = self.hit_line_y - (time_diff * self.note_fall_speed)

            # Draw a bar for each note in the chord
            for note in notes:
                if self.first_note <= note < self.first_note + self.total_keys:
                    x_pos = self.get_key_position(note)
                    if x_pos is None:
                        continue

                    # Determine width based on white or black key
                    width = (
                        self.black_key_width
                        if self.is_black_key(note)
                        else self.white_key_width - 1
                    )

                    # Draw the note bar
                    height = max(
                        10, min(duration * self.note_fall_speed, 100)
                    )  # Cap the height

                    # Color based on the note type - using the new color scheme
                    if self.is_black_key(note):
                        color = self.black_note_color
                    else:
                        color = self.white_note_color

                    pygame.draw.rect(
                        self.screen, color, (x_pos, y_pos - height, width, height)
                    )
                    pygame.draw.rect(
                        self.screen,
                        (0, 0, 0),
                        (x_pos, y_pos - height, width, height),
                        1,
                    )

    def load_midi(self, file_path):
        """Load a MIDI file and extract notes/chords for highlighting with timing"""
        try:
            midi_file = mido.MidiFile(file_path)

            # Get the MIDI file's tempo (microseconds per beat)
            tempo = 500000  # Default tempo (120 BPM)
            for track in midi_file.tracks:
                for msg in track:
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                        break
                if tempo != 500000:
                    break

            ticks_per_beat = midi_file.ticks_per_beat
            seconds_per_tick = tempo / (1000000 * ticks_per_beat)

            # Extract timed notes/chords
            self.timed_notes = []
            active_notes = {}  # note -> (start_time, velocity)

            for track in midi_file.tracks:
                absolute_ticks = 0

                for msg in track:
                    absolute_ticks += msg.time

                    # Convert ticks to seconds
                    abs_time = absolute_ticks * seconds_per_tick

                    if msg.type == "note_on" and msg.velocity > 0:
                        # Note is starting
                        active_notes[msg.note] = (abs_time, msg.velocity)

                    elif msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                    ):
                        # Note is ending
                        if msg.note in active_notes:
                            start_time, velocity = active_notes[msg.note]
                            duration = abs_time - start_time
                            # Store as (start_time, [note], duration)
                            self.timed_notes.append((start_time, [msg.note], duration))
                            del active_notes[msg.note]

            # Sort by start time
            self.timed_notes.sort(key=lambda x: x[0])

            # Also create the chord sequence for the existing highlight mode
            unique_chords = []
            seen_chords = set()

            for start_time, notes, _ in self.timed_notes:
                chord = frozenset(notes)
                if chord not in seen_chords:
                    unique_chords.append(set(notes))
                    seen_chords.add(chord)

            self.chord_sequence = unique_chords
            self.current_chord_idx = 0

            print(f"Loaded {len(self.timed_notes)} notes/chords from MIDI file")
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

        if self.current_chord_idx < len(self.chord_sequence):
            self.highlight_current_chord()

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
            self.highlight_current_chord()
            velocity = 100
            for note in chord:
                print(f"Playing note: {self.get_note_name(note)} (velocity: {velocity})")
                if self.midi_output is not None:
                    try:
                        # Note on: 0x90 = channel 1 note on, note number, velocity
                        self.midi_output.write_short(0x90, note, velocity)
                    except Exception as e:
                        print(f"Error sending MIDI note: {e}")
    
    def stop_note(self, note):
        """Stop a note playing and send MIDI note-off message if output available"""
        # Check if note is in supported range
        if note < self.first_note or note > self.last_note:
            return
            
    def highlight_current_chord(self):
        """Highlight the current chord and display its info"""
        chord = self.chord_sequence[self.current_chord_idx]
        for note in chord:
            if self.first_note <= note < self.first_note + self.total_keys:
                self.highlighted_notes[note] = True
        self.current_chord_idx += 1

        notes = [
            self.get_note_name(note)
            for note in chord
            if self.first_note <= note < self.first_note + self.total_keys
        ]
        print(
            f"Chord {self.current_chord_idx}/{len(self.chord_sequence)}: {', '.join(notes)}"
        )

    def play_note(self, note, velocity=100):
        if note in self.sounds:
            self.sounds[note].fadeout(50)  # 50ms fadeout
            
        # Still send note-off to MIDI output if available
        if self.midi_output is not None:
            try:
                # Note off: 0x80 = channel 1 note off, note number, 0 velocity
                self.midi_output.write_short(0x80, note, 0)
            except Exception as e:
                print(f"Error sending MIDI note off: {e}")

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
                status = event[0][0]
                note = event[0][1]
                velocity = event[0][2]

                # Note On event
                if 144 <= status <= 159 and velocity > 0:
                    self.active_notes[note] = True
                    self.play_note(note, velocity)  # Will handle out-of-range notes
                # Note Off event
                elif (128 <= status <= 143) or (144 <= status <= 159 and velocity == 0):
                    self.active_notes[note] = False
                    self.stop_note(note)  # Will handle out-of-range notes

    def handle_keyboard_events(self, events):
        """Handle keyboard events for navigation only (no piano playing)"""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # Signal to quit

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

                # Playback controls
                elif event.key == pygame.K_SPACE:
                    self.toggle_playback()
                elif event.key == pygame.K_r:  # Reset playback
                    self.playback_start_time = time.time()

                # Volume controls
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    self.volume = min(1.0, self.volume + 0.05)
                elif event.key == pygame.K_MINUS:
                    self.volume = max(0.0, self.volume - 0.05)

        return True  # Continue running

    def toggle_playback(self):
        """Toggle playback of the MIDI visualization"""
        if self.is_playing:
            self.is_playing = False
        else:
            self.is_playing = True
            self.playback_start_time = time.time()

    def check_highlighted_played(self):
        """Check if the user played all the currently highlighted notes"""
        if not self.highlighting_mode or not self.highlighted_notes:
            return

        # Get current highlighted notes
        highlighted = {
            note for note in self.highlighted_notes if self.highlighted_notes[note]
        }
        active = {note for note in self.active_notes if self.active_notes[note]}

        # If all highlighted notes are being played (and nothing extra)
        if highlighted == active and highlighted:
            # Flash confirmation and automatically move to next chord
            pygame.display.flip()
            pygame.time.delay(200)  # Brief flash
            self.next_chord()

    def render_ui(self):
        """Render minimalist UI elements at the bottom of the keyboard"""
        # Add bottom UI area
        bottom_ui_y = self.piano_start_y + self.white_key_height + 10
        bottom_ui_height = 50
        ui_rect = pygame.Rect(0, bottom_ui_y, self.width, bottom_ui_height)
        pygame.draw.rect(self.screen, self.ui_bg_color, ui_rect)
        
        # Draw subtle separator line
        pygame.draw.line(
            self.screen,
            (60, 60, 70),
            (0, bottom_ui_y),
            (self.width, bottom_ui_y),
            1
        )
        
        # Define text positions
        text_y = bottom_ui_y + 15
        left_x = 20
        center_x = self.width // 2
        right_x = self.width - 360
        
        # Left side: Mode indicator
        mode_text = f"Mode: {'Highlight' if self.highlighting_mode else 'Free Play'}"
        mode_surface = self.font.render(mode_text, True, self.ui_text_color)
        self.screen.blit(mode_surface, (left_x, text_y))
        
        # Center: MIDI and playback status
        midi_in_status = "MIDI In: " + ("Connected" if self.midi_input else "None")
        midi_out_status = f"MIDI Out: {self.midi_output_name if self.midi_output else 'None'}"
        play_status = "Playing" if self.is_playing else "Paused"
        
        # Display on separate lines for better readability
        status_text = f"{midi_in_status} | {midi_out_status}"
        status_surface = self.font.render(status_text, True, self.ui_text_color)
        status_rect = status_surface.get_rect(center=(center_x, text_y))
        self.screen.blit(status_surface, status_rect)
        
        play_surface = self.font.render(play_status, True, self.ui_text_color)
        play_rect = play_surface.get_rect(center=(center_x, text_y + 24))
        self.screen.blit(play_surface, play_rect)
        
        # Right side: Controls reminder
        controls_text = "ESC: Quit | SPACE: Play/Pause | H: Highlight Mode | ←/→: Nav Chords"
        controls_surface = self.font.render(controls_text, True, self.ui_text_color)
        self.screen.blit(controls_surface, (right_x, text_y))
        
        # If in highlight mode, show current chord above the keyboard
        if self.highlighting_mode and self.highlighted_notes and self.chord_sequence:
            if (highlighted := [self.get_note_name(note) for note in self.highlighted_notes if self.highlighted_notes[note]]):
                chord_text = f"Play: {', '.join(highlighted)} ({self.current_chord_idx}/{len(self.chord_sequence)})"
                chord_surface = self.font.render(chord_text, True, (255, 255, 100))
                chord_rect = chord_surface.get_rect(center=(center_x, self.piano_start_y - 20))
                self.screen.blit(chord_surface, chord_rect)

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
            self.draw_note_highway()  # Draw the falling notes
            self.draw_piano()
            self.render_ui()
            pygame.display.flip()

            # Cap at 60 FPS
            clock.tick(60)

        # Clean up MIDI resources
        if self.midi_input:
            self.midi_input.close()
        if self.midi_output:
            self.midi_output.close()
        pygame.midi.quit()
        pygame.quit()


if __name__ == "__main__":
    midi_file = sys.argv[1] if len(sys.argv) > 1 else None

    visualizer = PianoVisualizer()
    visualizer.run(midi_file)
