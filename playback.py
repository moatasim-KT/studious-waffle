import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np
from collections import defaultdict
import sounddevice as sd
import soundfile as sf
import pygame.midi
import argparse


class PianoVisualizer:
    def __init__(self, width=1600, height=500, mode="playback"):
        pygame.init()
        pygame.midi.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=512)
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("MIDI to Piano Keystrokes Visualizer")
        self.mode = mode  # "playback" or "play-along"

        # Piano dimensions
        self.white_key_width = 41  # Reduced from 60 to fit more keys
        self.white_key_height = 220
        self.black_key_width = 30  # Reduced from 40
        self.black_key_height = 130

        # Starting position for piano
        self.piano_start_x = 20
        self.piano_start_y = 200

        # Note data
        self.active_notes = defaultdict(lambda: False)
        self.note_colors = defaultdict(lambda: (180, 180, 255))

        # From C2 (36) to B5 (83) = 48 keys (4 octaves)
        self.first_note = 24  # MIDI note number for C2
        self.total_keys = 60  # 4 octaves

        # Font sizes
        self.font = pygame.font.SysFont("Arial", 24)
        self.key_font = pygame.font.SysFont(
            "Arial", 16
        )  # Reduced from 18 for smaller keys

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
        # Playback mode variables
        self.highlighting_mode = False
        self.chord_sequence = []
        self.current_chord_idx = 0
        self.highlighted_notes = defaultdict(lambda: False)  # Notes highlighted from MIDI

        # Sound variables
        self.sounds = {}
        self.volume = 0.5
        self.sample_rate = 44100  # Assuming 44.1kHz sample rate for the WAV files
        self.piano_samples = {}
        self.load_piano_samples()

        # MIDI Input setup
        self.midi_input = None
        self.setup_midi_input()

    def setup_midi_input(self):
        """Set up MIDI input device"""
        # Initialize pygame.midi if not already initialized
        if not pygame.midi.get_init():
            pygame.midi.init()

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

    def load_piano_samples(self):
        """Load piano samples from WAV files and map them to MIDI notes."""
        # Assuming WAV files are named like "piano_C4.wav", "piano_D#5.wav", etc.
        # and located in a "samples" directory
        try:
            import os

            samples_dir = "samples"  # You might need to adjust the path
            if not os.path.exists(samples_dir):
                os.makedirs(samples_dir)
                print(f"Created samples directory: {samples_dir}")
                return

            for note in range(self.first_note, self.first_note + self.total_keys):
                note_name = self.get_note_name(note)
                file_path = os.path.join(samples_dir, f"piano_{note_name}.wav")

                # Check if the sample file exists
                if not os.path.isfile(file_path):
                    print(f"Sample file not found: {file_path}")
                    # Generate sine wave instead
                    duration = 1.0  # 1 second duration
                    data = self.generate_sine_wave(note, duration)
                    self.piano_samples[note] = data
                    print(f"Synthesized sample for {note_name} ({note})")
                    continue

                try:
                    data, fs = sf.read(file_path)
                    if fs != self.sample_rate:
                        print(
                            f"Resampling {file_path} from {fs} Hz to {self.sample_rate} Hz"
                        )
                        # Simple resampling using numpy - for better quality consider librosa or scipy
                        ratio = self.sample_rate / fs
                        new_length = int(len(data) * ratio)
                        data_resampled = np.interp(
                            np.linspace(0, len(data), new_length),
                            np.arange(len(data)),
                            data,
                        )
                        data = data_resampled
                    self.piano_samples[note] = data
                    print(f"Loaded sample for {note_name} ({note})")
                except Exception as e:
                    print(f"Error loading sample for {note_name} ({note}): {e}")

            if not self.piano_samples:
                print("No piano samples loaded. Check the samples directory and file names.")

        except Exception as e:
            print(f"Error loading piano samples: {e}")

    def generate_sine_wave(self, note, duration=1.0):
        """Generates a sine wave for a given MIDI note and duration."""
        frequency = 440.0 * (2.0 ** ((note - 69) / 12.0))
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.sin(2 * np.pi * frequency * t)
        return wave

    def play_note(self, note, velocity=127):
        """Play a note with given velocity using Sounddevice."""
        if note in self.piano_samples:
            try:
                # Adjust volume based on velocity
                volume = (velocity / 127.0) * self.volume
                samples = self.piano_samples[note] * volume

                # Play the sample using Sounddevice
                sd.play(samples, self.sample_rate)
                # You might want to keep track of active streams for stopping notes later
                #  but for now we assume short samples that stop quickly

            except Exception as e:
                print(f"Error playing note {note}: {e}")
        else:
            print(f"No sample found for note {note}")

    def stop_note(self, note):
        """Stop a note from playing (if possible)."""
        # With this basic implementation, we can't stop individual notes
        #  since we don't track active streams.  A more advanced system
        #  would be needed for accurate note stopping, or using shorter samples.
        pass  # For now, do nothing

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

        # Count white keys before this note
        white_key_count = 0
        for n in range(self.first_note, note):
            if not self.is_black_key(n):
                white_key_count += 1

        if not self.is_black_key(note):
            # For white keys
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
                    text_y = (
                        self.piano_start_y
                        + self.black_key_height
                        - name_text.get_height()
                        - 5
                    )
                    self.screen.blit(name_text, (text_x, text_y))

    def load_midi(self, file_path):
        """Load a MIDI file"""
        try:
            midi_file = mido.MidiFile(file_path)
            # Extract all note events with absolute timing
            self.midi_events = []
            self.chord_sequence = []  # Initialize chord sequence
            self.current_chord_idx = 0
            ticks_per_beat = midi_file.ticks_per_beat
            tempo = 500000  # Default tempo in microseconds per beat
            
            # Variables for chord extraction
            events = []
            
            for track in midi_file.tracks:
                absolute_time = 0
                current_notes = set()
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
                        # Chord extraction
                        current_notes.add(msg.note)
                        events.append((absolute_time, "chord", set(current_notes)))
                    elif msg.type == "note_off" or (
                        msg.type == "note_on" and msg.velocity == 0
                    ):
                        self.midi_events.append((absolute_time, "note_off", msg.note))
                        # Chord extraction
                        if msg.note in current_notes:
                            current_notes.remove(msg.note)
                        

            # Sort events by time
            self.midi_events.sort(key=lambda x: x[0])
            
            #Chord Extraction
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
            

            print(
                f"Loaded {len(self.chord_sequence)} unique chords/notes from MIDI file"
            )
            return True
        except Exception as e:
            print(f"Error loading MIDI file: {e}")
            return False
        for time, event_type, chord in events:
            if chord != prev_chord and len(chord) > 0:
                unique_chords.append(chord)
                prev_chord = chord

        self.chord_sequence = unique_chords
        self.current_chord_idx = 0

        print(
            f"Loaded {len(self.chord_sequence)} unique chords/notes from MIDI file"
        )

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
        self.highlighted_notes.clear()

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
            #print(
            #    f"Chord {self.current_chord_idx}/{len(self.chord_sequence)}: {', '.join(notes)}"
            #)
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
            #print(
            #    f"Chord {self.current_chord_idx}/{len(self.chord_sequence)}: {', '.join(notes)}"
            #)
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

        # Draw chord progress if in highlight mode
        if self.highlighting_mode and self.chord_sequence:
            progress_text = self.font.render(
                f"Chord: {self.current_chord_idx}/{len(self.chord_sequence)}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(progress_text, (10, 190))

        # Display highlighted notes in play-along mode
        if self.highlighting_mode and self.highlighted_notes:
            highlighted = [
                self.get_note_name(note)
                for note in self.highlighted_notes
                if self.highlighted_notes[note]
            ]
            notes_str = "Highlighted: " + ", ".join(highlighted)
            notes_text = self.font.render(notes_str, True, (255, 255, 100))
            # Display chord progress
            progress_text = self.font.render(
                f"Chord: {self.current_chord_idx}/{len(self.chord_sequence)}",
                True,
                (255, 255, 255),
            )
            self.screen.blit(progress_text, (10, 220))
            self.screen.blit(notes_text, (10, 190))
            # Display currently highlighted notes
            if self.highlighted_notes:
                highlighted = [
                    self.get_note_name(note)
                    for note in self.highlighted_notes
                    if self.highlighted_notes[note]
                ]
                notes_str = "Highlighted: " + ", ".join(highlighted)
                notes_text = self.font.render(notes_str, True, (255, 255, 100))
                self.screen.blit(notes_text, (10, 220))
        # Display currently playing notes
        active_notes = [
            self.get_note_name(note)
            for note in self.active_notes
            if self.active_notes[note]
        ]
        if active_notes:
            notes_str = "Playing: " + ", ".join(active_notes[:8])
            if len(active_notes) > 8:
                notes_str += f" +{len(active_notes) - 8} more"
            notes_text = self.font.render(notes_str, True, (255, 255, 255))
            self.screen.blit(notes_text, (10, 160))
    def run(self, midi_file):
        """Main application loop"""
        if not self.load_midi(midi_file):
            print(f"Failed to load MIDI file: {midi_file}")
            return

        # Start playback automatically after loading the MIDI file
        self.play()

        clock = pygame.time.Clock()
        running = True

        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if not self.playing:
                            self.play()
                        else:
                            self.pause()
                    elif event.key == pygame.K_r:
                        self.stop()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Increase volume
                        self.volume = min(1.0, self.volume + 0.05)
                    elif event.key == pygame.K_MINUS:
                        # Decrease volume
                        self.volume = max(0.0, self.volume - 0.05)

            # Update state
            dt = clock.tick(60) / 1000.0  # Delta time in seconds
            self.update(dt)

            # Render
            self.screen.fill((40, 40, 40))
            self.draw_piano()
            self.render_ui()
            pygame.display.flip()

        pygame.quit()
    def handle_keyboard_events(self, events):
        """Handle keyboard events for piano playing"""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # Signal to quit

                # Navigation controls for highlight mode
                if event.key == pygame.K_RIGHT:
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
                elif event.key == pygame.K_SPACE:
                    if not self.playing:
                        self.play()
                    else:
                        self.pause()
                elif event.key == pygame.K_r:
                    self.stop()
                elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                    # Increase volume
                    self.volume = min(1.0, self.volume + 0.05)
                elif event.key == pygame.K_MINUS:
                    # Decrease volume
                    self.volume = max(0.0, self.volume - 0.05)

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
        # Draw playback status
        status = (
            "Playing"
            if self.playing and not self.paused
            else "Paused" if self.paused else "Stopped"
        )
        status_text = self.font.render(f"Status: {status}", True, (255, 255, 255))
        self.screen.blit(status_text, (10, 10))
        
        # Draw MIDI input status
        midi_status = "MIDI: Connected" if self.midi_input else "MIDI: Not Connected"
        midi_text = self.font.render(midi_status, True, (255, 255, 255))
        self.screen.blit(midi_text, (300, 10))


        # Draw current time
        time_text = self.font.render(
            f"Time: {self.current_time/1000:.2f}s", True, (255, 255, 255)
        )
        self.screen.blit(time_text, (10, 40))
        
        # Draw chord progress if in highlight mode
        if self.highlighting_mode:
            if self.chord_sequence:
                progress_text = self.font.render(
                    f"Chord: {self.current_chord_idx}/{len(self.chord_sequence)}",
                    True,
                    (255, 255, 255),
                )
                self.screen.blit(progress_text, (10, 190))
            if self.highlighted_notes:
                highlighted = [
                    self.get_note_name(note)
                    for note in self.highlighted_notes
                    if self.highlighted_notes[note]
                ]
                notes_str = "Highlighted: " + ", ".join(highlighted)
                notes_text = self.font.render(notes_str, True, (255, 255, 100))
                self.screen.blit(notes_text, (10, 220))
        
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
        
        # Display currently playing notes
        active_notes = [
            self.get_note_name(note)
            for note in self.active_notes
            if self.active_notes[note]
        ]
        if active_notes:
            notes_str = "Playing: " + ", ".join(active_notes[:8])
            if len(active_notes) > 8:
                notes_str += f" +{len(active_notes) - 8} more"
            notes_text = self.font.render(notes_str, True, (255, 255, 255))
            self.screen.blit(notes_text, (10, 160))

    def run(self, midi_file):
        """Main application loop"""
        if not self.load_midi(midi_file):
            print(f"Failed to load MIDI file: {midi_file}")
            return

        clock = pygame.time.Clock()
        running = True

        if self.mode == "playback":
            # Start playback automatically after loading the MIDI file
            self.play()

        while running:
            # Process MIDI input from external keyboard
            self.process_midi_input()
            
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        if self.mode == "playback":
                            if not self.playing:
                                self.play()
                            else:
                                self.pause()
                        # In play-along mode, space could be used for something else,
                        # or ignored, depending on desired behavior.
                        # For now, we'll just print a message.
                        elif self.mode == "play-along":
                            print("Space key not active in play-along mode.")
                    elif event.key == pygame.K_r:
                        self.stop()
                    elif event.key == pygame.K_PLUS or event.key == pygame.K_EQUALS:
                        # Increase volume
                        self.volume = min(1.0, self.volume + 0.05)
                    elif event.key == pygame.K_MINUS:
                        # Decrease volume
                        self.volume = max(0.0, self.volume - 0.05)
                        
            # Update state
            dt = clock.tick(60) / 1000.0  # Delta time in seconds
            if self.mode == "playback":
                self.update(dt)
            
            # Check if highlighted notes are played correctly
            self.check_highlighted_played()

            # Render
            self.screen.fill((40, 40, 40))
            self.draw_piano()
            self.render_ui()
            pygame.display.flip()

        pygame.quit()
    def handle_keyboard_events(self, events):
        """Handle keyboard events for piano playing"""
        for event in events:
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    return False  # Signal to quit

                # Navigation controls for highlight mode
                if event.key == pygame.K_RIGHT:
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
                elif event.key == pygame.K_SPACE:
                    if not self.playing:
                        self.play()
                    else:
                        self.pause()
                elif event.key == pygame.K_r:
                    self.stop()
                elif event.key in [pygame.K_PLUS, pygame.K_EQUALS]:
                    # Increase volume
                    self.volume = min(1.0, self.volume + 0.05)
                elif event.key == pygame.K_MINUS:
                    # Decrease volume
                    self.volume = max(0.0, self.volume - 0.05)


        return True  # Continue running


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MIDI Piano Visualizer")
    parser.add_argument("midi_file", nargs="?", help="Path to the MIDI file")
    parser.add_argument(
        "--mode",
        choices=["playback", "play-along"],
        default="playback",
        help="Mode: playback or play-along",
    )
    args = parser.parse_args()

    if not args.midi_file:
        print("Please provide a MIDI file as a command-line argument.")
        sys.exit(1)

    visualizer = PianoVisualizer(mode=args.mode)
    visualizer.run(args.midi_file)
