import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np  # Imported but not used in the current features
import argparse
from collections import defaultdict, Counter
import logging

# import yaml # Imported but not used - potentially for future config loading
from typing import Dict, List, Set, Optional, Any, Tuple

# import matplotlib.pyplot as plt # Imported but not used - potentially for future plotting
import io
import random


# --- Custom Exception ---
class MIDIAnalysisError(Exception):
    """Custom exception for MIDI analysis errors."""

    pass


# --- MIDI Parsing Logic ---
class AdvancedMIDIParser:
    """
    Enhanced MIDI file parsing with advanced analysis capabilities.
    Extracts note data, tempo, key/time signatures, and instrument changes.
    """

    def __init__(self):
        """Initialize advanced MIDI parser."""
        self.midi_analysis = self._get_default_analysis()
        # Initialize logging if not already configured by the main app
        if not logging.getLogger().hasHandlers():
            logging.basicConfig(
                level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
            )

    def _get_default_analysis(self) -> Dict[str, Any]:
        """Returns a dictionary with default values for MIDI analysis."""
        return {
            "total_notes": 0,
            "unique_notes": set(),
            "note_distribution": defaultdict(int),
            "note_duration_stats": {
                "min_duration": float("inf"),  # Seconds
                "max_duration": 0.0,  # Seconds
                "avg_duration": 0.0,  # Seconds
            },
            "tempo_changes": [],  # List of {"time_seconds", "tick", "tempo", "bpm"}
            "key_signature_changes": [],  # List of {"time_seconds", "tick", "key"}
            "time_signature_changes": [],  # List of {"time_seconds", "tick", "numerator", ...}
            "program_changes": defaultdict(
                list
            ),  # Dict[track_num, List[{"time_seconds", "tick", ...}]]
            "total_duration": 0.0,  # Seconds
            "ticks_per_beat": None,
            "filename": None,
            "tracks": [],  # List of track names or numbers
        }

    def parse_midi_file(self, midi_file_path: str) -> Dict[str, Any]:
        """
        Parse a MIDI file and perform comprehensive analysis.

        :param midi_file_path: Path to the MIDI file.
        :return: Dictionary containing detailed MIDI file analysis results.
        :raises MIDIAnalysisError: If the file cannot be found or parsed.
        """
        try:
            midi_file = mido.MidiFile(midi_file_path)
            return self._parse_midi_data(midi_file)
        except FileNotFoundError as e:
            logging.error(f"MIDI file not found: {midi_file_path}")
            raise MIDIAnalysisError(f"MIDI file not found: {midi_file_path}") from e
        except mido.KeySignatureError as e:
            logging.error(f"MIDI file parsing error (Key Signature): {e}")
            raise MIDIAnalysisError(f"Error parsing Key Signature in MIDI file: {e}") from e
        except Exception as e:
            logging.exception(f"Unexpected error parsing MIDI file '{midi_file_path}': {e}")
            raise MIDIAnalysisError(f"Error parsing MIDI file: {e}") from e

    def _parse_midi_data(self, midi_file: mido.MidiFile) -> Dict[str, Any]:
        """
        Parses MIDI file data and populates the analysis dictionary.

        :param midi_file: The loaded MIDI file object.
        :return: Dictionary containing detailed MIDI file analysis results.
        """
        self.midi_analysis["ticks_per_beat"] = midi_file.ticks_per_beat
        self.midi_analysis["filename"] = midi_file.filename
        absolute_tick_max = 0

        for track_num, track in enumerate(midi_file.tracks):
            self.midi_analysis["tracks"].append(track.name or f"Track {track_num}")
            absolute_tick_track = 0  # Ticks accumulated in this track
            active_notes_ticks = {}  # Tracks active notes {note: start_tick}

            for msg in track:
                absolute_tick_track += msg.time
                if msg.is_meta:
                    if msg.type == "set_tempo":
                        tempo = msg.tempo
                        bpm = mido.tempo2bpm(tempo)
                        self.midi_analysis["tempo_changes"].append({
                            "time_seconds": mido.tick2second(absolute_tick_track, midi_file.ticks_per_beat, tempo),
                            "tick": absolute_tick_track,
                            "tempo": tempo,
                            "bpm": bpm,
                        })
                    elif msg.type == "key_signature":
                        self.midi_analysis["key_signature_changes"].append({
                            "time_seconds": mido.tick2second(absolute_tick_track, midi_file.ticks_per_beat, tempo),
                            "tick": absolute_tick_track,
                            "key": msg.key,
                        })
                    elif msg.type == "time_signature":
                        self.midi_analysis["time_signature_changes"].append({
                            "time_seconds": mido.tick2second(absolute_tick_track, midi_file.ticks_per_beat, tempo),
                            "tick": absolute_tick_track,
                            "numerator": msg.numerator,
                            "denominator": msg.denominator,
                        })
                elif msg.type == "note_on" and msg.velocity > 0:
                    active_notes_ticks[msg.note] = absolute_tick_track
                    self.midi_analysis["unique_notes"].add(msg.note)
                    self.midi_analysis["note_distribution"][msg.note] += 1
                    self.midi_analysis["total_notes"] += 1
                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    if msg.note in active_notes_ticks:
                        note_start_tick = active_notes_ticks.pop(msg.note)
                        note_duration_ticks = absolute_tick_track - note_start_tick
                        note_duration_seconds = mido.tick2second(note_duration_ticks, midi_file.ticks_per_beat, tempo)
                        self.midi_analysis["note_duration_stats"]["min_duration"] = min(
                            self.midi_analysis["note_duration_stats"]["min_duration"], note_duration_seconds
                        )
                        self.midi_analysis["note_duration_stats"]["max_duration"] = max(
                            self.midi_analysis["note_duration_stats"]["max_duration"], note_duration_seconds
                        )
                        self.midi_analysis["total_duration"] += note_duration_seconds

            absolute_tick_max = max(absolute_tick_max, absolute_tick_track)

        if self.midi_analysis["total_notes"] > 0:
            self.midi_analysis["note_duration_stats"]["avg_duration"] = (
                self.midi_analysis["total_duration"] / self.midi_analysis["total_notes"]
            )

        return self.midi_analysis

    def generate_midi_analysis_report(self) -> str:
        """
        Generate a formatted text report summarizing the MIDI file analysis.

        :return: String containing the analysis report.
        """
        analysis = self.midi_analysis
        if not analysis or analysis.get("filename") is None:
            return "No MIDI analysis data available."

        report = f"### MIDI File Analysis Report: {analysis.get('filename', 'N/A')} ###\n\n"
        report += self._generate_general_info(analysis)
        report += self._generate_note_info(analysis)
        report += self._generate_duration_stats(analysis)
        report += self._generate_tempo_changes(analysis)
        report += self._generate_time_signature_changes(analysis)
        report += self._generate_key_signature_changes(analysis)
        report += self._generate_program_changes(analysis)
        return report

    def _generate_general_info(self, analysis):
        info = f"Total Duration: {analysis.get('total_duration', 'N/A'):.2f} seconds\n"
        info += f"Ticks Per Beat: {analysis.get('ticks_per_beat', 'N/A')}\n"
        info += f"Number of Tracks: {len(analysis.get('tracks', []))}\n"
        return info

    def _generate_note_info(self, analysis):
        info = f"Total Notes Played: {analysis['total_notes']}\n"
        info += f"Unique Notes Used: {len(analysis['unique_notes'])}\n"

        if analysis["unique_notes"]:
            min_note = min(analysis["unique_notes"])
            max_note = max(analysis["unique_notes"])
            info += f"Note Range: {min_note} ({self._get_note_name_static(min_note)}) - {max_note} ({self._get_note_name_static(max_note)})\n\n"

            # Top 5 most frequent notes
            sorted_notes = sorted(analysis["note_distribution"].items(), key=lambda item: item[1], reverse=True)[:5]
            info += "Most Frequent Notes (Top 5):\n"
            for note, count in sorted_notes:
                note_name = self._get_note_name_static(note)
                info += f"  Note {note} ({note_name}): {count} times\n"
        else:
            info += "Note Range: N/A (No notes found)\n\n"
            info += "Most Frequent Notes: No notes found.\n"

        return info + "\n"

    def _generate_duration_stats(self, analysis):
        duration_stats = analysis["note_duration_stats"]
        min_dur_str = f"{duration_stats['min_duration']:.4f}" if duration_stats["min_duration"] != float("inf") else "N/A"
        max_dur_str = f"{duration_stats['max_duration']:.4f}" if duration_stats["max_duration"] is not None else "N/A"
        avg_dur_str = f"{duration_stats['avg_duration']:.4f}" if duration_stats["avg_duration"] is not None else "N/A"

        stats = "Note Duration Statistics (in seconds):\n"
        stats += f"  Min Duration: {min_dur_str}\n"
        stats += f"  Max Duration: {max_dur_str}\n"
        stats += f"  Avg Duration: {avg_dur_str}\n\n"
        return stats

    def _generate_tempo_changes(self, analysis):
        changes = "Tempo Changes (BPM):\n"
        if analysis["tempo_changes"]:
            last_bpm = None
            for change in analysis["tempo_changes"]:
                bpm = change.get("bpm", mido.tempo2bpm(change["tempo"]))
                if bpm != last_bpm:
                    changes += f"  Tick {change['tick']} ({change['time_seconds']:.2f}s): {bpm:.2f} BPM\n"
                    last_bpm = bpm
        else:
            changes += f"  No tempo changes detected (Default: {mido.tempo2bpm(500000):.2f} BPM).\n"
        return changes + "\n"

    def _generate_time_signature_changes(self, analysis):
        changes = "Time Signature Changes:\n"
        if analysis["time_signature_changes"]:
            last_sig = None
            for change in analysis["time_signature_changes"]:
                current_sig = f"{change['numerator']}/{change['denominator']}"
                if current_sig != last_sig:  # Avoid redundant reports
                    changes += f"  Tick {change['tick']} ({change['time_seconds']:.2f}s): {current_sig}\n"
                    last_sig = current_sig
        else:
            changes += "  No time signature changes detected (Assumed 4/4).\n"
        return changes + "\n"

    def _generate_key_signature_changes(self, analysis):
        changes = "Key Signature Changes:\n"
        if analysis["key_signature_changes"]:
            last_key = None
            for change in analysis["key_signature_changes"]:
                if change["key"] != last_key:  # Avoid redundant reports
                    changes += f"  Tick {change['tick']} ({change['time_seconds']:.2f}s): {change['key']}\n"
                    last_key = change["key"]
        else:
            changes += "  No key signature changes detected.\n"
        return changes + "\n"

    def _generate_program_changes(self, analysis):
        changes = "Program (Instrument) Changes:\n"
        if analysis["program_changes"]:
            for track_num, changes_list in sorted(analysis["program_changes"].items()):
                track_name = (
                    analysis["tracks"][track_num]
                    if track_num < len(analysis["tracks"])
                    else f"Track {track_num}"
                )
                changes += f"  {track_name}:\n"
                last_prog = -1  # Track initial program too
                for change in changes_list:
                    if change["program"] != last_prog:
                        changes += f"    Tick {change['tick']} ({change['time_seconds']:.2f}s), Ch {change['channel']}: Prog {change['program']}\n"
                        last_prog = change["program"]
        else:
            changes += "  No program changes detected.\n"
        return changes

    @staticmethod
    def _get_note_name_static(note: int) -> str:
        """Static helper to get note name (e.g., C4) from MIDI number."""
        if not (0 <= note <= 127):
            return "??"
        note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (note // 12) - 1  # Standard MIDI octave (C4 = 60)
        note_in_octave = note_names[note % 12]
        return f"{note_in_octave}{octave}"


# --- Base Piano Trainer Class (Minimal) ---
class PianoTrainer:
    """Base class placeholder for the Piano Trainer application."""

    def __init__(self):
        logging.debug("Initializing Base PianoTrainer...")
        self.config_manager = None
        self.mode = None
        self.midi_analysis = None
        self.current_learning_challenge = None
        self.running = False
        logging.debug("Base PianoTrainer Initialized.")

    def _render_ui(self):
        pass  # Intended to be overridden

    def run(self, mode=None, midi_file=None):
        logging.debug("Running Base PianoTrainer...")
        self.mode = mode
        self.running = True
        # Basic loop structure would go here if base class had functionality
        logging.debug("Base PianoTrainer run finished.")


# --- Enhanced UI and Core Logic ---
def enhance_piano_trainer_ui(BasePianoTrainer):
    """
    Decorator function (used as inheritance source) to enhance the PianoTrainer
    with Pygame UI, MIDI handling, challenge integration, and performance tracking.
    """
    class EnhancedPianoTrainerUI(BasePianoTrainer):
        """The main application class with enhanced features."""

        def __init__(self, *args, **kwargs):
            logging.info("Initializing EnhancedPianoTrainerUI...")
            # Initialize Pygame modules
            try:
                pygame.init()
                pygame.midi.init()
                pygame.font.init()
            except Exception as e:
                logging.exception("Failed to initialize Pygame or its modules.")
                raise RuntimeError("Pygame initialization failed.") from e

            # Screen setup
            self.screen_width = 1450
            self.screen_height = 700
            try:
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
                pygame.display.set_caption("Enhanced Piano Trainer")
            except pygame.error as e:
                logging.exception("Failed to set up Pygame display.")
                pygame.quit()
                raise RuntimeError("Display setup failed.") from e

            # Call parent __init__ (though it's minimal)
            super().__init__(*args, **kwargs)

            # --- Core Components ---
            self.midi_parser = AdvancedMIDIParser()
            self.performance_history = []

            # --- Piano Visualization ---
            self.white_key_width = 40
            self.white_key_height = 200
            self.black_key_width = 26
            self.black_key_height = 120
            self.first_note = 36  # C2
            self.last_note = 96  # C7
            self.total_keys = self.last_note - self.first_note + 1
            num_white_keys = sum(
                not self.is_black_key(n)
                for n in range(self.first_note, self.last_note + 1)
            )
            required_piano_width = num_white_keys * self.white_key_width
            self.piano_start_x = max(
                20, (self.screen_width - required_piano_width) // 2
            )  # Center piano
            self.piano_start_y = 480  # Position lower on screen

            # --- Note State Tracking ---
            self.active_notes = defaultdict(lambda: False)  # MIDI Input state

            # --- Font Setup ---
            try:
                self.font = pygame.font.SysFont("Arial", 24)
                self.small_font = pygame.font.SysFont("Arial", 18)
                self.key_font = pygame.font.SysFont("Arial", 12)
                self.title_font = pygame.font.SysFont("Arial", 36, bold=True)
                self.report_font = pygame.font.SysFont(
                    "Courier New", 14
                )  # Monospace for reports
            except Exception:
                logging.warning(
                    "Arial/Courier New font not found, using Pygame default."
                )
                self.font = pygame.font.Font(None, 30)
                self.small_font = pygame.font.Font(None, 24)
                self.key_font = pygame.font.Font(None, 18)
                self.title_font = pygame.font.Font(None, 40)
                self.report_font = pygame.font.Font(None, 20)  # Default fallback

            # Note names cache
            self.note_names_map = {
                i: AdvancedMIDIParser._get_note_name_static(i) for i in range(128)
            }

            # --- MIDI Input ---
            self.midi_input = None
            self.midi_device_name = None  # <<< Store device name here
            self.setup_midi_input()

            # --- UI Colors ---
            self.colors = {
                "background_gradient_start": (25, 25, 40),
                "background_gradient_end": (50, 50, 70),
                "text_primary": (230, 230, 255),
                "text_secondary": (180, 180, 210),
                "text_highlight": (120, 230, 120),
                "text_error": (255, 100, 100),
                "piano_white_key": (250, 250, 250),
                "piano_black_key": (30, 30, 30),
                "piano_white_key_pressed": (170, 170, 255),
                "piano_black_key_pressed": (100, 100, 200),
                "key_border": (60, 60, 60),
                "key_text": (0, 0, 0),
                "key_text_black": (200, 200, 200),  # Text on black keys
            }

            # --- Application State ---
            self.clock = pygame.time.Clock()
            self.midi_analysis_report_str = None
            self.challenge_difficulty = "intermediate"  # Default

            logging.info("EnhancedPianoTrainerUI Initialized successfully.")

        def run(self, mode: Optional[str] = "freestyle", midi_file: Optional[str] = None, difficulty: str = "intermediate") -> None:
            """
            Main application loop. Handles mode switching, event processing,
            state updates, and rendering.
            """
            logging.info(f"Starting Piano Trainer. Mode: '{mode}', MIDI file: '{midi_file}', Difficulty: '{difficulty}'")
            self.mode = mode
            self.running = True
            self.challenge_difficulty = difficulty

            try:
                # --- Initial Mode Setup ---
                if midi_file:
                    logging.info(f"MIDI file provided: {midi_file}")
                    try:
                        self.midi_analysis = self.midi_parser.parse_midi_file(midi_file)
                        self.midi_analysis_report_str = self.midi_parser.generate_midi_analysis_report()
                        print("\n--- MIDI Analysis Report ---")
                        print(self.midi_analysis_report_str)  # Also print to console
                        print("--------------------------\n")
                        if self.mode == "freestyle":  # Switch mode if only MIDI file provided
                            self.mode = "analysis_view"
                            logging.info("Switching mode to 'analysis_view' due to provided MIDI file.")
                    except MIDIAnalysisError as e:
                        logging.error(f"Error during MIDI analysis: {e}")
                        self.midi_analysis_report_str = f"Error loading MIDI:\n{e}"
                        self.mode = "freestyle"  # Fallback mode on error

                # Setup based on the final mode
                self._setup_current_mode()

                # --- Main Loop ---
                while self.running:
                    self._handle_events()
                    self.process_midi_input()
                    self._render_ui()
                    pygame.display.flip()
                    self.clock.tick(60)  # Target 60 FPS

            except Exception as e:
                logging.exception("Unexpected error in main loop.")
            finally:
                self._cleanup()

        def _setup_current_mode(self):
            """Initializes state based on the current self.mode."""
            logging.info(f"Setting up mode: {self.mode}")
            if self.mode == "analysis_view":
                if not self.midi_analysis_report_str:
                    logging.warning(
                        "Analysis view mode selected but no analysis available."
                    )
            elif self.mode != "freestyle":
                logging.warning(f"Unknown mode '{self.mode}'. Defaulting to freestyle.")
                self.mode = "freestyle"

        def _handle_events(self):
            """Processes Pygame events like quitting or key presses."""
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        self.running = False

        def _render_ui(self) -> None:
            """Draws all UI elements onto the screen."""
            # 1. Background
            self._draw_background()

            # 2. Title / Mode Display
            self._draw_title()

            # 3. Main Content Area (Analysis / Freestyle Msg)
            self._draw_main_content()

            # 4. Piano Keyboard
            self.draw_piano()

            # 5. Status Bar (MIDI connection)
            self._draw_status_bar()

        def _draw_background(self):
            """Draws a gradient background."""
            screen_rect = self.screen.get_rect()
            start_color = self.colors["background_gradient_start"]
            end_color = self.colors["background_gradient_end"]
            for y in range(screen_rect.height):
                interp = y / screen_rect.height
                r = int(start_color[0] * (1 - interp) + end_color[0] * interp)
                g = int(start_color[1] * (1 - interp) + end_color[1] * interp)
                b = int(start_color[2] * (1 - interp) + end_color[2] * interp)
                pygame.draw.line(self.screen, (r, g, b), (0, y), (screen_rect.width, y))

        def _draw_title(self):
            """Draws the main application title and current mode."""
            mode_str = self.mode.replace("_", " ").title()
            title_text_str = f"Piano Trainer - {mode_str}"
            title_surf = self.title_font.render(
                title_text_str, True, self.colors["text_primary"]
            )
            title_rect = title_surf.get_rect(center=(self.screen_width // 2, 40))
            self.screen.blit(title_surf, title_rect)

        def _draw_main_content(self):
            """Draws content specific to the current mode."""
            content_y_start = 90
            content_x_start = 20
            if self.mode == "analysis_view" and self.midi_analysis_report_str:
                self._extracted_from__draw_main_content_436(content_y_start)
            elif self.mode == "freestyle":
                info_surf = self.font.render(
                    "Freestyle Mode - Play freely!", True, self.colors["text_highlight"]
                )
                self.screen.blit(info_surf, (content_x_start, content_y_start))
                hint_surf = self.small_font.render(
                    "Q/Esc: Quit",
                    True,
                    self.colors["text_secondary"],
                )
                self.screen.blit(hint_surf, (content_x_start, content_y_start + 40))

            else:  # Fallback for unknown mode or error state
                error_surf = self.font.render(
                    f"Current Mode: {self.mode}", True, self.colors["text_error"]
                )
                self.screen.blit(error_surf, (content_x_start, content_y_start))

        def _extracted_from__draw_main_content_436(self, content_y_start):
            # Display scrollable/paginated report (simple version: top lines)
            y_offset = content_y_start
            max_report_width = self.screen_width * 0.8
            report_x = (self.screen_width - max_report_width) / 2
            line_height = self.report_font.get_linesize()
            max_lines_display = 20  # Fit more lines with monospace font

            report_lines = self.midi_analysis_report_str.split("\n")
            for i, line in enumerate(report_lines[:max_lines_display]):
                if not line.strip() and i > 0 and not report_lines[i - 1].strip():
                    continue  # Skip multiple blank lines
                line_surf = self.report_font.render(
                    line, True, self.colors["text_secondary"]
                )
                # Simple truncate if needed
                if line_surf.get_width() > max_report_width:
                    avg_char_width = (
                        line_surf.get_width() / len(line) if len(line) > 0 else 10
                    )
                    max_chars = int(max_report_width / avg_char_width) - 3
                    line = line[:max_chars] + "..."
                    line_surf = self.report_font.render(
                        line, True, self.colors["text_secondary"]
                    )
                self.screen.blit(line_surf, (report_x, y_offset + i * line_height))

            if len(report_lines) > max_lines_display:
                more_text = self.small_font.render(
                    "... (Full report printed to console)",
                    True,
                    self.colors["text_highlight"],
                )
                self.screen.blit(
                    more_text,
                    (report_x, y_offset + max_lines_display * line_height + 5),
                )

        def _draw_status_bar(self):
            """Draws the bottom status bar (e.g., MIDI connection)."""
            status_text = (
                f"MIDI: {self.midi_device_name if self.midi_input else 'Not Connected'}"
            )
            status_color = (
                self.colors["text_highlight"]
                if self.midi_input
                else self.colors["text_error"]
            )
            status_surf = self.small_font.render(status_text, True, status_color)
            status_rect = status_surf.get_rect(
                bottomright=(self.screen_width - 15, self.screen_height - 15)
            )
            self.screen.blit(status_surf, status_rect)

        def setup_midi_input(self):
            """Detects and connects to the first available MIDI input device."""
            self.midi_input = None
            self.midi_device_name = None  # <<< Reset stored name
            input_id = -1
            found_device_name = "Unknown Device"  # Temporary name storage

            logging.info("Searching for MIDI input devices...")
            try:
                device_count = pygame.midi.get_count()
                for i in range(device_count):
                    device_info = pygame.midi.get_device_info(i)
                    if device_info is None:
                        logging.warning(
                            f"Could not get info for MIDI device index {i}."
                        )
                        continue

                    is_input = device_info[2]
                    if is_input:
                        name_bytes = device_info[1]
                        try:
                            found_device_name = name_bytes.decode(
                                "utf-8", errors="replace"
                            )
                        except Exception as decode_err:
                            logging.warning(
                                f"Could not decode name for MIDI device {i}: {decode_err}"
                            )
                            found_device_name = f"Device {i} (Decode Error)"

                        logging.info(
                            f"Found MIDI Input Device #{i}: {found_device_name}"
                        )
                        input_id = i
                        break  # Use the first input device found

            except pygame.midi.MidiException as e:
                logging.error(f"Error enumerating MIDI devices: {e}", exc_info=True)
                print(f"--- Error Accessing MIDI System: {e} ---")
                # No MIDI available, proceed without it
                return
            except Exception as e:
                logging.exception("Unexpected error during MIDI device enumeration.")
                # Proceed without MIDI
                return

            # --- Attempt connection ---
            if input_id != -1:
                try:
                    self.midi_input = pygame.midi.Input(input_id)
                    self.midi_device_name = (
                        found_device_name  # <<< STORE the name on successful connection
                    )
                    logging.info(
                        f"Successfully connected to MIDI device: {self.midi_device_name}"
                    )  # <<< Log stored name
                except pygame.midi.MidiException as e:
                    logging.error(
                        f"Could not open MIDI device ID {input_id} ({found_device_name}): {e}",
                        exc_info=True,
                    )
                    self.midi_input = None
                    self.midi_device_name = None  # Ensure reset on failure
                except Exception as e:
                    logging.exception(
                        f"Unexpected error opening MIDI device {input_id}."
                    )
                    self.midi_input = None
                    self.midi_device_name = None

            # --- Log final status ---
            if self.midi_input is None:
                logging.warning("No suitable MIDI input device connected or opened.")
                print("--- No MIDI Keyboard Detected ---")
                print(
                    "Connect a MIDI keyboard and restart, or use other input methods (if implemented)."
                )

        def process_midi_input(self):
            """Reads and processes incoming MIDI messages from the connected device."""
            if self.midi_input is None or not self.midi_input.poll():
                return  # No device or no new events

            try:
                # Read multiple events if available
                midi_events = self.midi_input.read(128)  # Read up to 128 events

                for event in midi_events:
                    data, timestamp = (
                        event  # [[status, data1, data2, data3], timestamp]
                    )
                    status = data[0]
                    note = data[1]
                    velocity = data[2]

                    # Note On (Status 0x90-0x9F, velocity > 0)
                    if 144 <= status <= 159 and velocity > 0:
                        logging.debug(f"MIDI IN: Note On - Note={note}, Vel={velocity}")
                        self.active_notes[note] = True

                    elif (128 <= status <= 143) or (
                        144 <= status <= 159 and velocity == 0
                    ):
                        logging.debug(f"MIDI IN: Note Off - Note={note}")
                        self.active_notes[note] = False

                    elif 176 <= status <= 191:
                        controller = data[1]
                        if controller == 64:  # Sustain Pedal
                            value = data[2]
                            sustain_on = value >= 64
                            logging.debug(
                                f"MIDI IN: Sustain Pedal {'On' if sustain_on else 'Off'} (Val={value})"
                            )

            except pygame.midi.MidiException as e:
                logging.error(
                    f"MIDI Read Error: {e}. Attempting to ignore.", exc_info=True
                )
            except Exception as e:
                logging.exception("Unexpected error processing MIDI input.")

        def is_black_key(self, note: int) -> bool:
            """Checks if a MIDI note number corresponds to a black piano key."""
            return (note % 12) in [1, 3, 6, 8, 10]

        def count_white_keys_before(self, note: int) -> int:
            """Counts white keys from the piano's start note up to (not including) the given note."""
            count = 0
            if note < self.first_note:
                return 0
            for n in range(self.first_note, note):
                if not self.is_black_key(n):
                    count += 1
            return count

        def get_note_name(self, note: int) -> str:
            """Gets the standard note name (e.g., C4) from the cached map."""
            return self.note_names_map.get(note, "??")

        def get_key_rect(self, note: int) -> Optional[pygame.Rect]:
            """Calculates the screen rectangle (position and size) for a given MIDI note key."""
            if not (self.first_note <= note <= self.last_note):
                return None  # Note is outside the displayed piano range

            if self.is_black_key(note):
                # Black keys are positioned relative to the white key to their left
                white_key_index_left = self.count_white_keys_before(note)
                # Center the black key over the gap (or slightly offset)
                x = (
                    self.piano_start_x
                    + (white_key_index_left * self.white_key_width)
                    - (self.black_key_width // 2)
                )
                return pygame.Rect(
                    x, self.piano_start_y, self.black_key_width, self.black_key_height
                )
            else:
                # White keys are positioned sequentially based on index
                white_key_index = self.count_white_keys_before(note)
                x = self.piano_start_x + white_key_index * self.white_key_width
                # Use width-1 to leave a 1px gap for the border drawing
                return pygame.Rect(
                    x,
                    self.piano_start_y,
                    self.white_key_width - 1,
                    self.white_key_height,
                )

        def draw_piano(self):
            """Draws the visual piano keyboard onto the screen."""
            # --- Draw White Keys ---
            for note in range(self.first_note, self.last_note + 1):
                if not self.is_black_key(note):
                    rect = self.get_key_rect(note)
                    if rect:
                        color = self.colors["piano_white_key"]
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(
                            self.screen, self.colors["key_border"], rect, 1
                        )  # Draw border

                        # Draw note name
                        note_name = self.get_note_name(note)
                        name_surf = self.key_font.render(
                            note_name, True, self.colors["key_text"]
                        )
                        name_rect = name_surf.get_rect(
                            centerx=rect.centerx, bottom=rect.bottom - 5
                        )
                        self.screen.blit(name_surf, name_rect)

            # --- Draw Black Keys (on top) ---
            for note in range(self.first_note, self.last_note + 1):
                if self.is_black_key(note):
                    rect = self.get_key_rect(note)
                    if rect:
                        color = self.colors["piano_black_key"]
                        pygame.draw.rect(self.screen, color, rect)

        def _cleanup(self):
            """Properly shuts down Pygame and MIDI resources."""
            logging.info("Cleaning up resources...")
            # Close MIDI input if open
            if self.midi_input:
                try:
                    self.midi_input.close()
                    logging.info("MIDI input closed.")
                except Exception as e:
                    logging.exception("Error closing MIDI input.")
            # Quit Pygame modules
            pygame.midi.quit()
            pygame.font.quit()
            pygame.quit()
            logging.info("Pygame quit.")
            print("\n--- Piano Trainer Exited ---")

    return EnhancedPianoTrainerUI


# Argument Parsing
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Piano Trainer with Pygame")
    parser.add_argument(
        "-m",
        "--mode",
        help="Run mode: freestyle, learning, analysis_view",
        type=str,
        default="freestyle",
        choices=["freestyle", "learning", "analysis_view"],
    )
    parser.add_argument(
        "-f",
        "--midi",
        help="Path to MIDI file for analysis or reference",
        type=str,
        default=None,
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        help="Learning challenge difficulty",
        type=str,
        default="intermediate",
        choices=["beginner", "intermediate", "advanced"],
    )

    args = parser.parse_args()

    try:
        EnhancedTrainer = enhance_piano_trainer_ui(PianoTrainer)
        trainer_app = EnhancedTrainer()
        trainer_app.run(mode=args.mode, midi_file=args.midi, difficulty=args.difficulty)
    except RuntimeError as e:
        print(f"\nApplication failed to start: {e}", file=sys.stderr)
        logging.critical(f"Application runtime error: {e}", exc_info=True)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupt received, exiting...")
        logging.info("Keyboard interrupt received. Exiting.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        logging.critical("Unhandled exception in main.", exc_info=True)
        sys.exit(1)