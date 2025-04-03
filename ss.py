import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np
import argparse
from collections import defaultdict, Counter, deque  # Added deque
import logging
from typing import Dict, List, Set, Optional, Any, Tuple
import io
import random

# --- Constants ---
# Learning Mode Settings
FALLING_NOTE_SPEED = 150  # Pixels per second
TARGET_LINE_Y_OFFSET = 50  # How far above the piano keys the target line is
HIT_WINDOW_MS = 150       # Time window (milliseconds) around target time to count as a hit
PREP_TIME_SEC = 3         # Seconds before the first note starts falling


# --- Custom Exception ---
class MIDIAnalysisError(Exception):
    """Custom exception for MIDI analysis errors."""
    pass


# --- Note/Chord Generation ---
class MusicTheory:
    """Helper class for basic music theory elements."""
    SCALE_INTERVALS = {
        "major": [0, 2, 4, 5, 7, 9, 11, 12], # W-W-H-W-W-W-H
        "natural_minor": [0, 2, 3, 5, 7, 8, 10, 12], # W-H-W-W-H-W-W
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11, 12], # W-H-W-W-H-WH-H
        "chromatic": list(range(13)),
    }
    CHORD_INTERVALS = {
        "maj": [0, 4, 7],       # Major Triad
        "min": [0, 3, 7],       # Minor Triad
        "dim": [0, 3, 6],       # Diminished Triad
        "aug": [0, 4, 8],       # Augmented Triad
        "maj7": [0, 4, 7, 11],  # Major 7th
        "min7": [0, 3, 7, 10],  # Minor 7th
        "dom7": [0, 4, 7, 10],  # Dominant 7th
        "dim7": [0, 3, 6, 9],   # Diminished 7th
        "sus4": [0, 5, 7],      # Suspended 4th
    }

    @staticmethod
    def generate_scale(root_note: int, scale_type: str = "major", octaves: int = 1) -> List[int]:
        """Generates MIDI notes for a scale."""
        intervals = MusicTheory.SCALE_INTERVALS.get(scale_type.lower())
        if not intervals:
            logging.warning(f"Unknown scale type: {scale_type}. Defaulting to major.")
            intervals = MusicTheory.SCALE_INTERVALS["major"]

        scale_notes = []
        for o in range(octaves):
            for interval in intervals[:-1]: # Exclude octave repetition within loop
                note = root_note + (o * 12) + interval
                if 0 <= note <= 127:
                    scale_notes.append(note)
        # Add the final octave note
        final_note = root_note + (octaves * 12)
        if 0 <= final_note <= 127:
             scale_notes.append(final_note)

        return scale_notes

    @staticmethod
    def generate_chord(root_note: int, chord_type: str = "maj") -> List[int]:
        """Generates MIDI notes for a chord."""
        intervals = MusicTheory.CHORD_INTERVALS.get(chord_type.lower())
        if not intervals:
            logging.warning(f"Unknown chord type: {chord_type}. Defaulting to major triad.")
            intervals = MusicTheory.CHORD_INTERVALS["maj"]

        chord_notes = []
        for interval in intervals:
            note = root_note + interval
            if 0 <= note <= 127:
                chord_notes.append(note)
        return chord_notes


# --- Falling Note Class ---
class FallingNote:
    """Represents a note to be played in learning mode."""
    def __init__(self, note: int, start_time_sec: float, duration_sec: float, target_y: int, screen_height: int):
        self.note = note
        self.start_time_sec = start_time_sec
        self.end_time_sec = start_time_sec + duration_sec
        self.duration_sec = duration_sec
        self.target_y = target_y
        self.start_y = target_y - (FALLING_NOTE_SPEED * start_time_sec)
        self.current_y = self.start_y - (screen_height + 100) # Start way off screen
        self.rect = None
        self.state = "upcoming" # upcoming, active, hit, missed, invalid_key
        self.hit_time_ms: Optional[int] = None
        self._note_name_cache = AdvancedMIDIParser._get_note_name_static(note)

    def update(self, current_time_sec: float, key_rect_map: Dict[int, pygame.Rect]):
        """Update the note's vertical position and state."""
        key_rect = key_rect_map.get(self.note)
        if not key_rect:
            if self.state != "invalid_key":
                self.state = "invalid_key"
            return

        time_to_hit_sec = self.start_time_sec - current_time_sec
        self.current_y = self.target_y - (time_to_hit_sec * FALLING_NOTE_SPEED)

        is_black = (self.note % 12) in [1, 3, 6, 8, 10]
        width = key_rect.width
        height = max(10, self.duration_sec * FALLING_NOTE_SPEED)

        center_x = key_rect.centerx
        self.rect = pygame.Rect(center_x - width // 2, self.current_y - height, width, height)

        if self.state in ["hit", "missed", "invalid_key"]:
            return

        current_time_ms = int(current_time_sec * 1000)
        start_time_ms = int(self.start_time_sec * 1000)

        if current_time_ms > (start_time_ms + HIT_WINDOW_MS):
            self.state = "missed"
        elif abs(current_time_ms - start_time_ms) <= HIT_WINDOW_MS:
            self.state = "active"
        else:
            self.state = "upcoming"

    def draw(self, screen, colors, font):
        """Draw the falling note."""
        if self.state == "invalid_key" or not self.rect or self.current_y > screen.get_height() + 50 or self.current_y < -self.rect.height:
             return

        color = colors["falling_note_upcoming"]
        border_color = colors["falling_note_border"]
        border = 1

        if self.state == "active": color = colors["falling_note_active"]
        elif self.state == "hit": color = colors["falling_note_hit"]; border = 0
        elif self.state == "missed": color = colors["falling_note_missed"]; border = 0

        pygame.draw.rect(screen, color, self.rect, border)
        if border > 0: pygame.draw.rect(screen, border_color, self.rect, 1)

        if self.rect.height > font.get_height() * 1.2:
            text_surf = font.render(self._note_name_cache, True, colors["key_text"])
            text_rect = text_surf.get_rect(center=self.rect.center)
            if text_rect.width < self.rect.width:
                 screen.blit(text_surf, text_rect)

    def check_hit(self, played_note: int, play_time_ms: int) -> bool:
        """Check if this note was hit correctly."""
        if self.state in ["upcoming", "active"]:
            start_time_ms = int(self.start_time_sec * 1000)
            time_diff = abs(play_time_ms - start_time_ms)
            if played_note == self.note and time_diff <= HIT_WINDOW_MS:
                self.state = "hit"
                self.hit_time_ms = play_time_ms
                logging.debug(f"Hit! Note {self.note} ({self._note_name_cache}). Time diff: {play_time_ms - start_time_ms} ms")
                return True
        return False


# =========================================================
# MODIFIED AdvancedMIDIParser Class with Revised Logic
# =========================================================
class AdvancedMIDIParser:
    """ Enhanced MIDI file parsing with overlap handling. """

    def __init__(self):
        self.midi_analysis = self._get_default_analysis()
        # Logging configured by main app

    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "total_notes": 0, "unique_notes": set(), "note_distribution": defaultdict(int),
            "note_duration_stats": {"min_duration": float("inf"), "max_duration": 0.0, "avg_duration": 0.0},
            "tempo_changes": [], "key_signature_changes": [], "time_signature_changes": [],
            "program_changes": defaultdict(list), "total_duration": 0.0, "ticks_per_beat": None,
            "filename": None, "tracks": [], "default_tempo": 500000,
            "timed_notes": [] # List of {"note", "start_sec", "duration_sec", "velocity", "track", "channel"}
        }

    def parse_midi_file(self, midi_file_path: str) -> Dict[str, Any]:
        try:
            return self._extracted_from_parse_midi_file_3(midi_file_path)
        except FileNotFoundError as e:
            logging.error(f"MIDI file not found: {midi_file_path}")
            raise MIDIAnalysisError(f"MIDI file not found: {midi_file_path}") from e
        except Exception as e:
            logging.exception(f"Unexpected error parsing MIDI file '{midi_file_path}': {e}")
            raise MIDIAnalysisError(f"Error parsing MIDI file: {e}") from e

    # TODO Rename this here and in `parse_midi_file`
    def _extracted_from_parse_midi_file_3(self, midi_file_path):
        self.midi_analysis = self._get_default_analysis() # Reset analysis
        logging.debug(f"Attempting to parse MIDI file: {midi_file_path}")
        midi_file = mido.MidiFile(midi_file_path)
        logging.debug(f"Mido opened file. Type: {midi_file.type}, Length: {midi_file.length:.2f}s, Ticks/Beat: {midi_file.ticks_per_beat}")
        if midi_file.ticks_per_beat is None or midi_file.ticks_per_beat == 0:
             logging.warning("MIDI file has invalid or missing ticks_per_beat. Using default 480.")
        return self._parse_midi_data(midi_file)

    def _parse_midi_data(self, midi_file: mido.MidiFile) -> Dict[str, Any]:
        # sourcery skip: low-code-quality
        self.midi_analysis["ticks_per_beat"] = midi_file.ticks_per_beat or 480
        self.midi_analysis["filename"] = midi_file.filename
        absolute_tick_max = 0
        current_tempo = self.midi_analysis["default_tempo"]
        timed_notes = []

        # --- Tempo Handling: Find first tempo globally ---
        initial_tempo_found = False
        for track in midi_file.tracks:
             for msg in track:
                  if msg.is_meta and msg.type == 'set_tempo':
                       current_tempo = msg.tempo
                       logging.debug(f"Found initial tempo {current_tempo} ({mido.tempo2bpm(current_tempo):.2f} BPM)")
                       initial_tempo_found = True
                       break
             if initial_tempo_found: break
        if not initial_tempo_found:
             logging.debug(f"No initial tempo found, using default {current_tempo} ({mido.tempo2bpm(current_tempo):.2f} BPM)")

        # --- Process Tracks ---
        for track_num, track in enumerate(midi_file.tracks):
            track_name = track.name or f"Track {track_num}"
            self.midi_analysis["tracks"].append(track_name)
            logging.debug(f"Processing {track_name}...")
            absolute_tick_track = 0
            # active_notes_ticks: { (note, channel) : (start_tick, start_tempo, velocity) }
            active_notes_ticks = {}
            track_tempo = current_tempo # Each track starts with the initial global tempo

            for msg in track:
                # --- Time Calculation ---
                delta_ticks = msg.time
                absolute_tick_track += delta_ticks
                current_time_sec = mido.tick2second(absolute_tick_track, self.midi_analysis["ticks_per_beat"], track_tempo)

                # --- Meta Messages ---
                if msg.is_meta:
                    if msg.type == "key_signature":
                        self.midi_analysis["key_signature_changes"].append({
                            "time_seconds": current_time_sec, "tick": absolute_tick_track, "key": msg.key })
                    elif msg.type == "set_tempo":
                        old_tempo = track_tempo
                        track_tempo = msg.tempo
                        bpm = mido.tempo2bpm(track_tempo)
                        logging.debug(f"    T{track_num} Tempo Change at tick {absolute_tick_track}: {old_tempo} -> {track_tempo} ({bpm:.2f} BPM)")
                        self.midi_analysis["tempo_changes"].append({
                            "time_seconds": current_time_sec, "tick": absolute_tick_track,
                            "tempo": track_tempo, "bpm": bpm,
                        })
                    elif msg.type == "time_signature":
                        self.midi_analysis["time_signature_changes"].append({
                             "time_seconds": current_time_sec, "tick": absolute_tick_track,
                             "numerator": msg.numerator, "denominator": msg.denominator })
                                # Ignore other meta messages for timed notes

                elif msg.type == "program_change":
                    self.midi_analysis["program_changes"][track_num].append({
                           "time_seconds": current_time_sec, "tick": absolute_tick_track,
                           "program": msg.program, "channel": msg.channel, })

                elif msg.type == "note_on" and msg.velocity > 0:
                    note_key = (msg.note, msg.channel)
                    if note_key in active_notes_ticks:
                        # Overlap: Log warning but keep the original note active
                        logging.warning(f"    T{track_num} Note On received for already active key {note_key} at tick {absolute_tick_track}. Ignoring this Note On for timing, waiting for Note Off.")
                        # Optional: Could update velocity if needed: active_notes_ticks[note_key] = (*active_notes_ticks[note_key][:2], msg.velocity)
                    else:
                        # New Note On: Store start tick, tempo at start, velocity
                        active_notes_ticks[note_key] = (absolute_tick_track, track_tempo, msg.velocity)
                        logging.debug(f"    T{track_num} Note On: {note_key} Vel: {msg.velocity} at tick {absolute_tick_track}, tempo {track_tempo}")
                        # Update basic stats
                        self.midi_analysis["unique_notes"].add(msg.note)
                        self.midi_analysis["note_distribution"][msg.note] += 1
                        self.midi_analysis["total_notes"] += 1

                elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                    note_key = (msg.note, msg.channel)
                    if note_key in active_notes_ticks:
                        # Found matching Note On: Calculate duration and times
                        start_tick, start_tempo, start_velocity = active_notes_ticks.pop(note_key) # Remove from active
                        end_tick = absolute_tick_track
                        duration_ticks = end_tick - start_tick

                        # Use tempo at START for duration and start time calculation
                        duration_seconds = mido.tick2second(duration_ticks, self.midi_analysis["ticks_per_beat"], start_tempo)
                        note_start_time_sec = mido.tick2second(start_tick, self.midi_analysis["ticks_per_beat"], start_tempo)

                        logging.debug(f"    T{track_num} Note Off: {note_key} at tick {end_tick}. Start tick: {start_tick}, Dur Ticks: {duration_ticks}, Dur Sec: {duration_seconds:.3f}")

                        # Add to timed_notes if duration is positive
                        if duration_seconds > 0:
                             timed_notes.append({
                                 "note": msg.note, "start_sec": note_start_time_sec, "duration_sec": duration_seconds,
                                 "velocity": start_velocity, "track": track_num, "channel": msg.channel
                             })
                             # Update global duration stats
                             self.midi_analysis["note_duration_stats"]["min_duration"] = min(
                                 self.midi_analysis["note_duration_stats"]["min_duration"], duration_seconds)
                             self.midi_analysis["note_duration_stats"]["max_duration"] = max(
                                 self.midi_analysis["note_duration_stats"]["max_duration"], duration_seconds)
                        else:
                            logging.warning(f"    T{track_num} Zero or negative duration ({duration_seconds:.4f}) calculated for note {note_key} ending at tick {end_tick}. Start tick {start_tick}. Ignoring note.")
                    #else: # Note off without matching Note On
                    #    logging.debug(f"    T{track_num} Ignoring Note Off for {note_key} at tick {absolute_tick_track} - no matching active Note On found.")

            # --- End of Track: Handle Notes Still On ---
            end_time_sec = mido.tick2second(absolute_tick_track, self.midi_analysis["ticks_per_beat"], track_tempo)
            if active_notes_ticks:
                 logging.debug(f"  End of {track_name}: Handling {len(active_notes_ticks)} notes still active (no Note Off found).")
                 for note_key, (start_tick, start_tempo, start_velocity) in list(active_notes_ticks.items()):
                     duration_ticks = absolute_tick_track - start_tick
                     duration_seconds = mido.tick2second(duration_ticks, self.midi_analysis["ticks_per_beat"], start_tempo)
                     note_start_time_sec = mido.tick2second(start_tick, self.midi_analysis["ticks_per_beat"], start_tempo)
                     note, channel = note_key
                     logging.debug(f"    Ending active note {note_key} at track end. Start tick: {start_tick}, Dur Ticks: {duration_ticks}, Dur Sec: {duration_seconds:.3f}")
                     if duration_seconds > 0:
                          timed_notes.append({
                              "note": note, "start_sec": note_start_time_sec, "duration_sec": duration_seconds,
                              "velocity": start_velocity, "track": track_num, "channel": channel
                          })
                          self.midi_analysis["note_duration_stats"]["min_duration"] = min(
                              self.midi_analysis["note_duration_stats"]["min_duration"], duration_seconds)
                          self.midi_analysis["note_duration_stats"]["max_duration"] = max(
                              self.midi_analysis["note_duration_stats"]["max_duration"], duration_seconds)
                     else:
                           logging.warning(f"    Zero duration calculated for note {note_key} active at track end. Ignoring.")
                     active_notes_ticks.pop(note_key) # Remove handled note

            absolute_tick_max = max(absolute_tick_max, absolute_tick_track)

        # --- Final Calculations ---
        final_tempo = self.midi_analysis["tempo_changes"][-1]["tempo"] if self.midi_analysis["tempo_changes"] else current_tempo
        self.midi_analysis["total_duration"] = mido.tick2second(absolute_tick_max, self.midi_analysis["ticks_per_beat"], final_tempo)

        if timed_notes:
             total_duration_sum_sec = sum(n['duration_sec'] for n in timed_notes)
             self.midi_analysis["note_duration_stats"]["avg_duration"] = total_duration_sum_sec / len(timed_notes)
        else: self.midi_analysis["note_duration_stats"]["avg_duration"] = 0.0
        if self.midi_analysis["note_duration_stats"]["min_duration"] == float("inf"):
             self.midi_analysis["note_duration_stats"]["min_duration"] = 0.0

        # Sort timed notes and store
        self.midi_analysis["timed_notes"] = sorted(timed_notes, key=lambda x: x['start_sec'])
        logging.info(f"Finished parsing. Total timed notes extracted: {len(self.midi_analysis['timed_notes'])}")

        return self.midi_analysis

    # --- Report Generation Methods ---
    def generate_midi_analysis_report(self) -> str:
        analysis = self.midi_analysis
        if not analysis or analysis.get("filename") is None: return "No MIDI analysis data available."
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
        info = f"Approx. Total Duration: {analysis.get('total_duration', 'N/A'):.2f} seconds\n"
        info += f"Ticks Per Beat: {analysis.get('ticks_per_beat', 'N/A')}\n"
        info += f"Number of Tracks: {len(analysis.get('tracks', []))}\n"
        info += f"Tracks: {', '.join(analysis.get('tracks', []))}\n\n"
        return info
    def _generate_note_info(self, analysis):
        info = f"Total Notes Played (raw NoteOn): {analysis['total_notes']}\n"
        info += f"Notes in Sequence (Timed): {len(analysis.get('timed_notes',[]))}\n"
        info += f"Unique Notes Used: {len(analysis['unique_notes'])}\n"
        if analysis["unique_notes"]:
            min_note, max_note = min(analysis["unique_notes"]), max(analysis["unique_notes"])
            info += f"Note Range: {min_note} ({self._get_note_name_static(min_note)}) - {max_note} ({self._get_note_name_static(max_note)})\n\n"
            sorted_notes = sorted(analysis["note_distribution"].items(), key=lambda item: item[1], reverse=True)[:5]
            info += "Most Frequent Notes (Top 5):\n" + "\n".join([f"  Note {n} ({self._get_note_name_static(n)}): {c} times" for n, c in sorted_notes]) + "\n"
        else: info += "Note Range: N/A\nMost Frequent Notes: N/A\n"
        return info + "\n"
    def _generate_duration_stats(self, analysis):
        stats = analysis["note_duration_stats"]
        min_d = f"{stats['min_duration']:.4f}" if stats['min_duration'] is not None and stats['min_duration'] != float('inf') else "N/A"
        max_d = f"{stats['max_duration']:.4f}" if stats['max_duration'] is not None else "N/A"
        avg_d = f"{stats['avg_duration']:.4f}" if stats['avg_duration'] is not None else "N/A"
        s = "Note Duration Statistics (Timed Notes):\n"
        s += f"  Min Duration: {min_d}\n  Max Duration: {max_d}\n  Avg Duration: {avg_d}\n\n"
        return s
    def _generate_tempo_changes(self, analysis):
        changes = "Tempo Changes (BPM):\n"
        if not analysis["tempo_changes"]:
             default_bpm = mido.tempo2bpm(analysis.get("default_tempo", 500000))
             changes += f"  No tempo changes detected (Using default/initial: {default_bpm:.2f} BPM).\n"
        else:
            # Use the first tempo change found *anywhere* as the initial practical tempo
            initial_change = analysis["tempo_changes"][0]
            initial_bpm = mido.tempo2bpm(initial_change['tempo'])
            initial_tick = initial_change['tick']
            if initial_tick > 0: # If first change wasn't at tick 0, report the default used before it
                 default_bpm_val = mido.tempo2bpm(analysis.get("default_tempo", 500000))
                 changes += f"  Initial Tempo (Default): {default_bpm_val:.2f} BPM (until tick {initial_tick})\n"
            changes += f"  Tick {initial_tick} ({initial_change['time_seconds']:.2f}s): {initial_bpm:.2f} BPM\n"

            last_bpm = initial_bpm
            for change in analysis["tempo_changes"][1:]:
                bpm = change.get("bpm", mido.tempo2bpm(change["tempo"]))
                if bpm != last_bpm:
                    changes += f"  Tick {change['tick']} ({change['time_seconds']:.2f}s): {bpm:.2f} BPM\n"
                    last_bpm = bpm
        return changes + "\n"
    def _generate_time_signature_changes(self, analysis):
        changes = "Time Signature Changes:\n"
        if analysis["time_signature_changes"]:
            last_sig = None
            sorted_changes = sorted(analysis['time_signature_changes'], key=lambda x: x['tick'])
            for i, change in enumerate(sorted_changes):
                current_sig = f"{change['numerator']}/{change['denominator']}"
                prefix = "Initial:" if i == 0 else f"Tick {change['tick']} ({change['time_seconds']:.2f}s):"
                if current_sig != last_sig: changes += f"  {prefix} {current_sig}\n"; last_sig = current_sig
        else: changes += "  No time signature changes detected (Assumed 4/4).\n"
        return changes + "\n"
    def _generate_key_signature_changes(self, analysis):
        changes = "Key Signature Changes:\n"
        if analysis["key_signature_changes"]:
            last_key = None
            sorted_changes = sorted(analysis['key_signature_changes'], key=lambda x: x['tick'])
            for i, change in enumerate(sorted_changes):
                 prefix = "Initial:" if i == 0 else f"Tick {change['tick']} ({change['time_seconds']:.2f}s):"
                 if change["key"] != last_key: changes += f"  {prefix} {change['key']}\n"; last_key = change["key"]
        else: changes += "  No key signature changes detected.\n"
        return changes + "\n"
    def _generate_program_changes(self, analysis):
        changes = "Program (Instrument) Changes:\n"
        if analysis["program_changes"]:
            for track_num, changes_list in sorted(analysis["program_changes"].items()):
                track_name = analysis["tracks"][track_num] if track_num < len(analysis["tracks"]) else f"Track {track_num}"
                changes += f"  {track_name}:\n"
                last_prog = -1
                for change in sorted(changes_list, key=lambda x: x['tick']):
                    if change["program"] != last_prog:
                        changes += f"    Tick {change['tick']} ({change['time_seconds']:.2f}s), Ch {change['channel']}: Prog {change['program']}\n"
                        last_prog = change["program"]
        else: changes += "  No program changes detected.\n"
        return changes
    @staticmethod
    def _get_note_name_static(note: int) -> str:
        if not (0 <= note <= 127): return "??"
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (note // 12) - 1
        return f"{names[note % 12]}{octave}"

# =========================================================
# END OF MODIFIED AdvancedMIDIParser Class
# =========================================================


# --- Base Piano Trainer Class ---
class PianoTrainer:
    def __init__(self): logging.debug("Initializing Base PianoTrainer...")
    def _render_ui(self): pass
    def run(self, mode=None, midi_file=None): logging.debug("Running Base PianoTrainer...")


# --- Enhanced UI and Core Logic ---
def enhance_piano_trainer_ui(BasePianoTrainer):






    class EnhancedPianoTrainerUI(BasePianoTrainer):
        def __init__(self, *args, **kwargs):
            # --- Basic Init ---
            logging.info("Initializing EnhancedPianoTrainerUI...")
            try:
                pygame.init()
                pygame.midi.init()
                pygame.font.init()
            except Exception as e:
                logging.exception("Failed to initialize Pygame modules.")
                raise RuntimeError("Pygame initialization failed.") from e

            self.screen_width = 1450; self.screen_height = 700
            try:
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
                pygame.display.set_caption("Enhanced Piano Trainer")
            except pygame.error as e:
                 logging.exception("Failed to set up Pygame display.")
                 pygame.quit(); raise RuntimeError("Display setup failed.") from e

            super().__init__(*args, **kwargs)

            # --- Core Components ---
            self.midi_parser = AdvancedMIDIParser()
            self.midi_analysis = None
            self.performance_history = []
            self.music_theory = MusicTheory()

            # --- Piano Visualization ---
            self.white_key_width = 40; self.white_key_height = 200
            self.black_key_width = 26; self.black_key_height = 120
            self.first_note = 36; self.last_note = 96 # C2-C7
            num_white_keys = sum(not self.is_black_key(n) for n in range(self.first_note, self.last_note + 1))
            required_piano_width = num_white_keys * self.white_key_width
            self.piano_start_x = max(20, (self.screen_width - required_piano_width) // 2)
            self.piano_start_y = self.screen_height - self.white_key_height - 20

            # --- Note State ---
            self.active_notes = defaultdict(lambda: False)

            # --- Fonts ---
            try:
                self.font = pygame.font.SysFont("Arial", 24)
                self.small_font = pygame.font.SysFont("Arial", 18)
                self.key_font = pygame.font.SysFont("Arial", 12)
                self.title_font = pygame.font.SysFont("Arial", 36, bold=True)
                self.report_font = pygame.font.SysFont("Courier New", 14)
                self.next_note_font = pygame.font.SysFont("Arial", 28, bold=True)
                self.falling_note_font = pygame.font.SysFont("Arial", 10)
            except Exception:
                logging.warning("Fonts not found, using default.")
                self.font=pygame.font.Font(None,30); self.small_font=pygame.font.Font(None,24)
                self.key_font=pygame.font.Font(None,18); self.title_font=pygame.font.Font(None,40)
                self.report_font=pygame.font.Font(None,20); self.next_note_font=pygame.font.Font(None,34)
                self.falling_note_font=pygame.font.Font(None, 16)

            self.note_names_map = {i: AdvancedMIDIParser._get_note_name_static(i) for i in range(128)}

            # --- MIDI Input ---
            self.midi_input = None
            self.midi_device_name = None
            self.setup_midi_input()

            # --- Colors ---
            self.colors = {
                "background_gradient_start": (25, 25, 40), "background_gradient_end": (50, 50, 70),
                "text_primary": (230, 230, 255), "text_secondary": (180, 180, 210),
                "text_highlight": (120, 230, 120), "text_error": (255, 100, 100),
                "piano_white_key": (250, 250, 250), "piano_black_key": (30, 30, 30),
                "piano_white_key_pressed": (170, 170, 255), "piano_black_key_pressed": (100, 100, 200),
                "key_border": (60, 60, 60), "key_text": (0, 0, 0), "key_text_black": (200, 200, 200),
                "pressed_black_key_border": (255, 255, 255),
                "target_line": (255, 80, 80, 200),
                "falling_note_upcoming": (80, 80, 220), "falling_note_active": (240, 240, 100),
                "falling_note_hit": (80, 220, 80), "falling_note_missed": (180, 100, 100),
                "falling_note_border": (200, 200, 220), "next_note_text": (220, 220, 255),
            }

            # --- App State ---
            self.clock = pygame.time.Clock()
            self.midi_analysis_report_str = None
            self.challenge_difficulty = "intermediate"

            # --- Learning Mode State ---
            self.learning_mode_active = False
            self.learning_content = []
            self.learning_start_time_ms = 0
            self.current_sequence_time_sec = 0.0
            self.target_line_y = self.piano_start_y - TARGET_LINE_Y_OFFSET
            self.key_rect_map = {}
            self._update_key_rect_map()
            self.upcoming_notes_display = []
            self.score = 0
            self.last_played_event = None

            logging.info("EnhancedPianoTrainerUI Initialized successfully.")

        def _update_key_rect_map(self):
            """Calculate and cache screen rectangles for all displayed keys."""
            self.key_rect_map.clear()
            for note in range(self.first_note, self.last_note + 1):
                rect = self.get_key_rect(note)
                if rect: self.key_rect_map[note] = rect

        def run( self, mode: Optional[str] = "freestyle", midi_file: Optional[str] = None, difficulty: str = "intermediate", learning_type: str = "scale", root_note: int = 60, scale_chord_type: str = "major") -> None:
            """ Main application loop. """
            logging.info(f"Starting Piano Trainer. Mode: '{mode}', MIDI file: '{midi_file}', Difficulty: '{difficulty}', Learning: {learning_type} {scale_chord_type} from {root_note}")

            requested_mode = mode
            self.running = True
            self.challenge_difficulty = difficulty

            # --- Parse MIDI file FIRST if provided ---
            if midi_file:
                logging.info(f"MIDI file provided: {midi_file}. Parsing...")
                try:
                    self.midi_analysis = self.midi_parser.parse_midi_file(midi_file)
                    self.midi_analysis_report_str = self.midi_parser.generate_midi_analysis_report()
                    if requested_mode != "learning": print("\n--- MIDI Analysis Report ---\n" + self.midi_analysis_report_str + "\n--------------------------\n")

                    if self.midi_analysis:
                         timed_notes_list = self.midi_analysis.get('timed_notes', None)
                         if timed_notes_list is not None:
                              logging.info(f"Successfully parsed MIDI. Found {len(timed_notes_list)} timed notes.")
                              if not timed_notes_list: logging.warning("Parsing finished, but the 'timed_notes' list is EMPTY.")
                         else: logging.error("Parsing finished, but 'timed_notes' key is missing!")
                    else: logging.warning("Parsing resulted in self.midi_analysis being None.")

                except MIDIAnalysisError as e:
                    logging.error(f"Error during MIDI analysis: {e}")
                    self.midi_analysis_report_str = f"Error loading MIDI:\n{e}"
                    self.midi_analysis = None
                    if requested_mode == "learning" and learning_type == "midi":
                         logging.error("Cannot enter MIDI learning mode due to parsing error. Switching to freestyle.")
                         requested_mode = "freestyle"
                    elif requested_mode == "analysis_view":
                         logging.error("Cannot enter analysis view mode due to parsing error. Switching to freestyle.")
                         requested_mode = "freestyle"

            # --- Determine Final Mode ---
            if requested_mode == "freestyle" and midi_file and self.midi_analysis:
                self.mode = "analysis_view"
                logging.info("Switching mode to 'analysis_view' as MIDI file was provided.")
            else:
                self.mode = requested_mode

            # --- Main Loop ---
            try:
                self._setup_current_mode(learning_type, root_note, scale_chord_type, midi_file)
                game_start_time_ms = pygame.time.get_ticks()

                while self.running:
                    current_frame_time_ms = pygame.time.get_ticks()
                    self._handle_events()
                    if not self.running: break
                    self.process_midi_input(current_frame_time_ms)
                    if self.learning_mode_active: self.update_learning_mode(current_frame_time_ms)
                    self._render_ui()
                    pygame.display.flip()
                    self.clock.tick(60)

            except Exception as e: logging.exception("Unexpected error in main loop.")
            finally: self._cleanup()


        def _setup_current_mode( self, learning_type="scale", root_note=60, scale_chord_type="major", midi_file=None):
            """Initializes state based on the current self.mode."""
            logging.info(f"Setting up mode: {self.mode}")
            self.learning_mode_active = False
            self.learning_content.clear()
            self.score = 0

            if self.mode == "learning":
                self.learning_mode_active = True
                self._generate_learning_sequence(learning_type, root_note, scale_chord_type, midi_file)
                self.learning_start_time_ms = pygame.time.get_ticks() + (PREP_TIME_SEC * 1000)
                self.current_sequence_time_sec = -PREP_TIME_SEC

            elif self.mode == "analysis_view":
                if not self.midi_analysis and not self.midi_analysis_report_str: logging.warning("Analysis view mode but no analysis available.")
                elif not self.midi_analysis_report_str: logging.warning("Analysis view mode but report string is missing.")
            elif self.mode == "freestyle": logging.info("Freestyle mode active.")
            else:
                logging.warning(f"Unknown mode '{self.mode}'. Defaulting to freestyle.")
                self.mode = "freestyle"


        def _generate_learning_sequence( self, type="scale", root=60, scale_chord="major", midi_file=None):
            """Generates the sequence of FallingNote objects for learning mode."""
            self.learning_content.clear()
            current_time = 0.0
            notes_to_add = []

            # --- Decide Learning Source ---
            if type == "chord_progression":
                progression = [(60, "maj"), (67, "maj"), (69, "min"), (65, "maj")] # Example
                logging.info(f"Generating chord progression: C-G-Am-F")
                all_chord_notes = [
                    self.music_theory.generate_chord(r, c_type)
                    for r, c_type in progression
                ]
                chord_duration = 1.8
                time_between_chords = 2.0
                for chord in all_chord_notes:
                    for note in chord: self.learning_content.append(FallingNote(note, current_time, chord_duration, self.target_line_y, self.screen_height))
                    current_time += time_between_chords
                logging.info(f"Generated {len(self.learning_content)} falling notes from chord progression.")
                return

            elif type == "scale":
                scale_notes = self.music_theory.generate_scale(root, scale_chord, octaves=1)
                logging.info(f"Generating scale: {scale_chord} starting at {root}. Notes: {scale_notes}")
                notes_to_add = scale_notes

            # --- MIDI File Learning Check ---
            logging.debug(f"Checking conditions for MIDI learning: type='{type}', midi_file='{midi_file is not None}', self.midi_analysis='{self.midi_analysis is not None}'")
            timed_notes_list = None
            is_valid_timed_notes = False
            if self.midi_analysis:
                 timed_notes_list = self.midi_analysis.get("timed_notes")
                 is_valid_timed_notes = isinstance(timed_notes_list, list) and len(timed_notes_list) > 0
                 logging.debug(f"self.midi_analysis['timed_notes'] check: Found key = '{timed_notes_list is not None}', Is list = '{isinstance(timed_notes_list, list)}', Length > 0 = '{len(timed_notes_list) > 0 if isinstance(timed_notes_list, list) else 'N/A'}'. Overall condition: {is_valid_timed_notes}")

            # Use MIDI data if conditions met
            if type == "midi" and midi_file and is_valid_timed_notes:
                logging.info(f"Generating learning sequence from MIDI file: {midi_file}")
                notes_added_count = 0; notes_skipped_count = 0
                for note_data in timed_notes_list:
                    fn = FallingNote( note_data["note"], note_data["start_sec"], note_data["duration_sec"], self.target_line_y, self.screen_height)
                    if self.first_note <= fn.note <= self.last_note:
                        self.learning_content.append(fn); notes_added_count += 1
                    else: notes_skipped_count += 1
                logging.info(f"Generated {notes_added_count} falling notes from MIDI. Skipped {notes_skipped_count} notes outside display range.")
                if notes_added_count == 0 and notes_skipped_count > 0: logging.warning("All notes from MIDI were outside display range.")
                elif notes_added_count == 0 and notes_skipped_count == 0: logging.warning("MIDI processing yielded no valid timed notes.")
                return

            # --- Fallback or Scale/Chord Default Timing ---
            else:
                if type == "midi": # Log specific reason for MIDI failure
                     reason = "Unknown reason"
                     if not midi_file: reason = "midi_file path not provided"
                     elif not self.midi_analysis: reason = "self.midi_analysis object is None"
                     elif not isinstance(timed_notes_list, list): reason = "'timed_notes' key missing or not list"
                     elif not timed_notes_list: reason = "'timed_notes' list was empty"
                     logging.warning(f"Could not generate from MIDI ({reason}). Generating fallback C Major scale.")
                     notes_to_add = self.music_theory.generate_scale(60, "major", octaves=1)
                elif not notes_to_add: # Handle invalid scale type etc.
                     logging.warning(f"Unsupported learning type '{type}' or no notes generated. Generating fallback C Major scale.")
                     notes_to_add = self.music_theory.generate_scale(60, "major", octaves=1)

            # Apply Default Timing
            if notes_to_add:
                note_duration = 0.4
                time_between_notes = 0.5
                for note in notes_to_add:
                    fn = FallingNote(note, current_time, note_duration, self.target_line_y, self.screen_height)
                    self.learning_content.append(fn)
                    current_time += time_between_notes
                logging.info(f"Generated {len(self.learning_content)} falling notes using default timing.")

            if not self.learning_content: logging.error("Failed to generate any learning content!")




        def update_learning_mode(self, current_frame_time_ms: int):
            """Update falling notes, check for hits, manage sequence time."""
            if not self.learning_mode_active: return
            time_since_sequence_start_ms = current_frame_time_ms - self.learning_start_time_ms
            self.current_sequence_time_sec = time_since_sequence_start_ms / 1000.0
            if not self.learning_content: return # No notes to update

            notes_to_remove_indices = []
            active_or_upcoming_notes = []

            for i, fn in enumerate(self.learning_content):
                fn.update(self.current_sequence_time_sec, self.key_rect_map)
                is_done = (fn.state == "hit" and current_frame_time_ms > fn.hit_time_ms + 500) or \
                          (fn.state == "missed" and fn.current_y > self.target_line_y + 100) or \
                          (fn.current_y > self.screen_height + 100)
                if is_done:
                    notes_to_remove_indices.append(i)
                    if fn.state not in ["hit", "missed", "invalid_key"]:
                        fn.state = "missed"; logging.debug(f"Missed note {fn.note} ({fn._note_name_cache}) (scrolled off/removed)")
                if fn.state in ["upcoming", "active"] and fn.current_y > -100:
                    active_or_upcoming_notes.append(fn)

            if notes_to_remove_indices:
                 for i in sorted(notes_to_remove_indices, reverse=True):
                     if 0 <= i < len(self.learning_content): del self.learning_content[i]
                     else: logging.warning(f"Attempted to remove note at invalid index {i}")

            active_or_upcoming_notes.sort(key=lambda x: x.start_time_sec)
            self.upcoming_notes_display = []
            if active_or_upcoming_notes:
                 earliest_start_time = min(fn.start_time_sec for fn in active_or_upcoming_notes)
                 current_group = [self.get_note_name(fn.note) for fn in active_or_upcoming_notes if abs(fn.start_time_sec - earliest_start_time) < 0.05]
                 self.upcoming_notes_display = sorted(list(set(current_group)))

            if not self.learning_content and notes_to_remove_indices:
                 logging.info("Learning sequence finished (no more notes).")


        def _handle_events(self):
            """Processes Pygame events."""
            for event in pygame.event.get():
                if event.type == pygame.QUIT: self.running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]: self.running = False


        def _render_ui(self) -> None:
            """Draws all UI elements."""
            self._draw_background()
            self._draw_title()
            if self.mode == "learning": self._draw_learning_ui()
            elif self.mode == "analysis_view": self._draw_analysis_report()
            elif self.mode == "freestyle": self._draw_freestyle_ui()
            else: self._draw_fallback_ui()
            self.draw_piano()
            self._draw_status_bar()

        def _draw_learning_ui(self):
            """Draws learning mode specific elements."""
            white_key_indices = [self.count_white_keys_before(n) for n in range(self.first_note, self.last_note + 1) if not self.is_black_key(n)]
            if white_key_indices:
                 piano_visual_width = (max(white_key_indices) + 1) * self.white_key_width -1
                 target_line_end_x = self.piano_start_x + piano_visual_width
                 pygame.draw.line(self.screen, self.colors["target_line"], (self.piano_start_x, self.target_line_y), (target_line_end_x, self.target_line_y), 3)

            for fn in self.learning_content: fn.draw(self.screen, self.colors, self.falling_note_font)

            if self.upcoming_notes_display:
                next_notes_str = " / ".join(self.upcoming_notes_display)
                text_surf = self.next_note_font.render(f"Next: {next_notes_str}", True, self.colors["next_note_text"])
                text_rect = text_surf.get_rect(centerx=self.screen_width // 2, y=90)
                self.screen.blit(text_surf, text_rect)
            elif self.learning_mode_active and not self.learning_content and self.current_sequence_time_sec > 0: # Show complete msg only after prep time
                 end_surf = self.next_note_font.render("Sequence Complete!", True, self.colors["text_highlight"])
                 end_rect = end_surf.get_rect(centerx=self.screen_width // 2, y=90)
                 self.screen.blit(end_surf, end_rect)

            score_surf = self.font.render(f"Score: {self.score}", True, self.colors["text_highlight"])
            score_rect = score_surf.get_rect(topright=(self.screen_width - 20, 90))
            self.screen.blit(score_surf, score_rect)

            if self.current_sequence_time_sec < 0:
                 countdown = max(1, int(abs(self.current_sequence_time_sec)))
                 cd_surf = self.title_font.render(f"{countdown}", True, self.colors["text_primary"])
                 cd_rect = cd_surf.get_rect(center=(self.screen_width // 2, self.screen_height // 3))
                 self.screen.blit(cd_surf, cd_rect)

        def _draw_analysis_report(self):
            content_y_start = 90
            if self.midi_analysis_report_str: self._extracted_from__draw_main_content_436(content_y_start)
            else:
                  message = "No MIDI Analysis data loaded."
                   if self.midi_analysis is None and hasattr(self, 'midi_analysis_report_str') and self.midi_analysis_report_str and 'Error' in self.midi_analysis_report_str:
                      message = "MIDI Analysis Failed. See console."
                  info_surf = self.font.render(message, True, self.colors["text_secondary"])
                  self.screen.blit(info_surf, (20, content_y_start))

        def _draw_freestyle_ui(self):
            content_y_start = 90
            info_surf = self.font.render("Freestyle Mode - Play your MIDI keyboard!", True, self.colors["text_highlight"])
            self.screen.blit(info_surf, (20, content_y_start))
            hint_surf = self.small_font.render("Q/Esc: Quit", True, self.colors["text_secondary"])
            self.screen.blit(hint_surf, (20, content_y_start + 40))

        def _draw_fallback_ui(self):
            content_y_start = 90
            error_surf = self.font.render(f"Current Mode: {self.mode}", True, self.colors["text_error"])
            self.screen.blit(error_surf, (20, content_y_start))

        def _draw_background(self):
            screen_rect = self.screen.get_rect(); start_color=self.colors["background_gradient_start"]; end_color=self.colors["background_gradient_end"]
            for y in range(screen_rect.height):
                interp = y / screen_rect.height
                r = int(start_color[0] * (1 - interp) + end_color[0] * interp); g = int(start_color[1] * (1 - interp) + end_color[1] * interp); b = int(start_color[2] * (1 - interp) + end_color[2] * interp)
                pygame.draw.line(self.screen, (r, g, b), (0, y), (screen_rect.width, y))

        def _draw_title(self):
            mode_str = self.mode.replace("_", " ").title()
            title_text_str = f"Piano Trainer - {mode_str}"
            if self.mode == "learning": title_text_str += f" ({self.challenge_difficulty})"
            title_surf = self.title_font.render(title_text_str, True, self.colors["text_primary"])
            title_rect = title_surf.get_rect(center=(self.screen_width // 2, 40))
            self.screen.blit(title_surf, title_rect)

        def _draw_status_bar(self):
            status_text = f"MIDI: {self.midi_device_name if self.midi_input else 'Not Connected'}"
            status_color = self.colors["text_highlight"] if self.midi_input else self.colors["text_error"]
            status_surf = self.small_font.render(status_text, True, status_color)
            status_rect = status_surf.get_rect(bottomright=(self.screen_width - 15, self.screen_height - 15))
            self.screen.blit(status_surf, status_rect)

        def _extracted_from__draw_main_content_436(self, content_y_start):
             y_offset = content_y_start; max_report_width = self.screen_width * 0.85; report_x = (self.screen_width - max_report_width) / 2
             line_height = self.report_font.get_linesize(); max_lines_display = 22
             if not self.midi_analysis_report_str: report_lines = ["Analysis report not available."]
             else: report_lines = self.midi_analysis_report_str.split("\n")
             lines_drawn = 0
             for line in report_lines:
                 if lines_drawn >= max_lines_display: break
                 display_line = line; line_surf = self.report_font.render(display_line, True, self.colors["text_secondary"])
                 if line_surf.get_width() > max_report_width:
                    avg_char_width = self.report_font.size("A")[0]; max_chars = int(max_report_width / avg_char_width) if avg_char_width > 0 else 50
                    display_line = line[:max_chars - 3] + "..."; line_surf = self.report_font.render(display_line, True, self.colors["text_secondary"])
                 self.screen.blit(line_surf, (report_x, y_offset + lines_drawn * line_height)); lines_drawn += 1
             if len(report_lines) > max_lines_display:
                more_text = self.small_font.render("... (Full report printed to console)", True, self.colors["text_highlight"])
                self.screen.blit(more_text, (report_x, y_offset + max_lines_display * line_height + 5))

        def setup_midi_input(self):
            self.midi_input = None; self.midi_device_name = None; input_id = -1; found_device_name = "Unknown Device"
            logging.info("Searching for MIDI input devices..."); print("\n--- Searching for MIDI Inputs ---")
            try:
                pygame.midi.quit(); pygame.midi.init() # Reset MIDI subsystem
                device_count = pygame.midi.get_count()
                if device_count == 0: logging.warning("No MIDI devices found."); print("No MIDI devices found."); return
                for i in range(device_count):
                    info = pygame.midi.get_device_info(i);
                    if info is None: continue
                    name_bytes = info[1]; is_input = info[2]
                    try: decoded_name = name_bytes.decode(errors='replace')
                    except Exception: decoded_name = f"Device {i} (Decode Error)"
                    print(f"Device #{i}: '{decoded_name}' (Input: {is_input}, Output: {info[3]}, Opened: {info[4]})")
                    if is_input and input_id == -1: input_id = i; found_device_name = decoded_name
            except pygame.midi.MidiException as e: logging.error(f"Pygame MIDI Error: {e}", exc_info=True); print(f"\n--- Error Accessing MIDI System: {e} ---"); return
            except Exception as e: logging.exception("Unexpected error during MIDI enum."); print("\n--- Unexpected Error During MIDI Scan ---"); return
            print("----------------------------------")
            if input_id != -1:
                try:
                    print(f"Attempting connection to: '{found_device_name}' (ID: {input_id})")
                    self.midi_input = pygame.midi.Input(input_id); self.midi_device_name = found_device_name
                    logging.info(f"Successfully connected to MIDI: {self.midi_device_name}"); print(f"Successfully connected: {self.midi_device_name}")
                except Exception as e:
                    logging.exception(f"Failed to open MIDI device {input_id} ('{found_device_name}')"); print(f"\n--- Error opening MIDI '{found_device_name}': {e} ---")
                    self.midi_input = None; self.midi_device_name = None
            if self.midi_input is None:
                logging.warning("No suitable MIDI input connected or opened.")
                if input_id != -1: print(f"\n--- Failed to connect to '{found_device_name}'. MIDI inactive. ---")
                elif device_count > 0: print("\n--- No MIDI *Input* devices were found. MIDI inactive. ---")
                else: print("\n--- No MIDI Keyboard Detected. Connect and restart. ---")

        def is_black_key(self, note: int) -> bool: return (note % 12) in [1, 3, 6, 8, 10]

        def count_white_keys_before(self, note: int) -> int:
            count = 0;
            if note < self.first_note: return 0
            for n in range(self.first_note, note):
                 if not self.is_black_key(n): count += 1
            return count

        def get_note_name(self, note: int) -> str: return self.note_names_map.get(note, "??")

        def get_key_rect(self, note: int) -> Optional[pygame.Rect]:
             if note in self.key_rect_map: return self.key_rect_map[note]
             if not (self.first_note <= note <= self.last_note): return None
             if self.is_black_key(note):
                 idx = self.count_white_keys_before(note); x = self.piano_start_x + (idx * self.white_key_width) - (self.black_key_width // 2)
                 rect = pygame.Rect(x, self.piano_start_y, self.black_key_width, self.black_key_height)
             else:
                 idx = self.count_white_keys_before(note); x = self.piano_start_x + idx * self.white_key_width
                 rect = pygame.Rect(x, self.piano_start_y, self.white_key_width - 1, self.white_key_height)
             if rect: self.key_rect_map[note] = rect
             return rect

        def process_midi_input(self, current_frame_time_ms: int):
            """ Reads MIDI input, updates active_notes, checks hits. """
            if not self.midi_input: return
            try:
                if self.midi_input.poll():
                    midi_events = self.midi_input.read(128)
                    for event in midi_events:
                        data, timestamp_in = event
                        if not isinstance(data, (list, tuple)) or len(data) < 3: continue # Skip malformed
                        status = data[0]; note = data[1]; velocity = data[2]; channel = status & 0x0F

                        if 144 <= status <= 159 and velocity > 0: # Note On
                            logging.debug(f"MIDI IN: Note On - Ch={channel} Note={note}, Vel={velocity}")
                            self.active_notes[note] = True
                            play_time_ms = current_frame_time_ms
                            self.last_played_event = {"note": note, "time_ms": play_time_ms}
                            if self.learning_mode_active:
                                hit_registered = False
                                for fn in self.learning_content:
                                     if fn.check_hit(note, play_time_ms): hit_registered = True; self.score += 10
                                if not hit_registered: logging.debug(f"Played {note} ({self.get_note_name(note)}) but no target found.")

                        elif (128 <= status <= 143) or (144 <= status <= 159 and velocity == 0): # Note Off
                            logging.debug(f"MIDI IN: Note Off - Ch={channel} Note={note}")
                            if note in self.active_notes: self.active_notes[note] = False

                        elif 176 <= status <= 191: # Control Change
                             controller = data[1]; value = data[2]
                             if controller == 64: # Sustain
                                 sustain_on = value >= 64; logging.debug(f"MIDI IN: Sustain Pedal {'On' if sustain_on else 'Off'} (Ch={channel}, Val={value})")
                                 # Add sustain logic if needed

            except pygame.midi.MidiException as e:
                logging.error(f"MIDI Read Error: {e}. Disconnecting.", exc_info=True)
                print(f"\n--- MIDI Device Error: {e}. Disconnecting. ---")
                if self.midi_input:
                    try:
                        self.midi_input.close()
                    except Exception:
                        pass
                self.midi_input = None
                self.midi_device_name = None
            except Exception as e:
                logging.exception("Unexpected error processing MIDI input.")

        def draw_piano(self):
            """Draws the piano keys, highlighting active ones."""
            # Draw White Keys
            for note in range(self.first_note, self.last_note + 1):
                if not self.is_black_key(note):
                    rect = self.get_key_rect(note)
                    if rect:
                        is_pressed = self.active_notes.get(note, False)
                        color = self.colors["piano_white_key_pressed"] if is_pressed else self.colors["piano_white_key"]
                        border_color = self.colors["key_border"]; text_color = self.colors["key_text"]
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, border_color, rect, 1)
                        note_name = self.get_note_name(note)
                        if note_name:
                            name_surf = self.key_font.render(note_name, True, text_color)
                            name_rect = name_surf.get_rect(centerx=rect.centerx, bottom=rect.bottom - 5)
                            self.screen.blit(name_surf, name_rect)
            # Draw Black Keys
            for note in range(self.first_note, self.last_note + 1):
                if self.is_black_key(note):
                    rect = self.get_key_rect(note)
                    if rect:
                        is_pressed = self.active_notes.get(note, False)
                        color = self.colors["piano_black_key_pressed"] if is_pressed else self.colors["piano_black_key"]
                        border_thickness = 2 if is_pressed else 0; border_color = self.colors["pressed_black_key_border"]
                        pygame.draw.rect(self.screen, color, rect)
                        if border_thickness > 0: pygame.draw.rect(self.screen, border_color, rect, border_thickness)

        def _cleanup(self):
            """Properly shuts down Pygame and MIDI resources."""
            logging.info("Cleaning up resources...")
            if self.midi_input:
                try: self.midi_input.close(); logging.info("MIDI input closed.")
                except Exception as e: logging.exception("Error closing MIDI input.")
            pygame.midi.quit(); pygame.font.quit(); pygame.quit()
            logging.info("Pygame quit."); print("\n--- Piano Trainer Exited ---")




    return EnhancedPianoTrainerUI




    return EnhancedPianoTrainerUI


# Argument Parsing and Main Execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Piano Trainer with Pygame")
    parser.add_argument( "-m", "--mode", type=str, default="freestyle", choices=["freestyle", "learning", "analysis_view"], help="Run mode")
    parser.add_argument( "-f", "--midi", type=str, default=None, help="Path to MIDI file for analysis or learning mode")
    parser.add_argument( "--learn", type=str, default="scale", choices=["scale", "chord_progression", "midi"], help="Type of content for learning mode")
    parser.add_argument( "--root", type=int, default=60, help="Root note (MIDI number) for scales/chords")
    parser.add_argument( "--type", type=str, default="major", help="Scale type (e.g., major) or 'progression'")
    parser.add_argument( "-d", "--difficulty", type=str, default="intermediate", choices=["beginner", "intermediate", "advanced"], help="Difficulty setting")
    parser.add_argument( "-v", "--verbose", action="store_true", help="Enable DEBUG logging")

    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    # Use filemode='a' to append if the file exists, useful for multiple runs during dev
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s", filename='piano_trainer.log', filemode='a')
    # Also add a handler to print INFO level messages to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO) # Console shows INFO and above
    formatter = logging.Formatter('%(levelname)s - %(module)s - %(message)s') # Simpler format for console
    console_handler.setFormatter(formatter)
    logging.getLogger().addHandler(console_handler)


    if args.mode == "learning" and args.learn == "midi" and not args.midi:
        parser.error("--learn midi requires -f/--midi argument.")
        logging.critical("--learn midi requires -f/--midi argument.") # Log critical error
        sys.exit(1) # Exit if invalid args

    try:
        logging.info(f"Application starting with args: {args}")
        EnhancedTrainer = enhance_piano_trainer_ui(PianoTrainer)
        trainer_app = EnhancedTrainer()
        trainer_app.run( mode=args.mode, midi_file=args.midi, difficulty=args.difficulty, learning_type=args.learn, root_note=args.root, scale_chord_type=args.type )

    except RuntimeError as e:
        print(f"\nApplication failed to start: {e}", file=sys.stderr)
        logging.critical(f"Application runtime error: {e}", exc_info=True)
        if 'trainer_app' in locals() and trainer_app: trainer_app._cleanup()
        else: pygame.quit()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupt received, exiting..."); logging.info("Keyboard interrupt.")
        if 'trainer_app' in locals() and trainer_app and getattr(trainer_app, 'running', False):
             trainer_app.running = False; trainer_app._cleanup()
        else: pygame.quit()
        sys.exit(0)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        logging.critical("Unhandled exception in main.", exc_info=True)
        if 'trainer_app' in locals() and trainer_app: trainer_app._cleanup()
        else: pygame.quit()
        sys.exit(1)
