#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pygame
import pygame.midi
import pygame.mixer  # Add this import for sound playback
import mido
import sys
import time
import numpy as np
import argparse
from collections import defaultdict, Counter, deque
import logging
import io
import random
from typing import List, Dict, Optional, Any, Set

# --- Constants ---
# Learning Mode Settings
FALLING_NOTE_SPEED = 150  # Pixels per second
TARGET_LINE_Y_OFFSET = 50  # How far above the piano keys the target line is
HIT_WINDOW_MS = 300  # Time window (milliseconds) around target time to count as a hit
PREP_TIME_SEC = 3  # Seconds before the first note starts falling


# --- Custom Exception ---
class MIDIAnalysisError(Exception):
    """Custom exception for MIDI analysis errors."""

    pass


# --- Note/Chord Generation ---
class MusicTheory:
    """Helper class for basic music theory elements."""

    SCALE_INTERVALS = {
        "major": [0, 2, 4, 5, 7, 9, 11, 12],  # W-W-H-W-W-W-H
        "natural_minor": [0, 2, 3, 5, 7, 8, 10, 12],  # W-H-W-W-H-W-W
        "harmonic_minor": [0, 2, 3, 5, 7, 8, 11, 12],  # W-H-W-W-H-WH-H
        "chromatic": list(range(13)),
    }
    CHORD_INTERVALS = {
        "maj": [0, 4, 7],  # Major Triad
        "min": [0, 3, 7],  # Minor Triad
        "dim": [0, 3, 6],  # Diminished Triad
        "aug": [0, 4, 8],  # Augmented Triad
        "maj7": [0, 4, 7, 11],  # Major 7th
        "min7": [0, 3, 7, 10],  # Minor 7th
        "dom7": [0, 4, 7, 10],  # Dominant 7th
        "dim7": [0, 3, 6, 9],  # Diminished 7th
        "sus4": [0, 5, 7],  # Suspended 4th
    }

    @staticmethod
    def generate_scale(
        root_note: int, scale_type: str = "major", octaves: int = 1
    ) -> List[int]:
        """Generates MIDI notes for a scale."""
        intervals = MusicTheory.SCALE_INTERVALS.get(scale_type.lower())
        if not intervals:
            logging.warning(f"Unknown scale type: {scale_type}. Defaulting to major.")
            intervals = MusicTheory.SCALE_INTERVALS["major"]

        scale_notes = []
        for o in range(octaves):
            for interval in intervals[:-1]:  # Exclude octave repetition within loop
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
            logging.warning(
                f"Unknown chord type: {chord_type}. Defaulting to major triad."
            )
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

    def __init__(
        self,
        note: int,
        start_time_sec: float,
        duration_sec: float,
        target_y: int,
        screen_height: int,
        velocity: int = 127  # new: capture velocity
    ):
        self.note = note
        self.start_time_sec = start_time_sec
        self.end_time_sec = start_time_sec + duration_sec
        self.duration_sec = duration_sec
        self.target_y = target_y
        self.velocity = velocity  # new: store velocity
        # Height remains proportional to duration.
        self.note_height = max(20, self.duration_sec * FALLING_NOTE_SPEED)
        self.start_y = target_y - (FALLING_NOTE_SPEED * start_time_sec)
        self.current_y = self.start_y - (screen_height + self.note_height + 100)
        self.rect = None
        self.state = "upcoming"  # upcoming, active, hit, missed, invalid_key
        self.hit_time_ms: Optional[int] = None
        self._note_name_cache = AdvancedMIDIParser._get_note_name_static(
            note
        )  # Cache note name

    # Static method added to FallingNote for easier access in check_hit logging
    @staticmethod
    def get_note_name(note: int) -> str:
        """Gets the standard note name (e.g., C#4) using the static method."""
        return AdvancedMIDIParser._get_note_name_static(note)

    # *** Method Updated with Enhanced Debug Logging ***
    def update(self, current_time_sec: float, key_rect_map: Dict[int, pygame.Rect]):
        """Update the note's vertical position and state."""
        key_rect = key_rect_map.get(self.note)
        if not key_rect:
            if self.state != "invalid_key":
                # Log only once when it becomes invalid
                logging.debug(f"Note {self.note} ({self._note_name_cache}) is outside the displayed piano range. Marking as invalid.")
                self.state = "invalid_key"
            return  # Cannot update position if key doesn't exist on piano

        # Calculate current Y position based on time (bottom of the note)
        time_to_hit_sec = self.start_time_sec - current_time_sec
        self.current_y = self.target_y - (time_to_hit_sec * FALLING_NOTE_SPEED)

        # Recalculate height (could be constant if duration doesn't change, but recalculating is safer)
        height = max(10, self.duration_sec * FALLING_NOTE_SPEED)
        width = key_rect.width  # Use the actual key width

        center_x = key_rect.centerx
        # Rect represents (left, top, width, height)
        # Note 'top' is current_y - height
        self.rect = pygame.Rect(
            center_x - width // 2, self.current_y - height, width, height
        )

        # Update state based on time, only if not already hit/missed/invalid
        if self.state not in ["hit", "missed", "invalid_key"]:
            current_time_ms = int(current_time_sec * 1000)
            start_time_ms = int(self.start_time_sec * 1000)
            previous_state = self.state  # Save for logging state transitions

            # Check for missed state (if current time is significantly past the hit window end)
            if current_time_ms > (start_time_ms + HIT_WINDOW_MS):
                if self.state != "missed":  # Log only on state change
                    logging.debug(
                        f"Note {self.note} ({self._note_name_cache}) changed state: {previous_state} -> MISSED. "
                        f"Current time {current_time_ms}ms > Target time {start_time_ms}ms + Window {HIT_WINDOW_MS}ms"
                    )
                self.state = "missed"
            # Check for active state (within the hit window)
            elif abs(current_time_ms - start_time_ms) <= HIT_WINDOW_MS:
                if self.state != "active":  # Log only on state change
                    logging.debug(
                        f"Note {self.note} ({self._note_name_cache}) changed state: {previous_state} -> ACTIVE. "
                        f"Current time {current_time_ms}ms, Target time {start_time_ms}ms, Diff: {current_time_ms - start_time_ms}ms"
                    )
                self.state = "active"
            # Otherwise, it's upcoming
            else:
                # Only log if state actually changed
                if self.state != "upcoming" and previous_state != "upcoming":
                    logging.debug(
                        f"Note {self.note} ({self._note_name_cache}) changed state: {previous_state} -> UPCOMING. "
                        f"Current time {current_time_ms}ms, Target time {start_time_ms}ms, Time to hit: {time_to_hit_sec:.3f}s"
                    )
                self.state = "upcoming"

    def draw(self, screen, colors, font):
        """Draw the falling note with the label inside from the start."""
        # Don't draw if invalid, or completely off-screen
        if self.state == "invalid_key" or not self.rect:
            return
        # Basic visibility check (generous bounds)
        if self.rect.bottom < -50 or self.rect.top > screen.get_height() + 50:
            return

        # --- Determine Color and Border ---
        color = colors["falling_note_upcoming"]
        border_color = colors["falling_note_border"]
        text_color = colors["key_text"]  # Usually black for labels inside
        border = 1  # Default border

        if self.state == "active":
            color = colors["falling_note_active"]
        elif self.state == "hit":
            color = colors["falling_note_hit"]
            border = 0  # No border when hit
        elif self.state == "missed":
            color = colors["falling_note_missed"]
            border = 0  # No border when missed

        # --- Draw the Bar ---
        pygame.draw.rect(screen, color, self.rect)
        if border > 0:
            pygame.draw.rect(screen, border_color, self.rect, 1)

        # --- Draw the Note Name Text ---
        # Render the note name using the provided font
        text_surf = font.render(self._note_name_cache, True, text_color)
        text_rect = text_surf.get_rect()

        # Center the text horizontally within the bar
        text_rect.centerx = self.rect.centerx

        # Position the text vertically: Place it near the *bottom* of the bar,
        text_rect.bottom = self.rect.bottom - 5  # Adjust '5' for padding

        # Optional: Ensure text doesn't go above the top of the bar if the bar is very short
        if text_rect.top < self.rect.top:
            text_rect.top = self.rect.top + 2  # Add a small top padding if clipped

        # Only draw the text if it fits horizontally and the bar has *some* height
        if (
                    text_rect.width < self.rect.width
                    and self.rect.height > font.get_height() * 0.5
                ) and text_rect.colliderect(screen.get_rect()):
            screen.blit(text_surf, text_rect)
        # --- End Text Drawing ---

    # *** Method Updated with Enhanced Debug Logging and Fixed Timing ***
    def check_hit(self, played_note: int, play_time_ms: int) -> bool:
        """Check if this note was hit correctly."""
        # Can only hit upcoming or active notes
        if self.state in ["upcoming", "active"]:
            start_time_ms = int(self.start_time_sec * 1000)
            time_diff = play_time_ms - start_time_ms  # Signed difference can be useful
            is_match = played_note == self.note

            # FIXED: Make the hit window much more lenient
            # Allow notes to be hit as long as they're visible and the correct key is pressed
            is_in_window = True  # Accept any timing for now to debug the issue

            # Also log the absolute time difference for debugging
            abs_time_diff = abs(time_diff)

            # --- Enhanced Debug Logging ---
            logging.debug(
                f"check_hit({played_note}/{self.get_note_name(played_note)}): Target={self.note}/{self._note_name_cache}, "
                f"TargetStart={start_time_ms}ms, PlayTime={play_time_ms}ms, Diff={time_diff}ms, "
                f"NoteMatch={is_match}, TimeWindow=UNLIMITED, CurrentState={self.state}"
            )
            # --- End Enhanced Logging ---

            # Check note match - temporarily ignore timing window
            if is_match:
                return self._extracted_from_check_hit_27(play_time_ms)
            else:
                # --- Enhanced Debug Logging ---
                logging.debug(
                    f"  Note mismatch: Played {self.get_note_name(played_note)} but expected {self._note_name_cache}"
                )
                # --- End Enhanced Debug Logging ---
        elif self.state == "hit":
            # --- Enhanced Debug Logging ---
            logging.debug(
                f"check_hit called for {self._note_name_cache} which is already 'hit'. Current state={self.state}, Hit time={self.hit_time_ms}ms"
            )
        elif self.state == "missed":
            # Try to hit even missed notes for debugging purposes
            if played_note == self.note:
                logging.debug(f"RECOVERY: Converting missed note {self._note_name_cache} to hit state")
                self.state = "hit"
                self.hit_time_ms = play_time_ms
                return True
            else:
                # --- Enhanced Debug Logging ---
                logging.debug(
                    f"check_hit called for {self._note_name_cache} which is already 'missed'. Ignoring."
                )
                # --- End Enhanced Debug Logging ---
        elif self.state == "invalid_key":
            # --- Enhanced Debug Logging ---
            logging.debug(
                f"check_hit called for {self._note_name_cache} which is 'invalid_key'. Ignoring."
            )
            # --- End Enhanced Debug Logging ---
        return False

    # TODO Rename this here and in `check_hit`
    def _extracted_from_check_hit_27(self, play_time_ms):
        # --- Enhanced Debug Logging ---
        old_state = self.state
        logging.debug(
            f"  >>> HIT REGISTERED for {self._note_name_cache}! Changing state from {old_state} to hit."
        )
        # --- End Enhanced Debug Logging ---
        self.state = "hit"
        self.hit_time_ms = play_time_ms  # Record hit time

        # --- Enhanced Debug Logging ---
        logging.debug(
            f"  >>> VERIFICATION: Note {self._note_name_cache} state is now '{self.state}' (was '{old_state}')"
        )
        # --- End Enhanced Debug Logging ---
        return True


# =========================================================
# AdvancedMIDIParser Class (Unchanged from previous version)
# =========================================================
class AdvancedMIDIParser:
    """Enhanced MIDI file parsing with overlap handling."""

    def __init__(self):
        self.midi_analysis = self._get_default_analysis()
        # Logging configured by main app

    def _get_default_analysis(self) -> Dict[str, Any]:
        return {
            "total_notes": 0,
            "unique_notes": set(),
            "note_distribution": defaultdict(int),
            "note_duration_stats": {
                "min_duration": float("inf"),
                "max_duration": 0.0,
                "avg_duration": 0.0,
            },
            "tempo_changes": [],
            "key_signature_changes": [],
            "time_signature_changes": [],
            "program_changes": defaultdict(list),
            "total_duration": 0.0,
            "ticks_per_beat": None,
            "filename": None,
            "tracks": [],
            "default_tempo": 500000,
            "timed_notes": [],  # List of {"note", "start_sec", "duration_sec", "velocity", "track", "channel"}
        }

    def parse_midi_file(self, midi_file_path: str) -> Dict[str, Any]:
        try:
            return self._extracted_from_parse_midi_file_3(midi_file_path)
        except FileNotFoundError as e:
            logging.error(f"MIDI file not found: {midi_file_path}")
            raise MIDIAnalysisError(f"MIDI file not found: {midi_file_path}") from e
        except Exception as e:
            logging.exception(
                f"Unexpected error parsing MIDI file '{midi_file_path}': {e}"
            )
            raise MIDIAnalysisError(f"Error parsing MIDI file: {e}") from e

    def _extracted_from_parse_midi_file_3(self, midi_file_path):
        self.midi_analysis = self._get_default_analysis()  # Reset analysis
        logging.debug(f"Attempting to parse MIDI file: {midi_file_path}")
        try:
            midi_file = mido.MidiFile(midi_file_path)
        except Exception as e:
            logging.exception(f"Mido failed to open or parse {midi_file_path}")
            raise MIDIAnalysisError(
                f"Mido parsing error for {midi_file_path}: {e}"
            ) from e

        logging.debug(
            f"Mido opened file. Type: {midi_file.type}, Length: {midi_file.length:.2f}s, Ticks/Beat: {midi_file.ticks_per_beat}"
        )
        if midi_file.ticks_per_beat is None or midi_file.ticks_per_beat == 0:
            logging.warning(
                "MIDI file has invalid or missing ticks_per_beat. Using default 480."
            )
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
                if msg.is_meta and msg.type == "set_tempo":
                    current_tempo = msg.tempo
                    logging.debug(
                        f"Found initial tempo {current_tempo} ({mido.tempo2bpm(current_tempo):.2f} BPM)"
                    )
                    initial_tempo_found = True
                    break
            if initial_tempo_found:
                break
        if not initial_tempo_found:
            logging.debug(
                f"No initial tempo found, using default {current_tempo} ({mido.tempo2bpm(current_tempo):.2f} BPM)"
            )

        # --- Process Tracks ---
        for track_num, track in enumerate(midi_file.tracks):
            track_name = track.name or f"Track {track_num}"
            self.midi_analysis["tracks"].append(track_name)
            logging.debug(f"Processing {track_name}...")
            absolute_tick_track = 0
            # active_notes_ticks: { (note, channel) : (start_tick, start_tempo, velocity) }
            active_notes_ticks = {}
            track_tempo = (
                current_tempo  # Each track starts with the initial global tempo
            )

            for msg in track:
                # --- Time Calculation ---
                delta_ticks = msg.time
                absolute_tick_track += delta_ticks
                # Use the track's current tempo for time calculation at this point
                current_time_sec_at_msg = mido.tick2second(
                    absolute_tick_track,
                    self.midi_analysis["ticks_per_beat"],
                    track_tempo,
                )

                # --- Meta Messages ---
                if msg.is_meta:
                    if msg.type == "key_signature":
                        self.midi_analysis["key_signature_changes"].append(
                            {
                                "time_seconds": current_time_sec_at_msg,
                                "tick": absolute_tick_track,
                                "key": msg.key,
                            }
                        )
                    elif msg.type == "set_tempo":
                        old_tempo = track_tempo
                        track_tempo = (
                            msg.tempo
                        )  # Update tempo for subsequent calculations IN THIS TRACK
                        bpm = mido.tempo2bpm(track_tempo)
                        logging.debug(
                            f"    T{track_num} Tempo Change at tick {absolute_tick_track}: {old_tempo} -> {track_tempo} ({bpm:.2f} BPM)"
                        )
                        self.midi_analysis["tempo_changes"].append(
                            {
                                "time_seconds": current_time_sec_at_msg,
                                "tick": absolute_tick_track,
                                "tempo": track_tempo,
                                "bpm": bpm,
                            }
                        )
                        # Re-calculate current time based on the NEW tempo from this point forward
                        current_time_sec_at_msg = mido.tick2second(
                            absolute_tick_track,
                            self.midi_analysis["ticks_per_beat"],
                            track_tempo,
                        )
                    elif msg.type == "time_signature":
                        self.midi_analysis["time_signature_changes"].append(
                            {
                                "time_seconds": current_time_sec_at_msg,
                                "tick": absolute_tick_track,
                                "numerator": msg.numerator,
                                "denominator": msg.denominator,
                            }
                        )
                    # Ignore other meta messages for timed notes

                elif msg.type == "program_change":
                    self.midi_analysis["program_changes"][track_num].append(
                        {
                            "time_seconds": current_time_sec_at_msg,
                            "tick": absolute_tick_track,
                            "program": msg.program,
                            "channel": msg.channel,
                        }
                    )

                elif msg.type == "note_on" and msg.velocity > 0:
                    note_key = (msg.note, msg.channel)
                    if note_key in active_notes_ticks:
                        # Overlap: Log warning but keep the original note active for timing simplicity
                        logging.warning(
                            f"    T{track_num} Note On received for already active key {note_key} at tick {absolute_tick_track}. Ignoring this Note On for timing, waiting for Note Off."
                        )
                        # Optional: Could end the previous note here and start a new one if strict re-triggering is needed
                    else:
                        # New Note On: Store start tick, tempo AT THE START, velocity
                        active_notes_ticks[note_key] = (
                            absolute_tick_track,
                            track_tempo,
                            msg.velocity,
                        )
                        logging.debug(
                            f"    T{track_num} Note On: {note_key} Vel: {msg.velocity} at tick {absolute_tick_track}, tempo {track_tempo}"
                        )
                        # Update basic stats
                        self.midi_analysis["unique_notes"].add(msg.note)
                        self.midi_analysis["note_distribution"][msg.note] += 1
                        self.midi_analysis["total_notes"] += 1

                elif msg.type == "note_off" or (
                    msg.type == "note_on" and msg.velocity == 0
                ):
                    note_key = (msg.note, msg.channel)
                    if note_key in active_notes_ticks:
                        # Found matching Note On: Calculate duration and times
                        start_tick, start_tempo, start_velocity = (
                            active_notes_ticks.pop(note_key)
                        )  # Remove from active
                        end_tick = absolute_tick_track
                        duration_ticks = end_tick - start_tick

                        # CRITICAL: Calculate start time and duration using the TEMPO AT THE START of the note
                        note_start_time_sec = mido.tick2second(
                            start_tick,
                            self.midi_analysis["ticks_per_beat"],
                            start_tempo,
                        )
                        duration_seconds = mido.tick2second(
                            duration_ticks,
                            self.midi_analysis["ticks_per_beat"],
                            start_tempo,
                        )

                        logging.debug(
                            f"    T{track_num} Note Off: {note_key} at tick {end_tick}. Start tick: {start_tick}, Dur Ticks: {duration_ticks}, Start Tempo: {start_tempo}, Dur Sec: {duration_seconds:.3f}, Start Sec: {note_start_time_sec:.3f}"
                        )

                        # Add to timed_notes if duration is positive
                        if (
                            duration_seconds > 0.001
                        ):  # Use a small threshold to avoid zero-duration noise
                            timed_notes.append(
                                {
                                    "note": msg.note,
                                    "start_sec": note_start_time_sec,
                                    "duration_sec": duration_seconds,
                                    "velocity": start_velocity,
                                    "track": track_num,
                                    "channel": msg.channel,
                                }
                            )
                            # Update global duration stats
                            self.midi_analysis["note_duration_stats"][
                                "min_duration"
                            ] = min(
                                self.midi_analysis["note_duration_stats"][
                                    "min_duration"
                                ],
                                duration_seconds,
                            )
                            self.midi_analysis["note_duration_stats"][
                                "max_duration"
                            ] = max(
                                self.midi_analysis["note_duration_stats"][
                                    "max_duration"],
                                duration_seconds,
                            )
                        else:
                            logging.warning(
                                f"    T{track_num} Zero or near-zero duration ({duration_seconds:.4f}) calculated for note {note_key} ending at tick {end_tick}. Start tick {start_tick}. Ignoring note."
                            )
                    else:  # Note off without matching Note On
                        logging.debug(
                            f"    T{track_num} Ignoring Note Off for {note_key} at tick {absolute_tick_track} - no matching active Note On found."
                        )

            # --- End of Track: Handle Notes Still On ---
            # Use the final tempo of this track for end time calculation
            end_time_sec_track = mido.tick2second(
                absolute_tick_track, self.midi_analysis["ticks_per_beat"], track_tempo
            )
            if active_notes_ticks:
                logging.debug(
                    f"  End of {track_name}: Handling {len(active_notes_ticks)} notes still active (no Note Off found). Track end tick {absolute_tick_track}"
                )
                for note_key, (start_tick, start_tempo, start_velocity) in list(
                    active_notes_ticks.items()
                ):
                    duration_ticks = absolute_tick_track - start_tick
                    # Calculate duration and start time using the tempo WHEN THE NOTE STARTED
                    note_start_time_sec = mido.tick2second(
                        start_tick, self.midi_analysis["ticks_per_beat"], start_tempo
                    )
                    duration_seconds = mido.tick2second(
                        duration_ticks,
                        self.midi_analysis["ticks_per_beat"],
                        start_tempo,
                    )

                    note, channel = note_key
                    logging.debug(
                        f"    Ending active note {note_key} at track end. Start tick: {start_tick}, Dur Ticks: {duration_ticks}, Start Tempo: {start_tempo}, Dur Sec: {duration_seconds:.3f}, Start Sec: {note_start_time_sec:.3f}"
                    )
                    if duration_seconds > 0.001:
                        timed_notes.append(
                            {
                                "note": note,
                                "start_sec": note_start_time_sec,
                                "duration_sec": duration_seconds,
                                "velocity": start_velocity,
                                "track": track_num,
                                "channel": channel,
                            }
                        )
                        self.midi_analysis["note_duration_stats"]["min_duration"] = min(
                            self.midi_analysis["note_duration_stats"]["min_duration"],
                            duration_seconds,
                        )
                        self.midi_analysis["note_duration_stats"]["max_duration"] = max(
                            self.midi_analysis["note_duration_stats"]["max_duration"],
                            duration_seconds,
                        )
                    else:
                        logging.warning(
                            f"    Zero duration calculated for note {note_key} active at track end. Ignoring."
                        )
                    active_notes_ticks.pop(note_key)  # Remove handled note

            absolute_tick_max = max(absolute_tick_max, absolute_tick_track)

        # --- Final Calculations ---
        # Determine the tempo active at the very end of the longest track for total duration
        final_tempo = self.midi_analysis["default_tempo"]
        if self.midi_analysis["tempo_changes"]:
            if last_tempo_change := max(
                (
                    tc
                    for tc in self.midi_analysis["tempo_changes"]
                    if tc["tick"] <= absolute_tick_max
                ),
                key=lambda tc: tc["tick"],
                default=None,
            ):
                final_tempo = last_tempo_change["tempo"]
            elif (
                self.midi_analysis["tempo_changes"][0]["tick"] > 0
            ):  # If first change is after tick 0
                # Use default tempo if max tick is before first change
                if absolute_tick_max < self.midi_analysis["tempo_changes"][0]["tick"]:
                    final_tempo = self.midi_analysis["default_tempo"]
                else:  # Should be covered by max() logic, but as fallback
                    final_tempo = self.midi_analysis["tempo_changes"][0][
                        "tempo"
                    ]  # Or maybe error?
            else:
                final_tempo = self.midi_analysis["tempo_changes"][0]["tempo"]

        self.midi_analysis["total_duration"] = mido.tick2second(
            absolute_tick_max, self.midi_analysis["ticks_per_beat"], final_tempo
        )

        if timed_notes:
            total_duration_sum_sec = sum(n["duration_sec"] for n in timed_notes)
            self.midi_analysis["note_duration_stats"]["avg_duration"] = (
                total_duration_sum_sec / len(timed_notes)
            )
        else:
            self.midi_analysis["note_duration_stats"]["avg_duration"] = 0.0
        if self.midi_analysis["note_duration_stats"]["min_duration"] == float("inf"):
            self.midi_analysis["note_duration_stats"]["min_duration"] = 0.0

        # Sort timed notes by start time and store
        self.midi_analysis["timed_notes"] = sorted(
            timed_notes, key=lambda x: x["start_sec"]
        )
        logging.info(
            f"Finished parsing. Total timed notes extracted: {len(self.midi_analysis['timed_notes'])}"
        )
        if (
            not self.midi_analysis["timed_notes"]
            and self.midi_analysis["total_notes"] > 0
        ):
            logging.warning(
                "Parsing found Note On events but resulted in zero timed notes (check durations/logic)."
            )

        return self.midi_analysis

    # --- Report Generation Methods (Unchanged) ---
    def generate_midi_analysis_report(self) -> str:
        analysis = self.midi_analysis
        if not analysis or analysis.get("filename") is None:
            return "No MIDI analysis data available."
        report = (
            f"### MIDI File Analysis Report: {analysis.get('filename', 'N/A')} ###\n\n"
        )
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
        track_names = analysis.get("tracks", [])
        info += f"Tracks: {', '.join(track_names) if track_names else 'N/A'}\n\n"
        return info

    def _generate_note_info(self, analysis):
        timed_notes_count = len(analysis.get("timed_notes", []))
        info = f"Total Notes Played (raw NoteOn): {analysis['total_notes']}\n"
        info += f"Notes in Sequence (Timed): {timed_notes_count}\n"
        info += f"Unique Notes Used: {len(analysis['unique_notes'])}\n"
        if analysis["unique_notes"]:
            min_note, max_note = min(analysis["unique_notes"]), max(
                analysis["unique_notes"]
            )
            info += f"Note Range: {min_note} ({self._get_note_name_static(min_note)}) - {max_note} ({self._get_note_name_static(max_note)})\n\n"
            if analysis["note_distribution"]:
                sorted_notes = sorted(
                    analysis["note_distribution"].items(),
                    key=lambda item: item[1],
                    reverse=True,
                )[:5]
                info += (
                    "Most Frequent Notes (Top 5):\n"
                    + "\n".join(
                        [
                            f"  Note {n} ({self._get_note_name_static(n)}): {c} times"
                            for n, c in sorted_notes
                        ]
                    )
                    + "\n"
                )
            else:
                info += "Most Frequent Notes: N/A\n"
        else:
            info += "Note Range: N/A\nMost Frequent Notes: N/A\n"
        return info + "\n"

    def _generate_duration_stats(self, analysis):
        stats = analysis["note_duration_stats"]
        min_d = (
            f"{stats['min_duration']:.4f}"
            if stats["min_duration"] is not None
            and stats["min_duration"] != float("inf")
            else "N/A"
        )
        max_d = (
            f"{stats['max_duration']:.4f}"
            if stats["max_duration"] is not None
            else "N/A"
        )
        avg_d = (
            f"{stats['avg_duration']:.4f}"
            if stats["avg_duration"] is not None
            else "N/A"
        )
        s = "Note Duration Statistics (Timed Notes):\n"
        s += f"  Min Duration: {min_d}\n  Max Duration: {max_d}\n  Avg Duration: {avg_d}\n\n"
        return s

    def _generate_tempo_changes(self, analysis):
        changes = "Tempo Changes (BPM):\n"
        default_bpm = mido.tempo2bpm(analysis.get("default_tempo", 500000))
        if tempo_events := sorted(
            analysis.get("tempo_changes", []), key=lambda x: x["tick"]
        ):
            last_bpm = -1  # Force printing the first one
            # Report default if first change is not at tick 0
            if tempo_events[0]["tick"] > 0:
                changes += f"  Initial Tempo (Default): {default_bpm:.2f} BPM (until tick {tempo_events[0]['tick']})\n"
                last_bpm = default_bpm  # Set last_bpm so we only report the first event if it's different

            for change in tempo_events:
                bpm = change.get("bpm", mido.tempo2bpm(change["tempo"]))
                if abs(bpm - last_bpm) > 0.01:  # Check for actual change
                    changes += f"  Tick {change['tick']} ({change['time_seconds']:.2f}s): {bpm:.2f} BPM\n"
                    last_bpm = bpm
        else:
            changes += f"  No tempo changes detected (Using default/initial: {default_bpm:.2f} BPM).\n"
        return changes + "\n"

    def _generate_time_signature_changes(self, analysis):
        changes = "Time Signature Changes:\n"
        ts_events = sorted(
            analysis.get("time_signature_changes", []), key=lambda x: x["tick"]
        )
        if ts_events:
            last_sig = None
            # Report assumed 4/4 if first change not at tick 0
            if ts_events[0]["tick"] > 0:
                changes += (
                    f"  Initial (Assumed): 4/4 (until tick {ts_events[0]['tick']})\n"
                )
                last_sig = "4/4"

            for change in ts_events:
                current_sig = f"{change['numerator']}/{change['denominator']}"
                # Prefix only really needed if first isn't at tick 0, otherwise Tick 0 is fine
                prefix = f"Tick {change['tick']} ({change['time_seconds']:.2f}s):"
                if current_sig != last_sig:
                    changes += f"  {prefix} {current_sig}\n"
                    last_sig = current_sig
        else:
            changes += "  No time signature changes detected (Assumed 4/4).\n"
        return changes + "\n"

    def _generate_key_signature_changes(self, analysis):
        changes = "Key Signature Changes:\n"
        ks_events = sorted(
            analysis.get("key_signature_changes", []), key=lambda x: x["tick"]
        )
        if ks_events:
            last_key = None
            # Report assumed C Major if first change not at tick 0
            if ks_events[0]["tick"] > 0:
                changes += f"  Initial (Assumed): C Major / A Minor (until tick {ks_events[0]['tick']})\n"
                last_key = "C"  # Assuming C is the default representation

            for change in ks_events:
                if change["key"] != last_key:
                    prefix = f"Tick {change['tick']} ({change['time_seconds']:.2f}s):"
                    changes += f"  {prefix} {change['key']}\n"
                    last_key = change["key"]
        else:
            changes += (
                "  No key signature changes detected (Assumed C Major / A Minor).\n"
            )
        return changes + "\n"

    def _generate_program_changes(self, analysis):
        changes = "Program (Instrument) Changes:\n"
        prog_changes_dict = analysis.get("program_changes", {})
        if prog_changes_dict:
            for track_num, changes_list in sorted(prog_changes_dict.items()):
                track_name = (
                    analysis["tracks"][track_num]
                    if track_num < len(analysis["tracks"])
                    else f"Track {track_num}"
                )
                if changes_list:  # Only report tracks with actual changes
                    changes += f"  {track_name}:\n"
                    last_prog = -1  # Track-specific last program
                    sorted_changes_list = sorted(changes_list, key=lambda x: x["tick"])
                    for change in sorted_changes_list:
                        if change["program"] != last_prog:
                            changes += f"    Tick {change['tick']} ({change['time_seconds']:.2f}s), Ch {change['channel']}: Prog {change['program']}\n"
                            last_prog = change["program"]
        else:
            changes += "  No program changes detected.\n"
        return changes

    @staticmethod
    def _get_note_name_static(note: int) -> str:
        if not (0 <= note <= 127):
            return "??"
        names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        octave = (note // 12) - 1
        return f"{names[note % 12]}{octave}"


# =========================================================
# END OF AdvancedMIDIParser Class
# =========================================================


# --- Base Piano Trainer Class ---
class PianoTrainer:
    """Base class (can be empty or have common methods)"""

    def __init__(self):
        logging.debug("Initializing Base PianoTrainer...")

    def _render_ui(self):
        pass  # To be implemented by subclass

    def run(self, mode=None, midi_file=None):
        logging.debug("Running Base PianoTrainer...")

    def _cleanup(self):
        logging.debug("Cleaning up Base PianoTrainer...")


# --- Enhanced UI and Core Logic ---
def enhance_piano_trainer_ui(BasePianoTrainer):
    """Decorator/Function to create the Enhanced UI class"""







    class EnhancedPianoTrainerUI(BasePianoTrainer):
        def __init__(self, *args, **kwargs):
            # --- Basic Init ---
            logging.info("Initializing EnhancedPianoTrainerUI...")
            try:
                pygame.init()
                pygame.midi.init()
                pygame.font.init()
                pygame.mixer.init()  # Initialize pygame.mixer for sound playback
            except Exception as e:
                logging.exception("Failed to initialize Pygame modules.")
                raise RuntimeError("Pygame initialization failed.") from e

            self.note_sounds = self._load_note_sounds()  # Load note sounds

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

            super().__init__(*args, **kwargs)  # Call base class init

            # --- Core Components ---
            self.midi_parser = AdvancedMIDIParser()
            self.midi_analysis = None
            self.performance_history = []  # For potential future stats
            self.music_theory = MusicTheory()

            # --- Chord Tracking ---
            # Track recently pressed keys for chord detection
            self.active_chord_keys = set()    # Currently pressed keys
            self.last_chord_press_time = 0    # When the most recent key was pressed

            # --- Piano Visualization ---
            self.white_key_width = 40
            self.white_key_height = 200
            self.black_key_width = 26
            self.black_key_height = 120
            self.first_note = 36
            self.last_note = 96  # C2-C7 (Adjust as needed)
            num_white_keys = sum(
                not self.is_black_key(n)
                for n in range(self.first_note, self.last_note + 1)
            )
            required_piano_width = num_white_keys * self.white_key_width
            self.piano_start_x = max(
                20, (self.screen_width - required_piano_width) // 2
            )
            self.piano_start_y = self.screen_height - self.white_key_height - 20

            # --- Note State ---
            self.active_notes = defaultdict(
                lambda: False
            )  # Tracks notes currently held down via MIDI IN

            # --- Fonts ---
            try:
                self.font = pygame.font.SysFont("Arial", 24)
                self.small_font = pygame.font.SysFont("Arial", 18)
                self.key_font = pygame.font.SysFont("Arial", 12)
                self.title_font = pygame.font.SysFont("Arial", 36, bold=True)
                self.report_font = pygame.font.SysFont("Courier New", 14)
                self.next_note_font = pygame.font.SysFont("Arial", 28, bold=True)
                self.falling_note_font = pygame.font.SysFont(
                    "Arial", 10
                )  # Font for labels inside falling notes
            except Exception:
                logging.warning("System fonts not found, using default Pygame fonts.")
                self.font = pygame.font.Font(None, 30)
                self.small_font = pygame.font.Font(None, 24)
                self.key_font = pygame.font.Font(None, 18)
                self.title_font = pygame.font.Font(None, 40)
                self.report_font = pygame.font.Font(None, 20)
                self.next_note_font = pygame.font.Font(None, 34)
                self.falling_note_font = pygame.font.Font(
                    None, 16
                )  # Default font for labels

            self.note_names_map = {
                i: AdvancedMIDIParser._get_note_name_static(i) for i in range(128)
            }

            # --- MIDI Input ---
            self.midi_input = None
            self.midi_device_name = None
            self.setup_midi_input()  # Attempt to connect to MIDI input on startup

            # --- Colors ---
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
                "key_text_black": (200, 200, 200),
                "pressed_black_key_border": (255, 255, 255),
                "target_line": (255, 80, 80, 200),  # Red target line
                "falling_note_upcoming": (
                    80,
                    80,
                    220,
                    200,
                ),  # Bluish upcoming notes (with alpha)
                "falling_note_active": (
                    240,
                    240,
                    100,
                    220,
                ),  # Yellowish active notes (with alpha)
                "falling_note_hit": (
                    80,
                    220,
                    80,
                    180,
                ),  # Greenish hit notes (slightly transparent)
                "falling_note_missed": (
                    180,
                    100,
                    100,
                    180,
                ),  # Reddish missed notes (slightly transparent)
                "falling_note_border": (
                    200,
                    200,
                    220,
                ),  # Light border for upcoming/active
                "next_note_text": (220, 220, 255),
            }

            # --- App State ---
            self.running = False  # Flag for main loop
            self.clock = pygame.time.Clock()
            self.mode = "freestyle"  # Current operating mode
            self.midi_analysis_report_str = None  # Stores report for display
            self.challenge_difficulty = "intermediate"  # Placeholder

            # --- Learning Mode State ---
            self.learning_mode_active = False
            self.learning_content: List[FallingNote] = (
                []
            )  # Holds the sequence of FallingNote objects
            self.learning_start_time_ms = 0  # When the sequence prep time ends
            self.current_sequence_time_sec = (
                0.0  # Time elapsed within the learning sequence (pauses!)
            )
            self.target_line_y = self.piano_start_y - TARGET_LINE_Y_OFFSET
            self.key_rect_map: Dict[int, pygame.Rect] = {}  # Cache for key rectangles
            self._update_key_rect_map()  # Populate the cache
            self.upcoming_notes_display: List[str] = (
                []
            )  # Note names for the "Next:" display
            self.score = 0
            self.last_played_event = None  # Info about the last MIDI note played
            self.is_waiting_for_hit = False  # NEW: Flag to pause time advancement
            self.notes_to_hit_this_step: Set[FallingNote] = (
                set()
            )  # NEW: Notes currently required at the target line

            # Ensure chord_notes is initialized as a dictionary
            self.chord_notes = defaultdict(list)

            logging.info("EnhancedPianoTrainerUI Initialized successfully.")

        def _load_note_sounds(self) -> Dict[int, pygame.mixer.Sound]:
            """Load sound files for each MIDI note using note names."""
            note_sounds = {}
            note_names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
            for note in range(21, 108):  # MIDI notes for A0 to G7
                try:
                    octave = (note // 12) - 1
                    note_name = note_names[note % 12]
                    sound_path = f"samples/piano_{note_name}{octave}.wav"
                    note_sounds[note] = pygame.mixer.Sound(sound_path)
                except FileNotFoundError as e:
                    logging.warning(f"Sound file not found for note {note} at {sound_path}: {e}")
                except Exception as e:
                    logging.exception(f"Error loading sound for note {note} at {sound_path}: {e}")
            return note_sounds

        def _update_key_rect_map(self):
            """Calculate and cache screen rectangles for all displayed piano keys."""
            self.key_rect_map.clear()
            logging.debug("Updating key rectangle map...")
            for note in range(self.first_note, self.last_note + 1):
                rect = self._calculate_key_rect(note)  # Use internal calculation method
                if rect:
                    self.key_rect_map[note] = rect
            logging.debug(
                f"Key rectangle map updated with {len(self.key_rect_map)} keys."
            )

        def run(
            self,
            mode: Optional[str] = "freestyle",
            midi_file: Optional[str] = None,
            difficulty: str = "intermediate",
            learning_type: str = "scale",
            root_note: int = 60,
            scale_chord_type: str = "major",
        ) -> None:
            """Main application loop."""
            logging.info(
                f"Starting Piano Trainer. Mode: '{mode}', MIDI file: '{midi_file}', Difficulty: '{difficulty}', Learning: {learning_type} {scale_chord_type} from {root_note}"
            )

            requested_mode = mode
            self.running = True
            self.challenge_difficulty = difficulty

            # --- Parse MIDI file FIRST if provided ---
            if midi_file:
                logging.info(f"MIDI file provided: {midi_file}. Parsing...")
                try:
                    self.midi_analysis = self.midi_parser.parse_midi_file(midi_file)
                    self.midi_analysis_report_str = (
                        self.midi_parser.generate_midi_analysis_report()
                    )
                    # Print report to console unless in learning mode (too spammy during gameplay)
                    if requested_mode != "learning":
                        print("\n" + "=" * 20 + " MIDI Analysis Report " + "=" * 20)
                        print(self.midi_analysis_report_str)
                        print("=" * 50 + "\n")

                    if self.midi_analysis:
                        timed_notes_list = self.midi_analysis.get("timed_notes")
                        if timed_notes_list is not None:
                            logging.info(
                                f"Successfully parsed MIDI. Found {len(timed_notes_list)} timed notes."
                            )
                            if (
                                not timed_notes_list
                                and self.midi_analysis.get("total_notes", 0) > 0
                            ):
                                logging.warning(
                                    "MIDI parsing finished, but the 'timed_notes' list is EMPTY despite NoteOns found. Check MIDI format or parser logic."
                                )
                        else:
                            logging.error(
                                "Parsing finished, but 'timed_notes' key is missing or None in analysis results!"
                            )
                            self.midi_analysis = (
                                None  # Invalidate analysis if timed notes are missing
                            )
                    else:
                        logging.warning(
                            "MIDI parsing resulted in self.midi_analysis being None."
                        )

                except MIDIAnalysisError as e:
                    logging.error(f"Error during MIDI analysis: {e}")
                    self.midi_analysis_report_str = (
                        f"Error loading or parsing MIDI:\n{e}"
                    )
                    self.midi_analysis = None  # Ensure analysis is None on error
                    # If error prevents intended mode, switch to freestyle
                    if requested_mode == "learning" and learning_type == "midi":
                        logging.error(
                            "Cannot enter MIDI learning mode due to parsing error. Switching to freestyle."
                        )
                        requested_mode = "freestyle"
                    elif requested_mode == "analysis_view":
                        logging.error(
                            "Cannot enter analysis view mode due to parsing error. Switching to freestyle."
                        )
                        requested_mode = "freestyle"
                except Exception as e:
                    logging.exception(
                        f"Unexpected error during MIDI file processing: {e}"
                    )
                    self.midi_analysis_report_str = (
                        f"Unexpected error processing MIDI:\n{e}"
                    )
                    self.midi_analysis = None
                    if requested_mode in ["learning", "analysis_view"]:
                        requested_mode = "freestyle"

            # --- Determine Final Mode ---
            # Switch to analysis view if a MIDI was successfully parsed and mode wasn't learning
            if requested_mode != "learning" and self.midi_analysis:
                self.mode = "analysis_view"
                logging.info(
                    "Mode set to 'analysis_view' as MIDI file was successfully parsed."
                )
            else:
                self.mode = requested_mode  # Use the requested mode (or the fallback from error handling)

            # --- Main Loop ---
            try:
                self._setup_current_mode(
                    learning_type, root_note, scale_chord_type, midi_file
                )
                game_start_time_ms = pygame.time.get_ticks()  # Overall start time

                while self.running:
                    current_frame_time_ms = (
                        pygame.time.get_ticks()
                    )  # Time at start of frame

                    # 1. Handle Events (Input, Quit)
                    self._handle_events()
                    if not self.running:
                        break  # Exit loop if quit event occurred

                    # 2. Process MIDI Input
                    self.process_midi_input(current_frame_time_ms)

                    # 3. Update Game State (Learning Mode)
                    if self.learning_mode_active:
                        self.update_learning_mode(current_frame_time_ms)

                    # 4. Render UI
                    self._render_ui()

                    # 5. Display Update and Tick Clock
                    pygame.display.flip()
                    self.clock.tick(60)  # Limit FPS to 60

            except Exception as e:
                logging.exception("Unexpected error in main loop.")
                print(
                    f"\nFATAL ERROR in main loop: {e}\nCheck piano_trainer.log for details.",
                    file=sys.stderr,
                )
            finally:
                self._cleanup()

        def _setup_current_mode(
            self,
            learning_type="scale",
            root_note=60,
            scale_chord="major",
            midi_file=None,
        ):
            """Initializes state based on the current self.mode."""
            logging.info(f"Setting up mode: {self.mode}")
            self.learning_mode_active = False
            self.learning_content.clear()
            self.score = 0
            self.is_waiting_for_hit = False  # Reset wait state
            self.notes_to_hit_this_step.clear()  # Clear required notes
            self.current_sequence_time_sec = (
                -PREP_TIME_SEC
            )  # Reset sequence time for countdown

            if self.mode == "learning":
                self.learning_mode_active = True
                self._generate_learning_sequence(
                    learning_type, root_note, scale_chord, midi_file
                )
                if not self.learning_content:
                    logging.error(
                        "Learning mode selected, but failed to generate any learning content. Switching to freestyle."
                    )
                    self.mode = "freestyle"
                    self.learning_mode_active = False
                else:
                    # Start time marks the beginning of the *sequence*, after prep time
                    self.learning_start_time_ms = pygame.time.get_ticks() + (
                        PREP_TIME_SEC * 1000
                    )
                    logging.info(
                        f"Learning mode setup complete. {len(self.learning_content)} notes loaded. Prep time: {PREP_TIME_SEC}s."
                    )

            elif self.mode == "analysis_view":
                if not self.midi_analysis and not self.midi_analysis_report_str:
                    logging.warning(
                        "Analysis view mode selected, but no analysis data or report string available."
                    )
                    # If report string exists (e.g., from error), show that, otherwise show generic message
                    if not self.midi_analysis_report_str:
                        self.midi_analysis_report_str = (
                            "No MIDI file loaded or analysis available."
                        )
                elif not self.midi_analysis_report_str:
                    logging.warning(
                        "Analysis view mode: analysis object exists, but report string is missing. Regenerating."
                    )
                    if self.midi_analysis:
                        self.midi_analysis_report_str = (
                            self.midi_parser.generate_midi_analysis_report()
                        )
                    else:
                        self.midi_analysis_report_str = (
                            "Error: Analysis data missing, cannot generate report."
                        )

            elif self.mode == "freestyle":
                logging.info("Freestyle mode active.")

            else:
                logging.warning(
                    f"Unknown mode '{self.mode}' encountered during setup. Defaulting to freestyle."
                )
                self.mode = "freestyle"

        def _generate_learning_sequence(
            self, type="scale", root=60, scale_chord="major", midi_file=None
        ):
            """Generates the sequence of FallingNote objects for learning mode."""
            self.learning_content.clear()
            logging.info(
                f"Generating learning sequence. Type: {type}, Root: {root}, Scale/Chord: {scale_chord}, MIDI: {midi_file}"
            )
            current_time_sequence = 0.0  # Time within the sequence
            notes_data_source = []  # Source of note data

            if type == "midi":
                if (
                    midi_file
                    and self.midi_analysis
                    and (timed_notes := self.midi_analysis.get("timed_notes"))
                    and isinstance(timed_notes, list)
                    and timed_notes
                ):
                    logging.info(f"Using timed notes from MIDI: {midi_file}")
                    notes_data_source = timed_notes
                else:
                    # Fallback logic (same as before)
                    logging.warning("Cannot use MIDI source. Falling back.")
                    type = "scale"

            if type == "chord_progression":
                logging.info("Generating chord progression: C-G-Am-F")
                # Format: list of (root_note, chord_type)
                progression = [(60, "maj"), (67, "maj"), (69, "min"), (65, "maj")]
                chord_duration = 1.8  # How long notes within a chord are held
                time_between_chords = (
                    2.0  # Time from start of one chord to start of next
                )

                for root_note, c_type in progression:
                    chord_notes = self.music_theory.generate_chord(root_note, c_type)
                    for note in chord_notes:
                        notes_data_source.append(
                            {
                                "note": note,
                                "start_sec": current_time_sequence,
                                "duration_sec": chord_duration,
                                # Velocity, track, channel could be added if needed, default otherwise
                            }
                        )
                    current_time_sequence += time_between_chords

            elif type == "scale":  # Handles fallback from MIDI failure too
                scale_notes = self.music_theory.generate_scale(
                    root, scale_chord, octaves=1
                )
                logging.info(
                    f"Generating scale: {scale_chord} starting at {root}. Notes: {scale_notes}"
                )
                note_duration = 0.4
                time_between_notes = 0.5
                for note in scale_notes:
                    notes_data_source.append(
                        {
                            "note": note,
                            "start_sec": current_time_sequence,
                            "duration_sec": note_duration,
                        }
                    )
                    current_time_sequence += time_between_notes

            # --- Create FallingNote Objects ---
            notes_added_count = 0
            notes_skipped_count = 0
            if not notes_data_source:
                logging.error("No note data was generated from any source.")
                return  # Exit if no data

            # Make notes start relative to the *end* of prep time
            # Note start times in notes_data_source are relative to 0
            min_start_time = min((n["start_sec"] for n in notes_data_source), default=0)

            for note_data in notes_data_source:
                fn = FallingNote(
                    note=note_data["note"],
                    start_time_sec=note_data["start_sec"],
                    duration_sec=note_data["duration_sec"],
                    target_y=self.target_line_y,
                    screen_height=self.screen_height,
                    velocity=note_data.get("velocity", 127)  # new: pass velocity if available
                )
                if self.first_note <= fn.note <= self.last_note:
                    self.learning_content.append(fn)
                    notes_added_count += 1
                else:
                    notes_skipped_count += 1
                    logging.debug(
                        f"Skipping note {fn._note_name_cache} ({fn.note}) - outside display range {self.first_note}-{self.last_note}."
                    )

            logging.info(
                f"Generated {notes_added_count} falling notes for sequence. Skipped {notes_skipped_count} notes outside display range."
            )
            if notes_added_count == 0 and notes_skipped_count > 0:
                logging.warning(
                    "All notes from the source were outside the piano display range."
                )
            elif notes_added_count == 0 and notes_skipped_count == 0:
                logging.warning("Note generation process yielded zero valid notes.")

            # Sort by start time just in case source wasn't ordered (MIDI parse should be, others maybe not)
            self.learning_content.sort(key=lambda fn: fn.start_time_sec)

        def update_learning_mode(self, current_frame_time_ms: int):
            """Update falling notes, check for hits, manage sequence time with waiting."""
            if not self.learning_mode_active:
                return

            # --- Calculate Time Delta ---
            dt_sec = self.clock.get_time() / 1000.0  # Time since last frame in seconds

            # --- Advance Sequence Time (Conditionally) ---
            previous_sequence_time_sec = self.current_sequence_time_sec
            if not self.is_waiting_for_hit:
                self.current_sequence_time_sec += dt_sec

                # Check if the countdown finished in this step
                if (
                    previous_sequence_time_sec < 0
                    and self.current_sequence_time_sec >= 0
                ):
                    logging.info("Countdown finished. Learning sequence starts now.")
                    # Adjust start time slightly to be exactly 0 if it overshot
                    self.current_sequence_time_sec = 0.0

            notes_to_remove_indices = []
            for i, fn in enumerate(self.learning_content):
                # Update position and state based on the *current* sequence time
                fn.update(self.current_sequence_time_sec, self.key_rect_map)

                # Cleanup: Mark notes for removal if they are finished
                is_hit_and_faded = (
                    fn.state == "hit"
                    and fn.hit_time_ms is not None
                    and current_frame_time_ms > fn.hit_time_ms + 500
                )  # Give hit notes time to display
                # Remove missed notes once they are well below the target line
                is_missed_and_past = (
                    fn.state == "missed"
                    and fn.rect is not None
                    and fn.rect.top > self.target_line_y + 100
                )
                is_invalid = (
                    fn.state == "invalid_key"
                )  # Remove invalid notes immediately

                if is_hit_and_faded or is_missed_and_past or is_invalid:
                    notes_to_remove_indices.append(i)

            # Remove notes that are done (iterate in reverse to avoid index issues)
            if notes_to_remove_indices:
                for index in sorted(notes_to_remove_indices, reverse=True):
                    if 0 <= index < len(self.learning_content):  # Bounds check
                        removed_note = self.learning_content.pop(index)
                        logging.debug(
                            f"Removed note {removed_note._note_name_cache} (State: {removed_note.state}, Index: {index}, Hit time: {removed_note.hit_time_ms}ms)"
                        )
                    else:
                        logging.warning(
                            f"Attempted to remove note at invalid index {index}. Content length: {len(self.learning_content)}"
                        )

            # --- Manage Waiting State ---
            # Only manage waiting if we are past the countdown
            if self.current_sequence_time_sec >= 0 and self.is_waiting_for_hit:
                all_required_notes_processed = True
                if not self.notes_to_hit_this_step:
                    logging.warning(
                        "WAITING CHECK: notes_to_hit_this_step is unexpectedly empty!"
                    )
                    all_required_notes_processed = (
                        True  # Assume processed if set is empty
                    )
                else:
                    for required_note in self.notes_to_hit_this_step:
                        # If any required note is still 'active' or 'upcoming', we are not done waiting
                        current_req_state = required_note.state
                        if current_req_state in ["active", "upcoming"]:
                            all_required_notes_processed = False
                            break  # No need to check further
                        # Note: 'hit' or 'missed' count as processed for resuming time

                if all_required_notes_processed:
                    self.is_waiting_for_hit = False
                    self.notes_to_hit_this_step.clear()

            # --- Update Upcoming Notes Display and required set ---
            next_note_time = float("inf")
            next_notes_candidates = []
            for fn in self.learning_content:
                if fn.state not in ["hit", "missed", "invalid_key"]:
                    if fn.start_time_sec < next_note_time:
                        next_note_time = fn.start_time_sec
                        next_notes_candidates = [fn]
                    elif fn.start_time_sec == next_note_time:
                        next_notes_candidates.append(fn)
            self.upcoming_notes_display = [self.get_note_name(fn.note) for fn in next_notes_candidates]
            # NEW: Only allow hit registration for the next (proper sequence) note(s)
            if next_notes_candidates:
                self.notes_to_hit_this_step = set(next_notes_candidates)
                self.is_waiting_for_hit = True
            else:
                self.notes_to_hit_this_step.clear()
                self.is_waiting_for_hit = False

        def _handle_events(self):
            """Processes Pygame events (Keyboard, Quit)."""
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                    logging.info("QUIT event received. Shutting down.")
                elif event.type == pygame.KEYDOWN:
                    if event.key in [pygame.K_q, pygame.K_ESCAPE]:
                        self.running = False
                        logging.info(
                            f"'{pygame.key.name(event.key)}' key pressed. Shutting down."
                        )
                    # Add other key bindings here if needed

        def _render_ui(self) -> None:
            """Draws all UI elements based on the current mode."""
            self._draw_background()
            self._draw_title()  # Title depends on mode

            # --- Mode-Specific Content ---
            if self.mode == "learning":
                self._draw_learning_ui()
            elif self.mode == "analysis_view":
                self._draw_analysis_report()
            elif self.mode == "freestyle":
                self._draw_freestyle_ui()
            else:  # Fallback for unknown mode
                self._draw_fallback_ui()

            # --- Common Elements ---
            self.draw_piano()  # Piano always visible
            self._draw_status_bar()  # MIDI status always visible

        def _draw_learning_ui(self):
            """Draws learning mode specific elements: target line, falling notes, score, next note."""
            # --- Draw Target Line ---
            white_key_indices = [
                self.count_white_keys_before(n)
                for n in range(self.first_note, self.last_note + 1)
                if not self.is_black_key(n)
            ]
            if white_key_indices:
                piano_visual_width = (max(white_key_indices) + 1) * self.white_key_width
                target_line_end_x = (
                    self.piano_start_x + piano_visual_width - 1
                )  # Adjust for border

                # Make the target line more noticeable when notes are active
                target_line_color = self.colors["target_line"]
                target_line_width = 3

                # Make the target line wider and brighter when waiting for hit
                if self.is_waiting_for_hit:
                    # Use a bolder, brighter line for the target
                    target_line_color = (255, 215, 0)  # Gold
                    target_line_width = 5

                pygame.draw.line(
                    self.screen,
                    target_line_color,
                    (self.piano_start_x, self.target_line_y),
                    (target_line_end_x, self.target_line_y),
                    target_line_width,
                )

            # --- Draw Falling Notes ---
            for fn in self.learning_content:
                fn.draw(
                    self.screen, self.colors, self.falling_note_font
                )  # Pass font for label

            # --- Draw Next Note Indicator ---
            y_pos_indicator = 90  # Y position for "Next:" text
            if self.upcoming_notes_display:
                next_notes_str = " / ".join(self.upcoming_notes_display)
                text_surf = self.next_note_font.render(
                    f"Next: {next_notes_str}", True, self.colors["next_note_text"]
                )
                text_rect = text_surf.get_rect(
                    centerx=self.screen_width // 2, y=y_pos_indicator
                )
                self.screen.blit(text_surf, text_rect)
            # Show complete message only after prep time and when content is empty
            elif (
                self.learning_mode_active
                and not self.learning_content
                and self.current_sequence_time_sec >= 0
            ):
                end_surf = self.next_note_font.render(
                    "Sequence Complete!", True, self.colors["text_highlight"]
                )
                end_rect = end_surf.get_rect(
                    centerx=self.screen_width // 2, y=y_pos_indicator
                )
                self.screen.blit(end_surf, end_rect)

            # --- Draw Score ---
            score_surf = self.font.render(
                f"Score: {self.score}", True, self.colors["text_highlight"]
            )
            score_rect = score_surf.get_rect(
                topright=(self.screen_width - 20, y_pos_indicator)
            )
            self.screen.blit(score_surf, score_rect)

            # --- Display countdown if in prep time ---
            if self.current_sequence_time_sec < 0:
                countdown_val = int(abs(self.current_sequence_time_sec))  # Initialize countdown_val
                countdown_val = max(1, countdown_val)  # Ensure minimum value is 1
                cd_surf = self.title_font.render(
                    f"{countdown_val}", True, self.colors["text_primary"]
                )
                cd_rect = cd_surf.get_rect(
                    center=(self.screen_width // 2, self.screen_height // 3)
                )
                self.screen.blit(cd_surf, cd_rect)

                # Add a "Get Ready" message
                ready_surf = self.font.render(
                    "Get Ready!", True, self.colors["text_highlight"]
                )
                ready_rect = ready_surf.get_rect(
                    centerx=self.screen_width // 2, y=self.screen_height // 3 + 60
                )
                self.screen.blit(ready_surf, ready_rect)

        def _draw_analysis_report(self):
            """Draws the MIDI analysis report text."""
            content_y_start = 90  # Below the title
            max_report_height = (
                self.screen_height - self.piano_start_y - 40
            )  # Space above piano
            max_report_width = self.screen_width * 0.85  # Use most of the width
            report_x = (self.screen_width - max_report_width) / 2
            line_height = self.report_font.get_linesize()
            max_lines_display = (
                int(max_report_height // line_height) - 1
            )  # Leave space for scroll hint

            if not self.midi_analysis_report_str:
                report_lines = ["Analysis report not available."]
            else:
                report_lines = self.midi_analysis_report_str.split("\n")

            y_offset = content_y_start
            lines_drawn = 0
            for line in report_lines:
                if lines_drawn >= max_lines_display:
                    break
                if y_offset + line_height > self.piano_start_y - 20:
                    break

                display_line = line
                try:
                    line_surf = self.report_font.render(
                        display_line, True, self.colors["text_secondary"]
                    )
                    if line_surf.get_width() > max_report_width:
                        avg_char_width = self.report_font.size("A")[0]
                        max_chars = (
                            int(max_report_width / avg_char_width)
                            if avg_char_width > 0
                            else 50
                        )
                        display_line = line[: max_chars - 3] + "..."
                        line_surf = self.report_font.render(
                            display_line, True, self.colors["text_secondary"]
                        )
                    self.screen.blit(line_surf, (report_x, y_offset))
                    y_offset += line_height
                    lines_drawn += 1
                except Exception as e:
                    logging.error(f"Error rendering report line: {e} - Line: '{line}'")
                    error_surf = self.report_font.render(
                        "...", True, self.colors["text_error"]
                    )
                    self.screen.blit(error_surf, (report_x, y_offset))
                    y_offset += line_height
                    lines_drawn += 1

            if len(report_lines) > lines_drawn:
                more_text = self.small_font.render(
                    "... (Full report printed to console)",
                    True,
                    self.colors["text_highlight"],
                )
                hint_y = y_offset + 5
                if hint_y < self.piano_start_y - 20:
                    self.screen.blit(more_text, (report_x, hint_y))

        def _draw_freestyle_ui(self):
            """Draws simple instructions for freestyle mode."""
            content_y_start = 90
            info_surf = self.font.render(
                "Freestyle Mode - Play your MIDI keyboard!",
                True,
                self.colors["text_highlight"],
            )
            info_rect = info_surf.get_rect(
                centerx=self.screen_width // 2, y=content_y_start
            )
            self.screen.blit(info_surf, info_rect)

            hint_surf = self.small_font.render(
                "Q/Esc: Quit", True, self.colors["text_secondary"]
            )
            hint_rect = hint_surf.get_rect(
                centerx=self.screen_width // 2, y=content_y_start + 40
            )
            self.screen.blit(hint_surf, hint_rect)

        def _draw_fallback_ui(self):
            """Draws a message indicating an unknown or error state."""
            content_y_start = 90
            error_surf = self.font.render(
                f"Error: Unknown Mode or State ({self.mode})",
                True,
                self.colors["text_error"],
            )
            error_rect = error_surf.get_rect(
                centerx=self.screen_width // 2, y=content_y_start
            )
            self.screen.blit(error_surf, error_rect)

        def _draw_background(self):
            """Draws a vertical gradient background."""
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
            """Draws the main title, varying based on mode."""
            mode_str = self.mode.replace("_", " ").title()
            title_text_str = f"Piano Trainer - {mode_str}"
            if self.mode == "learning":
                title_text_str += (
                    f" ({self.challenge_difficulty})"  # Add difficulty in learning mode
                )
            title_surf = self.title_font.render(
                title_text_str, True, self.colors["text_primary"]
            )
            title_rect = title_surf.get_rect(
                center=(self.screen_width // 2, 40)
            )  # Centered top
            self.screen.blit(title_surf, title_rect)

        def _draw_status_bar(self):
            """Draws the MIDI connection status at the bottom right."""
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
            """Scans for MIDI devices and attempts to connect to the first input found."""
            self.midi_input = None
            self.midi_device_name = None
            input_id = -1
            found_device_name = "Unknown Device"
            logging.info("Searching for MIDI input devices...")
            print("\n--- Searching for MIDI Inputs ---")
            try:
                pygame.midi.quit()
                pygame.midi.init()
                device_count = pygame.midi.get_count()
                if device_count == 0:
                    logging.warning("No MIDI devices found.")
                    print("No MIDI devices found.")
                    pygame.midi.quit()  # Quit if no devices
                    return

                print(f"Found {device_count} MIDI devices:")
                for i in range(device_count):
                    info = pygame.midi.get_device_info(i)
                    if info is None:
                        print(f"  Device #{i}: Error retrieving info")
                        continue
                    interface_bytes, name_bytes, is_input, is_output, is_opened = info
                    try:
                        decoded_name = name_bytes.decode("utf-8", errors="replace")
                    except Exception as decode_err:
                        decoded_name = f"Device {i} (Decode Error: {decode_err})"
                    print(
                        f"  Device #{i}: '{decoded_name}' (Input: {is_input}, Output: {is_output}, Opened: {is_opened})"
                    )
                    if is_input and input_id == -1:
                        input_id = i
                        found_device_name = decoded_name
            except pygame.midi.MidiException as e:
                logging.error(f"Pygame MIDI Error during scan: {e}", exc_info=True)
                print(f"\n--- Error Accessing MIDI System: {e} ---")
                pygame.midi.quit()
                return
            except Exception as e:
                logging.exception("Unexpected error during MIDI device enumeration.")
                print("\n--- Unexpected Error During MIDI Scan ---")
                pygame.midi.quit()
                return

            print("----------------------------------")

            if input_id != -1:
                try:
                    return self._extracted_from_setup_midi_input_943(found_device_name, input_id)
                except Exception as e:
                    logging.exception("Unexpected error during MIDI device enumeration.")
                    print("\n--- Unexpected Error During MIDI Scan ---")
                    pygame.midi.quit()
                    return


            if input_id != -1:
                try:
                    print(
                        f"Attempting connection to input device: '{found_device_name}' (ID: {input_id})"
                    )
                    self.midi_input = pygame.midi.Input(input_id)
                    self.midi_device_name = found_device_name
                    logging.info(
                        f"Successfully connected to MIDI input: {self.midi_device_name}")
                    print(f"Successfully connected: {self.midi_device_name}")
                except Exception as e:
                    logging.exception(
                        f"Failed to open MIDI device {input_id} ('{found_device_name}')"
                    )
                    print(
                        f"\n--- Error opening MIDI input '{found_device_name}': {e} ---"
                    )
                    self.midi_input = None
                    self.midi_device_name = None
                    pygame.midi.quit()  # Quit if open failed
            else:
                logging.warning(
                    "No suitable MIDI input device found among listed devices."
                )
                if device_count > 0:
                    print(
                        "\n--- No MIDI *Input* devices were found or selected. MIDI inactive. ---"
                    )
                pygame.midi.quit()  # Quit MIDI if no input selected

        def _extracted_from_setup_midi_input_943(self, found_device_name, input_id):
            print(
                f"Attempting connection to input device: '{found_device_name}' (ID: {input_id})"
            )
            self.midi_input = pygame.midi.Input(input_id)
            self.midi_device_name = found_device_name
            logging.info(
                f"Successfully connected to MIDI input: {self.midi_device_name}")
            pygame.midi.quit()
            return

        def is_black_key(self, note: int) -> bool:
            """Checks if a MIDI note number corresponds to a black key."""
            return (note % 12) in [1, 3, 6, 8, 10]

        def count_white_keys_before(self, note: int) -> int:
            """Counts the number of white keys from self.first_note up to (but not including) the given note."""
            count = 0
            if note < self.first_note:
                return 0
            for n in range(self.first_note, note):
                if not self.is_black_key(n):
                    count += 1
            return count

        def get_note_name(self, note: int) -> str:
            """Gets the standard note name (e.g., C#4) using the cached map."""
            return self.note_names_map.get(note, "??")

        def _calculate_key_rect(self, note: int) -> Optional[pygame.Rect]:
            """Calculates the screen rectangle for a given MIDI note number."""
            if not (self.first_note <= note <= self.last_note):
                return None
            if self.is_black_key(note):
                white_key_index = self.count_white_keys_before(note)
                x = (
                    self.piano_start_x
                    + (white_key_index * self.white_key_width)
                    - (self.black_key_width // 2)
                )
                return pygame.Rect(
                    x,
                    self.piano_start_y,
                    self.black_key_width,
                    self.black_key_height,
                )
            else:
                white_key_index = self.count_white_keys_before(note)
                x = self.piano_start_x + white_key_index * self.white_key_width
                return pygame.Rect(
                    x,
                    self.piano_start_y,
                    self.white_key_width - 1,
                    self.white_key_height,
                )

        def get_key_rect(self, note: int) -> Optional[pygame.Rect]:
            """Gets the cached screen rectangle for a given MIDI note."""
            return self.key_rect_map.get(note)

        # *** Method Updated with Debug Logging ***
        def process_midi_input(self, current_frame_time_ms: int):
            """Reads MIDI input, updates active_notes, and checks for hits in learning mode."""
            if not self.midi_input:
                return  # Do nothing if MIDI is not connected

            try:
                if self.midi_input.poll():  # Check if there are events waiting
                    midi_events = self.midi_input.read(128)  # Read up to 128 events
                    for event in midi_events:
                        data, timestamp_in = (
                            event  # timestamp_in is from pygame.midi internal clock
                        )
                        if not isinstance(data, (list, tuple)) or len(data) < 3:
                            logging.warning(f"Received malformed MIDI data: {data}")
                            continue  # Skip malformed event

                        status, note, velocity = data[0], data[1], data[2]
                        channel = status & 0x0F  # Extract MIDI channel (0-15)

                        # --- Note On Event ---
                        if 144 <= status <= 159 and velocity > 0:
                            self.active_notes[note] = (
                                True  # Mark key as pressed for drawing
                            )
                            play_time_ms = (
                                current_frame_time_ms  # Use frame time for consistency
                            )

                            # Store last played note info
                            self.last_played_event = {
                                "note": note,
                                "time_ms": play_time_ms,
                            }

                            # --- Enhanced Chord Detection ---
                            self.active_chord_keys.add(note)
                            self.last_chord_press_time = play_time_ms

                            # Log all currently pressed keys (for chord tracking)
                            chord_note_names = [self.get_note_name(n) for n in self.active_chord_keys]
                            logging.debug(f"Current chord keys: {chord_note_names} ({len(self.active_chord_keys)} keys)")
                            # --- End Enhanced Chord Detection ---

                            # Play the corresponding sound
                            if note in self.note_sounds:
                                self.note_sounds[note].play()

                            # --- Check Hit in Learning Mode ---
                            if self.learning_mode_active:
                                hit_registered_this_press = False
                                note_played_name = self.get_note_name(
                                    note
                                )  # Get name for logging

                                # --- Added Debug Logging ---
                                logging.debug(
                                    f"--- Processing MIDI NoteOn: {note_played_name} ({note}) ---"
                                )
                                if self.is_waiting_for_hit:
                                    logging.debug(
                                        f"  Mode: WAITING. Required notes: {[fn._note_name_cache for fn in self.notes_to_hit_this_step]}"
                                    )
                                else:
                                    logging.debug(f"  Mode: NOT WAITING.")
                                # --- End Added Logging ---

                                # Prioritize checking notes we are specifically waiting for
                                if self.is_waiting_for_hit:
                                    for fn in list(
                                        self.notes_to_hit_this_step
                                    ):  # Iterate copy just to be safe
                                        # --- Added Debug Logging ---
                                        logging.debug(
                                            f"  Checking required note: {fn._note_name_cache} (State: {fn.state})"
                                        )
                                        # --- End Added Debug Logging ---
                                        if fn.state in ["upcoming", "active"]:
                                            # --- Enhanced Debug Logging ---
                                            # Log before the hit check
                                            logging.debug(
                                                f"  Attempting to hit required note: {fn._note_name_cache} "
                                                f"(State: {fn.state}, Start: {fn.start_time_sec:.3f}s, "
                                                f"Current time: {self.current_sequence_time_sec:.3f}s)"
                                            )
                                            # --- End Enhanced Logging ---

                                            if fn.check_hit(note, play_time_ms):
                                                self.score += 10  # Basic score for a hit
                                                hit_registered_this_press = True

                                                # --- Enhanced Debug Logging ---
                                                logging.debug(
                                                    f"    >>> HIT REGISTERED via wait check for {fn._note_name_cache}. "
                                                    f"New state: {fn.state}, Hit time: {fn.hit_time_ms}ms"
                                                )
                                                # --- End Enhanced Logging ---
                                                # No break: allow hitting multiple simultaneous notes

                                # Check general list if not waiting OR if hit wasn't registered in wait step
                                if (
                                    not self.is_waiting_for_hit
                                    or not hit_registered_this_press
                                ):
                                    # --- Added Debug Logging ---
                                    if not self.is_waiting_for_hit:
                                        logging.debug(
                                            "  Checking general list (not waiting)..."
                                        )
                                    elif not hit_registered_this_press:
                                        logging.debug(
                                            "  Checking general list (hit not found in required set)..."
                                        )
                                    # --- End Added Logging ---
                                    for fn in self.learning_content:
                                        if (
                                            fn not in self.notes_to_hit_this_step
                                            and fn.state in ["upcoming", "active"]
                                        ):
                                            # --- Added Debug Logging ---
                                            logging.debug(
                                                f"  Checking general note: {fn._note_name_cache} (State: {fn.state})"
                                            )
                                            # --- End Added Debug Logging ---
                                            # --- Enhanced Debug Logging ---
                                            logging.debug(
                                                f"  Attempting to hit general note: {fn._note_name_cache} "
                                                f"(State: {fn.state}, Start: {fn.start_time_sec:.3f}s, "
                                                f"Current time: {self.current_sequence_time_sec:.3f}s)"
                                            )
                                            # --- End Enhanced Logging ---

                                            if fn.check_hit(note, play_time_ms):
                                                self.score += 10
                                                hit_registered_this_press = True

                                                # --- Enhanced Debug Logging ---
                                                logging.debug(
                                                    f"    >>> HIT REGISTERED via general check for {fn._note_name_cache}. "
                                                    f"New state: {fn.state}, Hit time: {fn.hit_time_ms}ms"
                                                )
                                                # --- End Enhanced Logging ---

                                if not hit_registered_this_press:
                                    logging.debug(
                                        f"  --- NOTE {note_played_name} ({note}) DID NOT MATCH any active/upcoming target ---"
                                    )

                        # --- Note Off Event ---
                        elif (128 <= status <= 143) or (
                            144 <= status <= 159 and velocity == 0
                        ):
                            # logging.debug(f"MIDI IN: Note Off - Ch={channel} Note={note} ({self.get_note_name(note)})") # Reduce log spam
                            if note in self.active_notes:
                                self.active_notes[note] = (
                                    False  # Mark key as released for drawing
                                )

                                # --- Enhanced Chord Detection ---
                                # Remove the note from active chord keys
                                if note in self.active_chord_keys:
                                    self.active_chord_keys.remove(note)

                                    # Log remaining keys if any
                                    if self.active_chord_keys:
                                        remaining_notes = [self.get_note_name(n) for n in self.active_chord_keys]
                                        logging.debug(f"Released {self.get_note_name(note)}. Remaining keys: {remaining_notes}")
                                    else:
                                        logging.debug(f"Released {self.get_note_name(note)}. No more keys pressed.")
                                # --- End Enhanced Chord Detection ---

            except pygame.midi.MidiException as e:
                logging.error(
                    f"MIDI Read Error: {e}. Disconnecting MIDI input.", exc_info=True
                )
                print(
                    f"\n--- MIDI Read Error: {e}. Disconnecting. ---", file=sys.stderr
                )
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
            """Draws the piano keys, highlighting active ones based on self.active_notes."""
            # --- Draw White Keys First ---
            for note in range(self.first_note, self.last_note + 1):
                if not self.is_black_key(note):
                    rect = self.get_key_rect(note)  # Get cached rect
                    if rect:
                        is_pressed = self.active_notes.get(note, False)
                        color = (
                            self.colors["piano_white_key_pressed"]
                            if is_pressed
                            else self.colors["piano_white_key"]
                        )
                        border_color = self.colors["key_border"]
                        text_color = self.colors["key_text"]
                        pygame.draw.rect(self.screen, color, rect)
                        pygame.draw.rect(self.screen, border_color, rect, 1)
                        note_name = self.get_note_name(note)
                        if note_name:
                            try:
                                name_surf = self.key_font.render(
                                    note_name, True, text_color
                                )
                                name_rect = name_surf.get_rect(
                                    centerx=rect.centerx, bottom=rect.bottom - 5
                                )
                                self.screen.blit(name_surf, name_rect)
                            except Exception as e:
                                logging.error(
                                    f"Error rendering white key name '{note_name}': {e}"
                                )

            # --- Draw Black Keys Second ---
            for note in range(self.first_note, self.last_note + 1):
                if self.is_black_key(note):
                    rect = self.get_key_rect(note)  # Get cached rect
                    if rect:
                        is_pressed = self.active_notes.get(note, False)
                        color = (
                            self.colors["piano_black_key_pressed"]
                            if is_pressed
                            else self.colors["piano_black_key"]
                        )
                        border_thickness = 2 if is_pressed else 0
                        border_color = self.colors["pressed_black_key_border"]
                        pygame.draw.rect(self.screen, color, rect)
                        if border_thickness > 0:
                            pygame.draw.rect(
                                self.screen, border_color, rect, border_thickness
                            )

        def _cleanup(self):
            """Properly shuts down Pygame and MIDI resources."""
            logging.info("Cleaning up resources...")
            if self.midi_input:
                try:
                    logging.debug("Closing MIDI input device...")
                    self.midi_input.close()
                    self.midi_input = None
                    self.midi_device_name = None
                    logging.info("MIDI input closed.")
                except Exception as e:
                    logging.exception("Error closing MIDI input.")
            try:
                self._extracted_from__cleanup_1297()
            except Exception as e:
                logging.exception("Error during Pygame cleanup.")
            print("\n--- Piano Trainer Exited ---")




        def _extracted_from__cleanup_1297(self):
            pygame.mixer.quit()  # Quit the mixer
            logging.debug("Pygame mixer subsystem quit.")
            pygame.midi.quit()
            logging.debug("Pygame MIDI subsystem quit.")
            pygame.font.quit()
            logging.debug("Pygame font subsystem quit.")
            pygame.quit()
            logging.info("Pygame quit successfully.")


    # Return the enhanced class
    return EnhancedPianoTrainerUI


    # Return the enhanced class
    return EnhancedPianoTrainerUI
    return EnhancedPianoTrainerUI


    # Return the enhanced class
    return EnhancedPianoTrainerUI
    # Return the enhanced class
    return EnhancedPianoTrainerUI
    return EnhancedPianoTrainerUI


    # Return the enhanced class
    return EnhancedPianoTrainerUI
# =========================================================
# Main Execution Block (Unchanged)
# =========================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Piano Trainer with Pygame")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="freestyle",
        choices=["freestyle", "learning", "analysis_view"],
        help="Run mode: freestyle, learning (falling notes), analysis_view (show MIDI report)",
    )
    parser.add_argument(
        "-f",
        "--midi",
        type=str,
        default=None,
        help="Path to MIDI file for analysis or 'learning - midi' mode",
    )
    parser.add_argument(
        "--learn",
        type=str,
        default="scale",
        choices=["scale", "chord_progression", "midi"],
        help="Type of content for learning mode (if mode=learning)",
    )
    parser.add_argument(
        "--root",
        type=int,
        default=60,  # Default to C4
        help="Root note (MIDI number, e.g., 60 for C4) for scales/chords",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="major",
        help="Scale type (e.g., major, natural_minor) or chord type ('progression' uses hardcoded progression)",
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        type=str,
        default="intermediate",
        choices=["beginner", "intermediate", "advanced"],
        help="Difficulty setting (currently only affects title display)",
    )
    parser.add_argument(
        "-l",
        "--log",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level for file output (e.g., DEBUG for more detail)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG logging (shortcut for --log DEBUG)",
    )

    args = parser.parse_args()

    # --- Configure Logging ---
    log_level_str = "DEBUG" if args.verbose else args.log
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)

    log_filename = "piano_trainer.log"
    log_format = "%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(message)s"
    logging.basicConfig(
        level=log_level, format=log_format, filename=log_filename, filemode="w"
    )  # Overwrite log each run

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)  # Console shows INFO and above
    console_formatter = logging.Formatter("%(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    logging.getLogger().addHandler(console_handler)

    # --- Argument Validation ---
    if args.mode == "learning" and args.learn == "midi" and not args.midi:
        logging.critical(
            "Argument Error: --learn midi requires a MIDI file specified with -f/--midi."
        )
        parser.error(
            "When using '--mode learning' with '--learn midi', you must provide a MIDI file using -f or --midi."
        )
        sys.exit(1)
    if args.mode == "analysis_view" and not args.midi:
        logging.critical(
            "Argument Error: analysis_view mode requires a MIDI file specified with -f/--midi."
        )
        parser.error(
            "Mode 'analysis_view' requires a MIDI file specified with -f or --midi."
        )
        sys.exit(1)

    # --- Application Startup ---
    trainer_app = None
    try:
        logging.info("=" * 10 + " Application Starting " + "=" * 10)
        logging.info(f"Command line arguments: {vars(args)}")

        EnhancedTrainer = enhance_piano_trainer_ui(PianoTrainer)
        trainer_app = EnhancedTrainer()
        trainer_app.run(
            mode=args.mode,
            midi_file=args.midi,
            difficulty=args.difficulty,
            learning_type=args.learn,
            root_note=args.root,
            scale_chord_type=args.type,
        )

    except RuntimeError as e:
        print(f"\nApplication failed to start or run: {e}", file=sys.stderr)
        logging.critical(f"Application runtime error: {e}", exc_info=True)
        if trainer_app:
            trainer_app._cleanup()
        else:
            pygame.quit()
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupt received, exiting gracefully...")
        logging.info("Keyboard interrupt received.")
        # Cleanup should happen in run()'s finally block
        sys.exit(0)
    except FileNotFoundError as e:
        print(f"\nError: File not found - {e}", file=sys.stderr)
        logging.critical(f"File not found error: {e}", exc_info=True)
        if trainer_app:
            trainer_app._cleanup()
        else:
            pygame.quit()
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        logging.critical("Unhandled exception in main execution block.", exc_info=True)
        if trainer_app:
            trainer_app._cleanup()
        else:
            pygame.quit()
        sys.exit(1)

    logging.info("=" * 10 + " Application Finished " + "=" * 10)
    sys.exit(0)
