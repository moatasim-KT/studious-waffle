import pygame
import pygame.midi
import mido
import sys
import time
import numpy as np
from collections import defaultdict, Counter
import logging
import yaml
from typing import Dict, List, Set, Optional, Any, Tuple
import matplotlib.pyplot as plt
from play_along import PianoTrainer
import io
import random


class AdvancedMIDIParser:
    """
    Enhanced MIDI file parsing with advanced analysis capabilities
    """

    def __init__(self):
        """
        Initialize advanced MIDI parser
        """
        self.midi_analysis = {
            "total_notes": 0,
            "unique_notes": set(),
            "note_distribution": defaultdict(int),
            "note_duration_stats": {
                "min_duration": float("inf"),
                "max_duration": 0,
                "avg_duration": 0,
            },
            "tempo_changes": [],
            "key_signature_changes": [],
        }

    def parse_midi_file(self, midi_file_path: str) -> Dict[str, Any]:
        """
        Parse MIDI file with comprehensive analysis

        :param midi_file_path: Path to MIDI file
        :return: Detailed MIDI file analysis
        """
        try:
            midi_file = mido.MidiFile(midi_file_path)

            # Reset analysis
            self.midi_analysis = {
                "total_notes": 0,
                "unique_notes": set(),
                "note_distribution": defaultdict(int),
                "note_duration_stats": {
                    "min_duration": float("inf"),
                    "max_duration": 0,
                    "avg_duration": 0,
                },
                "tempo_changes": [],
                "key_signature_changes": [],
            }

            total_note_duration = 0
            note_count = 0

            for track in midi_file.tracks:
                current_time = 0
                active_notes = {}

                for msg in track:
                    current_time += msg.time

                    # Tempo changes
                    if msg.type == "set_tempo":
                        self.midi_analysis["tempo_changes"].append(
                            {"time": current_time, "tempo": msg.tempo}
                        )

                    # Key signature changes
                    if msg.type == "key_signature":
                        self.midi_analysis["key_signature_changes"].append(
                            {"time": current_time, "key": f"{msg.key}/{msg.scale}"}
                        )

                    # Note analysis
                    if msg.type == "note_on" and msg.velocity > 0:
                        # Track note start
                        active_notes[msg.note] = current_time
                        self.midi_analysis["unique_notes"].add(msg.note)
                        self.midi_analysis["note_distribution"][msg.note] += 1
                        self.midi_analysis["total_notes"] += 1

                    elif (
                        msg.type == "note_off"
                        or (msg.type == "note_on" and msg.velocity == 0)
                    ) and msg.note in active_notes:
                        # Calculate note duration
                        note_start = active_notes.pop(msg.note)
                        note_duration = current_time - note_start

                        # Update duration statistics
                        self.midi_analysis["note_duration_stats"]["min_duration"] = min(
                            self.midi_analysis["note_duration_stats"]["min_duration"],
                            note_duration,
                        )
                        self.midi_analysis["note_duration_stats"]["max_duration"] = max(
                            self.midi_analysis["note_duration_stats"]["max_duration"],
                            note_duration,
                        )
                        total_note_duration += note_duration
                        note_count += 1

            # Calculate average note duration
            if note_count > 0:
                self.midi_analysis["note_duration_stats"]["avg_duration"] = (
                    total_note_duration / note_count
                )

            return self.midi_analysis

        except Exception as e:
            logging.error(f"MIDI file parsing error: {e}")
            return {}

    def generate_midi_analysis_report(self) -> str:
        """
        Generate a detailed text report of MIDI file analysis

        :return: Formatted analysis report
        """
        analysis = self.midi_analysis
        report = "### MIDI File Analysis Report ###\n\n"

        report += f"Total Notes: {analysis['total_notes']}\n"
        report += f"Unique Notes: {len(analysis['unique_notes'])}\n\n"

        # Top 5 most frequent notes
        sorted_notes = sorted(
            analysis["note_distribution"].items(), key=lambda x: x[1], reverse=True
        )[:5]
        report += "Most Frequent Notes:\n"
        for note, count in sorted_notes:
            report += f"  Note {note}: {count} times\n"

        report += "\nNote Duration Statistics:\n"
        duration_stats = analysis["note_duration_stats"]
        report += f"  Min Duration: {duration_stats['min_duration']:.2f}\n"
        report += f"  Max Duration: {duration_stats['max_duration']:.2f}\n"
        report += f"  Avg Duration: {duration_stats['avg_duration']:.2f}\n"

        report += "\nTempo Changes:\n"
        for change in analysis["tempo_changes"]:
            report += (
                f"  At time {change['time']}: {change['tempo']} microseconds per beat\n"
            )

        report += "\nKey Signature Changes:\n"
        for change in analysis["key_signature_changes"]:
            report += f"  At time {change['time']}: {change['key']}\n"

        return report


class AdvancedLearningChallenges:
    """
    More complex and diverse learning challenges
    """

    def __init__(self, config: Any):
        """
        Initialize advanced learning challenges

        :param config: Configuration manager
        """
        self.config = config

        # Enhanced challenge types
        self.challenge_types = {
            "scale_variations": [
                {
                    "name": "Ascending and Descending Scales",
                    "description": "Play a scale both up and down",
                    "generator": self._generate_bidirectional_scale,
                },
                {
                    "name": "Scale with Varied Rhythm",
                    "description": "Play scale with different note durations",
                    "generator": self._generate_rhythmic_scale,
                },
            ],
            "chord_progressions": [
                {
                    "name": "Common Chord Progressions",
                    "description": "Practice typical chord sequences",
                    "generator": self._generate_chord_progression,
                },
                {
                    "name": "Jazz Chord Voicings",
                    "description": "Explore extended and altered chords",
                    "generator": self._generate_jazz_chords,
                },
            ],
            "improvisation_challenges": [
                {
                    "name": "Modal Improvisation",
                    "description": "Improvise within a specific musical mode",
                    "generator": self._generate_modal_improv,
                },
                {
                    "name": "Call and Response",
                    "description": "Respond to a musical phrase",
                    "generator": self._generate_call_response,
                },
            ],
        }

    def generate_learning_challenge(
        self, difficulty: str = "intermediate"
    ) -> Dict[str, Any]:
        """
        Generate a sophisticated learning challenge

        :param difficulty: Challenge difficulty level
        :return: Detailed learning challenge
        """
        # Select challenge type based on difficulty
        challenge_pools = {
            "beginner": ["scale_variations"],
            "intermediate": ["scale_variations", "chord_progressions"],
            "advanced": [
                "scale_variations",
                "chord_progressions",
                "improvisation_challenges",
            ],
        }

        # Select a random challenge type from the appropriate pool
        challenge_pool = challenge_pools.get(
            difficulty, challenge_pools["intermediate"]
        )
        selected_type = random.choice(challenge_pool)

        # Select a specific challenge within the type
        challenge = random.choice(self.challenge_types[selected_type])

        # Generate specific challenge details
        return challenge["generator"]()

    def _generate_bidirectional_scale(self) -> Dict[str, Any]:
        """
        Generate a bidirectional scale challenge

        :return: Bidirectional scale challenge details
        """
        # Common scales
        scales = {
            "C Major": [60, 62, 64, 65, 67, 69, 71, 72],  # C4 to C5
            "G Major": [67, 69, 71, 72, 74, 76, 78, 79],  # G4 to G5
            "F Major": [65, 67, 69, 70, 72, 74, 76, 77],  # F4 to F5
        }

        selected_scale = random.choice(list(scales.keys()))
        notes = scales[selected_scale]

        # Create ascending and descending notes
        full_scale = notes + list(reversed(notes[:-1]))

        return {
            "type": "scale",
            "name": f"Bidirectional {selected_scale} Scale",
            "notes": full_scale,
            "instructions": f"Play the {selected_scale} scale up and down smoothly",
        }

    def _generate_rhythmic_scale(self) -> Dict[str, Any]:
        """
        Generate a scale with varied rhythmic pattern

        :return: Rhythmic scale challenge details
        """
        # Use C Major scale
        notes = [60, 62, 64, 65, 67, 69, 71, 72]

        # Create varied rhythm (represented by note durations)
        rhythmic_patterns = [
            {"name": "Staccato", "pattern": [0.25] * len(notes)},
            {"name": "Legato", "pattern": [1.0] * len(notes)},
            {
                "name": "Syncopated",
                "pattern": [0.5, 0.25, 0.75, 0.5, 0.25, 0.75, 0.5, 0.5],
            },
        ]

        selected_pattern = random.choice(rhythmic_patterns)

        return {
            "type": "rhythmic_scale",
            "name": f'{selected_pattern["name"]} C Major Scale',
            "notes": notes,
            "rhythm": selected_pattern["pattern"],
            "instructions": f"Play the C Major scale with {selected_pattern['name'].lower()} articulation",
        }

    def _generate_chord_progression(self) -> Dict[str, Any]:
        """
        Generate a chord progression challenge

        :return: Chord progression challenge details
        """
        # Common chord progressions
        progressions = [
            {
                "name": "I-V-vi-IV Progression",
                "chords": ["C", "G", "Am", "F"],
                "notes": [
                    [60, 64, 67],  # C Major
                    [67, 71, 74],  # G Major
                    [69, 72, 76],  # A minor
                    [65, 69, 72],  # F Major
                ],
            },
            {
                "name": "ii-V-I Jazz Progression",
                "chords": ["Dm", "G7", "CMaj7"],
                "notes": [
                    [62, 65, 69],  # D minor
                    [67, 71, 74, 77],  # G dominant 7th
                    [60, 64, 67, 71],  # C Major 7th
                ],
            },
        ]

        selected_progression = random.choice(progressions)

        return {
            "type": "chord_progression",
            "name": selected_progression["name"],
            "chords": selected_progression["chords"],
            "notes": selected_progression["notes"],
            "instructions": f"Practice the {selected_progression['name']} chord progression",
        }

    def _generate_jazz_chords(self) -> Dict[str, Any]:
        """
        Generate a jazz chord voicing challenge

        :return: Jazz chord voicing challenge details
        """
        # Extended and altered jazz chords
        jazz_chords = [
            {"name": "Major 9th Chord", "notes": [60, 64, 67, 71, 74]},  # C Maj9
            {"name": "Minor 11th Chord", "notes": [62, 65, 69, 72, 77]},  # D min11
            {"name": "Dominant 13th Chord", "notes": [67, 71, 74, 77, 81]},  # G13
        ]

        selected_chord = random.choice(jazz_chords)

        return {
            "type": "jazz_chord",
            "name": selected_chord["name"],
            "notes": selected_chord["notes"],
            "instructions": f"Practice voicing the {selected_chord['name']} chord",
        }

    def _generate_modal_improv(self) -> Dict[str, Any]:
        """
        Generate a modal improvisation challenge

        :return: Modal improvisation challenge details
        """
        # Musical modes with their characteristic notes
        modes = [
            {
                "name": "Dorian Mode",
                "base_note": 62,  # D
                "mode_notes": [62, 64, 65, 67, 69, 71, 72],
            },
            {
                "name": "Mixolydian Mode",
                "base_note": 67,  # G
                "mode_notes": [67, 69, 71, 72, 74, 76, 77],
            },
        ]

        selected_mode = random.choice(modes)

        return {
            "type": "modal_improvisation",
            "name": f'{selected_mode["name"]} Improvisation',
            "allowed_notes": selected_mode["mode_notes"],
            "base_note": selected_mode["base_note"],
            "instructions": f"Improvise using only notes from the {selected_mode['name']}",
        }

    def _generate_call_response(self) -> Dict[str, Any]:
        """
        Generate a call and response challenge

        :return: Call and response challenge details
        """
        # Predefined musical phrases
        call_phrases = [
            {"call": [60, 62, 64, 65], "expected_response": [67, 69, 71, 72]},
            {"call": [67, 69, 71, 72], "expected_response": [74, 76, 77, 79]},
        ]

        selected_phrase = random.choice(call_phrases)

        return {
            "type": "call_response",
            "name": "Musical Call and Response",
            "call_notes": selected_phrase["call"],
            "response_notes": selected_phrase["expected_response"],
            "instructions": "Listen to the call phrase, then respond with the matching phrase",
        }


# Extend the existing UI to support more interactive elements
def enhance_piano_trainer_ui(PianoTrainer):
    """
    Enhance PianoTrainer with more advanced UI features
    """

    class EnhancedPianoTrainerUI(PianoTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Advanced MIDI parsing
            self.midi_parser = AdvancedMIDIParser()

            # Enhanced learning challenges
            self.advanced_challenges = AdvancedLearningChallenges(self.config_manager)

            # Add new UI components
            self._setup_additional_ui_elements()

            # Performance tracking
            self.performance_history = []

        def _setup_additional_ui_elements(self):
            """
            Set up additional UI components
            """
            # Additional fonts
            self.small_font = pygame.font.Font(None, 24)

            # UI colors
            self.colors = {
                "background": (30, 30, 50),
                "text_primary": (255, 255, 255),
                "text_secondary": (200, 200, 255),
                "highlight": (100, 200, 100),
            }

        def run(
            self, mode: Optional[str] = None, midi_file: Optional[str] = None
        ) -> None:
            """
            Enhanced run method with additional features
            """
            # Parse MIDI file if provided
            if midi_file:
                self.midi_analysis = self.midi_parser.parse_midi_file(midi_file)
                # Generate analysis report
                analysis_report = self.midi_parser.generate_midi_analysis_report()
                print(analysis_report)  # Or log to file

            # Generate learning challenge
            if mode == "learning":
                self.current_learning_challenge = (
                    self.advanced_challenges.generate_learning_challenge()
                )

            # Call parent run method
            super().run(mode, midi_file)

        def _render_ui(self) -> None:
            """
            Enhanced UI rendering with additional information
            """
            # Clear the screen with a gradient background
            screen_rect = self.screen.get_rect()
            for y in range(screen_rect.height):
                # Create a gradient background
                r = int(30 + (y / screen_rect.height) * 50)
                g = int(30 + (y / screen_rect.height) * 50)
                b = int(50 + (y / screen_rect.height) * 100)
                pygame.draw.line(self.screen, (r, g, b), (0, y), (screen_rect.width, y))

            # Call parent rendering
            super()._render_ui()

            # Render additional challenge information
            if self.mode == "learning" and self.current_learning_challenge:
                challenge = self.current_learning_challenge

                # Challenge name
                challenge_text = self.font.render(
                    f"Challenge: {challenge.get('name', 'None')}",
                    True,
                    self.colors["text_primary"],
                )

                # Challenge instructions
                instructions_text = self.small_font.render(
                    challenge.get("instructions", "No specific instructions"),
                    True,
                    self.colors["text_secondary"],
                )

                # Render challenge information
                self.screen.blit(challenge_text, (10, 90))
                self.screen.blit(instructions_text, (10, 130))

            # Render MIDI file analysis (if available)
            if hasattr(self, "midi_analysis"):
                analysis_info = [
                    f"Total Notes: {self.midi_analysis.get('total_notes', 0)}",
                    f"Unique Notes: {len(self.midi_analysis.get('unique_notes', set()))}",
                ]

                for i, info in enumerate(analysis_info):
                    info_text = self.small_font.render(
                        info, True, self.colors["highlight"]
                    )
                    self.screen.blit(
                        info_text, (10, self.screen.get_height() - 50 - i * 30)
                    )

        def track_performance(self, accuracy: float, challenge: Dict[str, Any]):
            """
            Track performance across different challenges

            :param accuracy: Performance accuracy
            :param challenge: Challenge details
            """
            performance_entry = {
                "timestamp": time.time(),
                "challenge_type": challenge.get("type", "unknown"),
                "challenge_name": challenge.get("name", "Unnamed"),
                "accuracy": accuracy,
            }

            self.performance_history.append(performance_entry)

            # Limit performance history
            if len(self.performance_history) > 100:
                self.performance_history.pop(0)

        def generate_performance_report(self) -> str:
            """
            Generate a comprehensive performance report

            :return: Formatted performance report
            """
            if not self.performance_history:
                return "No performance data available."

            # Analyze performance history
            challenge_performance = defaultdict(list)
            for entry in self.performance_history:
                challenge_performance[entry["challenge_type"]].append(entry["accuracy"])

            report = "### Performance Report ###\n\n"

            for challenge_type, accuracies in challenge_performance.items():
                report += f"{challenge_type.capitalize()} Performance:\n"
                report += (
                    f"  Average Accuracy: {sum(accuracies) / len(accuracies):.2f}%\n"
                )
                report += f"  Best Accuracy: {max(accuracies):.2f}%\n"
                report += f"  Worst Accuracy: {min(accuracies):.2f}%\n\n"

            return report

    return EnhancedPianoTrainerUI


# Example usage and main function
def main():
    """
    Enhanced application entry point with advanced features
    """
    # Apply enhancements to PianoTrainer
    EnhancedTrainer = enhance_piano_trainer_ui(PianoTrainer)

    # Create enhanced trainer
    trainer = EnhancedTrainer()

    # Run in learning mode with optional MIDI file
    trainer.run(mode="learning", midi_file="optional_midi_file.mid")


if __name__ == "__main__":
    main()
