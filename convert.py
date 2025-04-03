import mido
import json
import time  # For potential debugging/timing if needed

# --- 1. Your JSON Data ---
with open("/Users/moatasimfarooque/MEGA/studious-waffle/to_midi.json", "r") as f:
    json_data_string = f.read()

try:
    note_events = json.loads(json_data_string)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    exit()
# --- End of JSON Data ---

try:
    note_events_json = json.loads(json_data_string)
except json.JSONDecodeError as e:
    print(f"Error decoding JSON: {e}")
    exit()

# --- 2. MIDI Parameters ---
ticks_per_beat = 480  # Standard resolution
tempo = 500000  # Default MIDI tempo (120 BPM in microseconds per beat)

# --- 3. Create MIDI File and Track ---
mid = mido.MidiFile(ticks_per_beat=ticks_per_beat, type=1)  # Type 1 for multiple tracks
track = mido.MidiTrack()
mid.tracks.append(track)

# --- 4. Add Initial Meta Messages (Optional but Recommended) ---
track.append(mido.MetaMessage("track_name", name="Converted Track", time=0))
# Set tempo (microseconds per beat)
track.append(mido.MetaMessage("set_tempo", tempo=tempo, time=0))
# Set instrument (Program Change) - 0 = Acoustic Grand Piano
track.append(mido.Message("program_change", program=0, time=0, channel=0))

# --- 5. Process JSON Events into MIDI Messages ---
# MIDI works with Note On and Note Off events. Create a list of these events
# with their absolute time in seconds.
midi_events = []
for note in note_events_json:
    start_sec = float(note["start"])
    duration_sec = float(note["duration"])
    pitch = int(note["pitch"])
    velocity = int(note["velocity"])
    end_sec = start_sec + duration_sec

    midi_events.extend(
        (
            {
                "time_sec": start_sec,
                "type": "note_on",
                "pitch": pitch,
                "velocity": velocity,
            },
            {
                "time_sec": end_sec,
                "type": "note_off",
                "pitch": pitch,
                "velocity": 64,
            },
        )
    )
# Sort events chronologically by absolute time in seconds
midi_events.sort(key=lambda x: x["time_sec"])

# --- 6. Convert to Mido Messages with Delta Times ---
last_time_ticks = 0
for event in midi_events:
    absolute_time_sec = event["time_sec"]
    # Convert absolute seconds to absolute ticks
    absolute_time_ticks = int(
        mido.second2tick(absolute_time_sec, ticks_per_beat, tempo)
    )

    # Calculate delta time (time since the last event) in ticks
    delta_time = absolute_time_ticks - last_time_ticks

    # Create the mido message
    if event["type"] == "note_on":
        msg = mido.Message(
            "note_on",
            note=event["pitch"],
            velocity=event["velocity"],
            time=delta_time,  # Delta time in ticks
            channel=0,
        )
    elif event["type"] == "note_off":
        msg = mido.Message(
            "note_off",
            note=event["pitch"],
            velocity=event["velocity"],  # Often 64 for note_off
            time=delta_time,  # Delta time in ticks
            channel=0,
        )
    else:
        continue  # Skip if not note_on or note_off

    track.append(msg)
    last_time_ticks = absolute_time_ticks  # Update last event time

# --- 7. Add End of Track Meta Message ---
# Find the time of the very last event to place the end_of_track message
final_event_time_ticks = last_time_ticks if midi_events else 0
track.append(
    mido.MetaMessage("end_of_track", time=0)
)  # Usually time=0, placed after last event

# --- 8. Save the MIDI file ---
output_mido_filename = "daredevil.mid"
try:
    mid.save(output_mido_filename)
    print(f"Successfully created MIDI file with mido: {output_mido_filename}")
except Exception as e:
    print(f"Error writing MIDI file with mido: {e}")
