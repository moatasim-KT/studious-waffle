import os
import shutil

# Set the directory path where the mp3 files are located
source_directory = "/Users/moatasimfarooque/MEGA/studious-waffle/midi-js-soundfonts/FatBoy/acoustic_grand_piano-mp3"
# Set the destination directory where you want to move the wav files
destination_directory = "/Users/moatasimfarooque/MEGA/studious-waffle/samples"

# Create destination directory if it doesn't exist
os.makedirs(destination_directory, exist_ok=True)

# Dictionary to convert flat notes to sharps or their correct enharmonic equivalents
flat_to_sharp = {
    "Bb": "A#",
    "Ab": "G#",
    "Db": "C#",
    "Eb": "D#",
    "Gb": "F#",
    "Cb": "B",  # Convert Cb to B
}


# Function to convert flats to their equivalent sharps
def convert_flats_to_sharps(note):
    for flat, sharp in flat_to_sharp.items():
        if flat in note:
            note = note.replace(flat, sharp)
    return note


# Iterate through the files in the source directory
for filename in os.listdir(source_directory):
    if filename.endswith(".mp3"):
        # Extract the note from the filename (e.g., C4 from C4.mp3)
        note = filename.split(".")[0]

        # Convert flat notes to sharp notes
        note = convert_flats_to_sharps(note)

        # Create the new name in the format piano_<note>.wav
        new_name = f"piano_{note}.wav"

        # Get the full paths for the old and new file names
        old_file = os.path.join(source_directory, filename)
        new_file = os.path.join(destination_directory, new_name)

        # Rename and move the file to the new directory
        shutil.copy(old_file, new_file)

        print(f"Moved: {filename} -> {new_name}")
