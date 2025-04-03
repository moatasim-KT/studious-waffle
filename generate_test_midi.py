import mido
from mido import Message, MidiFile, MidiTrack
import time

def create_test_midi():
    # Create a new MIDI file with a single track
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)
    
    # Add some test notes (a simple C major scale)
    notes = [60, 62, 64, 65, 67, 69, 71, 72]  # C4 to C5
    delay = 480  # Delay between notes (in MIDI ticks)
    
    # Add some right hand notes (higher octave)
    for note in notes:
        track.append(Message('note_on', note=note, velocity=64, time=delay))
        track.append(Message('note_off', note=note, velocity=64, time=delay))
    
    # Add some left hand notes (lower octave)
    for note in notes:
        track.append(Message('note_on', note=note-12, velocity=64, time=delay))
        track.append(Message('note_off', note=note-12, velocity=64, time=delay))
    
    # Save the MIDI file
    mid.save('test_song.mid')
    print("Created test_song.mid")

if __name__ == '__main__':
    create_test_midi()


