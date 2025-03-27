# Piano Trainer

## Overview

Piano Trainer is an interactive application designed to help musicians practice and improve their piano skills through various interactive modes.

## Features

- MIDI Input Support
- Multiple Training Modes
  - Play-along
  - Learning
  - Freestyle
- Configurable Keyboard Mapping
- Real-time Sound Generation
- Chord Sequence Tracking

## Prerequisites

- Python 3.8+
- Pygame
- Numpy
- PyYAML
- Mido (optional)

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/piano-trainer.git
cd piano-trainer
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

## Configuration

The application uses a YAML configuration file (`config.yaml`) for customization:

### Key Configuration Options

- **Window Settings**
  - Adjust window dimensions
  - Set window title

- **Keyboard Mapping**
  - Customize computer keyboard to MIDI note mappings
  - Example: `a` key mapped to C3 note

- **Audio Settings**
  - Configure sample rate
  - Adjust volume
  - Modify sound generation envelope

- **MIDI Configuration**
  - Set note range
  - Define total number of keys

- **Logging**
  - Set logging level
  - Customize log format

## Usage

### Running the Application

```bash
python piano_trainer.py
```

### Modes

1. **Play-along Mode**
   - Follow chord progressions
   - Real-time feedback on note accuracy

2. **Learning Mode**
   - Guided practice exercises
   - Skill development tools

3. **Freestyle Mode**
   - Open-ended piano exploration
   - No specific constraints

## Development

### Project Structure

- `piano_trainer.py`: Main application logic
- `config.yaml`: Configuration file
- `tests/`: Unit tests directory

### Running Tests

```bash
python -m unittest discover tests
```

## Advanced Configuration

Modify `config.yaml` to customize:
- Keyboard mappings
- Audio settings
- MIDI input configuration
- Logging preferences

## Troubleshooting

- Ensure MIDI devices are properly connected
- Check Python and dependency versions
- Verify configuration file syntax

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here]

## Future Roadmap

- Enhanced chord recognition
- More sophisticated learning algorithms
- Additional training modes
- Improved sound synthesis
- Performance analytics

## Contact

[Your contact information or project maintainer details]
# midi_experiments
