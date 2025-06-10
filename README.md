# DL_music_MouseGAN

MuseGAN PyTorch - Multi-Track Music Generation
A PyTorch implementation of MuseGAN for generating multi-track polyphonic music using Generative Adversarial Networks (GANs). This project can generate music across 5 different tracks: Drums, Piano, Guitar, Bass, and Strings.
Features

Multi-track Music Generation: Generates music for 5 instrument tracks simultaneously
Flexible Parameters: Adjustable temperature and threshold for creative control
Rich Visualizations: Piano roll visualizations for individual tracks and comparisons
Multiple Export Formats: Save as MIDI files or NumPy arrays
Statistical Analysis: Detailed statistics for each generated track
Interactive Generation: Easy-to-use functions for experimentation

Project Structure
musegan-pytorch/
├── README.md
├── run_pretrained.ipynb       # Main Jupyter notebook
├── generator_final.pth        # Pre-trained model weights
├── generator_final2.pth       # Alternative pre-trained model
└── generated_music/           # Output directory (created automatically)
    ├── generated_music_sample_0.mid
    ├── generated_music_sample_0.npy
    └── ...
Quick Start
Prerequisites
bashpip install torch torchvision numpy matplotlib pretty_midi pathlib
Basic Usage

Load the model and generate music:

python# Initialize generator
generator = MuseGANGeneratorTorch("generator_final2.pth", verbose=True)

# Load model
if generator.load_model():
    # Generate 4 music samples
    logits = generator.generate_music(n_samples=4, temperature=1.0)
    processed_music = generator.postprocess_music(logits, threshold=0.04)
    
    # Save as MIDI
    generator.save_as_midi(processed_music, "my_music.mid")

Visualize the results:

python# Plot piano roll for Piano track
generator.plot_pianoroll(processed_music, sample_idx=0, track_idx=1)

# Compare all tracks side by side
generator.plot_track_comparison(processed_music, sample_idx=0)

# Analyze statistics
stats = generator.analyze_music_statistics(processed_music)
Model Architecture
The generator uses a 3D CNN architecture with the following specifications:

Input: Latent vector of dimension 128
Output: Multi-track music tensor (5 tracks × 64 time steps × 72 pitches)
Tracks:

Track 0: Drums
Track 1: Piano
Track 2: Guitar
Track 3: Bass
Track 4: Strings



Hyperparameters
pythonlatent_dim = 128           # Latent vector dimension
n_tracks = 5               # Number of instrument tracks
n_measures = 4             # Number of measures
measure_resolution = 16    # Time steps per measure
n_pitches = 72             # Number of pitches
lowest_pitch = 24          # Lowest MIDI pitch (C1)
Customization Options
Temperature Control
Controls the randomness of generation:

temperature=0.5: More conservative, structured music
temperature=1.0: Balanced (default)
temperature=2.0: More random, experimental music

Threshold Control
Controls note activation sensitivity:

threshold=0.3: More notes active (denser music)
threshold=0.5: Balanced (default)
threshold=0.7: Fewer notes active (sparser music)

Interactive Generation
python# Generate with custom parameters
music = generate_and_visualize(
    n_samples=2,
    temperature=1.2,
    threshold=0.4,
    track_to_plot=1  # Piano track
)



Custom Visualization
python# Extract piano roll for specific track
piano_roll = generator.music_to_pianoroll(processed_music, track_idx=1)

# Custom plotting
plt.figure(figsize=(12, 6))
plt.imshow(piano_roll.T, aspect='auto', origin='lower', cmap='viridis')
plt.title('Custom Piano Roll Visualization')
plt.show()

File Formats
NumPy Arrays

Shape: (n_samples, n_tracks, n_steps, n_pitches)
Values: Binary (0 or 1) indicating note on/off
Usage: For further processing or analysis

MIDI Files

Standard MIDI format: Compatible with most DAWs and music software
Multi-track: Each instrument on separate track
Tempo: Configurable (default 120 BPM)



    
References

Original MuseGAN paper: MuseGAN: Multi-track Sequential Generative Adversarial Networks for Symbolic Music Generation and Accompaniment
PyTorch documentation: pytorch.org
Pretty MIDI: pretty_midi documentation

Contributing
Feel free to submit issues, feature requests, or pull requests to improve this implementation.

License
This project is open source. Please check the original MuseGAN repository for licensing details.
