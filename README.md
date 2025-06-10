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
```
musegan-pytorch/
├── README.md
├── run_pretrained.ipynb       # Main Jupyter notebook
├── generator_final.pth        # Pre-trained model weights
├── generator_final2.pth       # Alternative pre-trained model
└── generated_music/           # Output directory (created automatically)
    ├── generated_music_sample_0.mid
    ├── generated_music_sample_0.npy
    └── ...
```    
