[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/Q2iaFOq8)

# Setup

1. Clone the repo
2. `cd` into the **root** directory
3. Setup and activate a virtual env **(Python 3.12)**
4. `pip install -r requirements.txt`

# Requirements

In addition to the python dependencies, you will need to install [FluidSynth](https://www.fluidsynth.org).  
If you have [chocolatey](https://chocolatey.org/) installed, you can run the following command:

```bash
choco install fluidsynth
```

The application will directly pickup `FluidSynth` after installation.    
For more installation options visit the [FluidSynth wiki](https://github.com/FluidSynth/fluidsynth/wiki/Download).

# Running the Application

```bash
cd implementation
python main.py -c 0 -d
```

## Command Line Parameters

| Parameter | Short Option | Default | Description |
|-----------|--------------|---------|-------------|
| `--video-id` | `-c` | `1` | Video ID |
| `--debug` | `-d` | `False` | Show debug visualization |
| `--press-threshold` | `-p` | `320` | Minimum distance between base and tip for a finger press detection (pixels) |
| `--velocity-threshold` | `-v` | `20` | Minimum distance velocity for finger press detection (pixels/frame) |

# Documentation

The project was documented according to the [assignment requirements](./assignment08.pdf).  
The documentation can be found [here](./documentation.md).