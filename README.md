# N-Body Problem Simulation

A real-time 3D N-body gravitational simulation implemented with CUDA and OpenGL, featuring GPU acceleration and interactive controls.

## Requirements

- **Operating System**: Debian 12 x64
- **Hardware**: CUDA-compatible GPU
- **Software Dependencies**:
    - CUDA 12 compiler
    - CMake 3.25+
    - OpenGL
    - GLFW
    - GLM (OpenGL Mathematics)

## Installation & Usage

### Building the Project

Follow the standard CMake build procedure:

```bash
# Create build directory
mkdir build && cd build

# Configure the project
cmake -S .. -B .

# Compile and create binary
make

# Run the simulation
./CC7515_T3
```

## Features

### 🔧 CUDA-OpenGL Interoperability
Seamless integration between CUDA compute kernels and OpenGL rendering pipeline for optimal performance.

### 🎨 Graphics & Rendering
- **3D Simulation**: Full 3D particle physics simulation
- **Lighting System**: Simple lighting with specular reflection
- **Textured Particles**: Spherical particles with football texture (`resources/football.png`)
- **Shaders**: Custom vertex and fragment shaders located in `src/shaders/`

### ⚙️ Physics Engine
- **Particle Model**: Based on `Body` class with position, velocity, and mass attributes
- **Special Particles**: Support for particles with different masses and properties
- **GPU/CPU Processing**: Toggle between GPU and CPU computation modes

### 🎮 Interactive Controls

#### Camera Controls (First Person)
| Key | Action |
|-----|--------|
| `W` | Move forward |
| `A` | Move left |
| `S` | Move backward |
| `D` | Move right |
| `SPACE` | Move up |
| `LEFT CTRL` | Move down |
| `LEFT SHIFT` | Speed boost (4x) |
| `Left Mouse Button` | Look around (hold and move mouse) |
| `T` | Reset camera position |

#### Simulation Controls
| Key | Action |
|-----|--------|
| `C` | Open/close configuration window |
| `R` | Reset time step |
| `I` | Increase time step |
| `K` | Decrease time step |
| `ESC` | Exit application |

### 🖥️ Real-time Interface

Press `C` to access the configuration panel with the following controls:

- **Simulation Speed**: Adjust time step via slider
- **Number of Bodies**: Control total particle count
- **Special Bodies**: Set number of special particles
- **Normal Mass**: Adjust mass of regular particles
- **Special Mass**: Set mass for special particles
- **Processing Mode**: Toggle between GPU and CPU computation
- **Reset**: Restart the simulation with current parameters

## Technical Implementation

### Core Components
- **Body Class**: Particle representation with GLM vec3 position/velocity vectors
- **Light Source**: Single light positioned at (0.5, 0.5, 0.5)
- **Texture System**: Football texture applied to all particle spheres

### Current Status
✅ **Completed Features:**
- 3D particle simulation
- CUDA-OpenGL interoperability
- First-person camera system
- Real-time parameter adjustment
- Textured spherical particles
- Basic lighting system

⏳ **Pending Features:**
- Variable particle colors
- Spherical light source (currently cubic)

## File Structure
```
├── src/
│   ├── shaders/          # Vertex and fragment shaders
│   └── ...              # Source code files
├── resources/
│   └── football.png     # Particle texture
├── build/               # Build directory (created during compilation)
└── README.md           # This file
```