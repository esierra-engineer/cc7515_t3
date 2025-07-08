# N-Body Problem Simulation

A real-time 3D N-body gravitational simulation implemented with CUDA and OpenGL, featuring GPU acceleration and interactive controls.

## Requirements

- **Operating System**: Debian 12 x64
- **Hardware**: CUDA-compatible GPU
- **Software Dependencies**:
    - CUDA 12 compiler [**(Download)**](https://developer.nvidia.com/cuda-downloads)
    - CMake 3.25+ [**(Download)**](https://cmake.org/download/)
    - OpenGL [**(Download)**](https://www.khronos.org/opengl/wiki/Getting_Started#Downloading_OpenGL)
    - GLFW [(**Installation Guide**)](https://www.glfw.org/docs/3.3/compile.html#compile_compile)
    - GLM (OpenGL Mathematics) [**(Installation Guide)**](https://github.com/g-truc/glm?tab=readme-ov-file#build-and-install)

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
./CC7515_T3 /path/to/config_file/config.json
```
Default configuration file is ```resources/defaultConfig.json```.
## Features

### 🔧 CUDA-OpenGL Interoperability
Seamless integration between CUDA compute kernels and OpenGL rendering pipeline for optimal performance.

### 🎨 Graphics & Rendering
- **3D Simulation**: Full 3D particle physics simulation
- **Lighting System**: Simple lighting with specular reflection
- **Textured Particles**: Spherical particles with football texture (`resources/football.png`)
- **Shaders**: Custom vertex and fragment shaders located in `src/shaders/`

This project includes GLSL shaders used for a 3D N-body simulation. There are two shader programs:

1. **Default Shader** – used for rendering the particles (spheres) with lighting and texture.
2. **Light Shader** – used for rendering the light source (a cube or sphere).

---

#### 1. `default.vert` – Vertex Shader (Particles)

##### Inputs:
- `layout (location = 0) in vec3 aPos`: Vertex position.
- `layout (location = 1) in vec3 aNormal`: Vertex normal.
- `layout (location = 2) in vec2 aTex`: Texture coordinate.

##### Uniforms:
- `mat4 model`: Model transformation matrix.
- `mat4 camMatrix`: Combined view-projection matrix.

##### Outputs:
- `vec3 FragPos`: World-space position of the vertex.
- `vec3 Normal`: Transformed normal vector.
- `vec2 TexCoord`: Passed-through texture coordinate.

##### Function:
This shader transforms vertex positions to clip space and calculates world-space normals and positions for lighting in the fragment shader.

---

#### 2. `default.frag` – Fragment Shader (Particles)

##### Inputs:
- `vec3 FragPos`: From vertex shader, world-space position.
- `vec3 Normal`: From vertex shader, transformed normal.
- `vec2 TexCoord`: Texture coordinate.

##### Uniforms:
- `sampler2D tex0`: The texture applied to regular particle.
- `sampler2D tex1`: The texture applied to special particles.
- `bool isSpecial`: Flag to identify special particles.
- `vec3 lightPos`: Position of the light source.
- `vec3 viewPos`: Position of the camera.

##### Output:
- `vec4 FragColor`: Final color of the fragment.

##### Function:
Implements Phong lighting model:
- **Ambient** lighting.
- **Diffuse** reflection (based on Lambert's cosine law).
- **Specular** reflection (using Blinn-Phong model).

The final fragment color is the combination of these lighting components modulated by the texture color.

---

#### 3. `light.frag` – Fragment Shader (Light Source)

##### Output:
- `vec4 FragColor`: Light color.

##### Function:
This shader outputs color to visualize the light source. No lighting calculations are done here.

---

#### 4. `light.vert` – Vertex Shader (Light Source)

##### Inputs:
- `layout (location = 0) in vec3 aPos`: Vertex position.

##### Uniforms:
- `mat4 model`: Model transformation matrix for the light.
- `mat4 camMatrix`: Combined view-projection matrix.

##### Function:
Transforms the light source geometry into clip space for rendering.

---

#### Notes:
- Lighting calculations are done per-fragment (pixel), giving smooth shading.
---

### ⚙️ Physics Engine
- **Particle Model**: Based on `Body` class with position, velocity, and mass attributes
- **Special Particles**: Support for particles with different masses and properties
- **GPU/CPU Processing**: Toggle between GPU and CPU computation modes

### 🎮 Interactive Controls

#### Camera Controls (First Person)
| Key                 | Action |
|---------------------|--------|
| `W`                 | Move forward |
| `A`                 | Move left |
| `S`                 | Move backward |
| `D`                 | Move right |
| `SPACE`             | Move up |
| `LEFT CTRL`         | Move down |
| `LEFT SHIFT`        | Speed boost (4x) |
| `Left Mouse Button` | Look around (hold and move mouse) |
| `V`                 | Reset camera position |

#### Simulation Controls
| Key          | Action                          |
|--------------|---------------------------------|
| `RIGHT CTRL` | Pause/unpause movement          |
| `C`          | Open/close configuration window |
| `R`          | Reset time step                 |
| `I`          | Increase time step              |
| `K`          | Decrease time step              |
| `ESC`        | Exit application                |

### 🖥️ Real-time Interface

Press `C` to access the configuration panel with the following controls:

- **Processing Mode**: Toggle between GPU and CPU computation
- **Simulation Speed**: Adjust time step via slider
- **Number of Bodies**: Control total particle count
- **Special Bodies**: Set number of special particles
- **Normal Mass**: Adjust mass of regular particles
- **Special Mass**: Set mass for special particles
- **Light Color**: Set color for illumination.
- **Reset**: Restart the simulation with current parameters

### Custom Simulation Parameters
---

#### 📐 Display Settings

- **`width`**: `1280`  
  Width of the simulation window in pixels.

- **`height`**: `720`  
  Height of the simulation window in pixels.

---

#### 🎥 Camera

- **`camera_init_pos`**:
  - `x`: `0.0`
  - `y`: `0.0`
  - `z`: `100.0`  
    Initial position of the camera in world space.

---

#### 🌌 Particle Properties

- **`particleSize`**: `1.0`  
  Size of common particles.

- **`specialParticleSize`**: `3.0`  
  Size of special particles.

- **`mass`**: `2.0e0`  
  Mass of each common particle.

- **`specialMass`**: `1.0e2`  
  Mass of each special particle.

---

#### ⏱️ Simulation Settings

- **`dt`**: `50.0`  
  Time step (delta time) for each simulation iteration.

- **`numBodies`**: `2048`  
  Number of common particles in the simulation.

- **`specialBodies`**: `2048`  
  Number of special particles.

- **`stop`**: `false`  
  Boolean flag to stop or resume the simulation.

- **`useGPU`**: `true`  
  Enables GPU calculations using CUDA.

---

#### 🧠 Compute Settings

- **`kernel_filename`**: `"kernel.ptx"`  
  Filename of the compiled CUDA kernel used for force computation.

- **`local_size`**: `128`  
  Workgroup size for CUDA kernel execution.

---

#### 💡 Lighting

- **`sourceLightColor`**: `[1.0, 1.0, 1.0, 1.0]`  
  RGBA color of the main light source.

---

#### 🖼️ Textures

- **`particleTextureFile`**: `"resources/football.png"`  
  Texture file applied to common particles.

- **`specialParticleTextureFile`**: `"resources/grass.png"`  
  Texture file applied to special particles.

---

#### 📂 File I/O

- **`read_particles_from_file`**: `false`  
  Indicates whether to load particle data from a file or generate randomly.

## Technical Implementation

### Core Components
- **Body Class**: Particle representation with GLM vec3 position/velocity vectors
- **Light Source**: Single light positioned at (0.5, 0.5, 0.5)
- **Texture System**: Image texture applied to all particle spheres

### Current Status
✅ **Completed Features:**
- 3D particle simulation
- CUDA-OpenGL interoperability
- First-person camera system
- Real-time parameter adjustment
- Textured spherical particles
- Basic lighting system

## File Structure
```
├── include/              # Header code files
├── Libraries/              # External libraries
│   ├── include/          
│   └── lib/              
├── src/
│   ├── shaders/          # Vertex and fragment shaders
│   └── ...              # Source code files
├── resources/          # Config files and Textures
│  
├── build/               # Build directory (created during compilation)
│
└── CMakeLists.txt      # cmake configuration file
└── README.md           # This file
```