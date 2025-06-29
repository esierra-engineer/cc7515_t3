# Tarea 3 - Problema de los N cuerpos
## Interoperabilidad
Implemented interoperability between OpenGL and CUDA.
## Vertex shader and fragment shader
Vertex and fragment shader for both bodies and light source in ```src/shaders``` folder.
## La simulación debe ser en 3D.

Particle model is based on class ```Body```, which attributes are position, velocity and mass. Position and velocity 
are defined as GLM vec3<float> objects.  
## Debe haber iluminación simple.

A single source of light is always initialized at (0.5, 0.5, 0.5).
Specular reflection is implemented in each shader.

TODO: Color is white. Implement variable color.

TODO: Source of light is a cube, better if it is a sphere.

## El modelo de las particulas debe ser una esfera con una textura.

Each particle is drawn as a texturized sphere. Texture file is ```football.png``` 
located in ```resources``` folder.

## Debe existir unas particulas especiales con masa distinta al resto, y se debe poder controlar la cantidad de partículas especiales de la simulación, así como la masa especial de dichas partículas.

Pending.

## Debe existir un control en primera persona de la cámara.

Camera system is implemented using OpenGL, GLFW, and GLM. It supports first-person camera movement and orientation using keyboard and mouse input.

| Key                           | Action                      |
|-------------------------------|-----------------------------|
| `W`                           | Move forward                |
| `A`                           | Move left                   |
| `S`                           | Move backward               |
| `D`                           | Move right                  |
| `SPACE`                       | Move up                     |
| `LEFT CONTROL`                | Move down                   |
| `LEFT SHIFT (pressed)`        | Speed up (x4)               |
| `Left Mouse Button (pressed)` | Look around (mouse movement) |
| `R`                           | Reset time step             |
| `T`                           | Reset camera                |
| `I`                           | Increase time step          |
| `K`                           | Decrease time step          |
| `Esc`                         | Exit                        |

* Debe existir una interfaz que permita al usuario cambiar los parámetros de la simulación
en tiempo real, como la velocidad de la simulación, el número de cuerpos, la masa general
de los cuerpos y si se procesa la lógica en la CPU o en la GPU.

Pending.
* interfaz
* velocidad simulacion
* numero cuerpos
* masa general
* GPU/CPU

* La interfaz se debe activar con una tecla y desactivar con la misma para facilitar el manejo
de la cámara

Pending.