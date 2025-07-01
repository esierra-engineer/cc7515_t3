//
// Created by erick on 5/8/25.
//
#include <glm/glm.hpp>

#ifndef NBODY_H
#define NBODY_H
#define debug false

/**
 * A structure to define a body
 */
class Body {
public:
    glm::vec3 posVec; // position vector (x,y,z)
    glm::vec3 velVec; // velocity vector (x,y,z)
    bool special = false; // is special?
};

void simulateNBodyCPU(Body* bodies, int n, float dt, float *mass, float* special_mass);
void simulateNBodyCUDA(Body* h_bodies, const char* kernelFilename, int localSize, int n, float dt, float* mass, float* special_mass);
void generateRandomBodies(Body* bodies, int n, int n_specials);

#endif
