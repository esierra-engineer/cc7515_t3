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
    glm::vec3 posVec;
    glm::vec3 velVec;
    float mass;
};

void simulateNBodyCPU(Body* bodies, int n, int steps, float dt = 0.01f);
void simulateNBodyCUDA(Body* h_bodies, const char* kernelFilename, int localSize, int n, float dt=1.0/60.0);
void generateRandomBodies(Body* bodies, int n, float mass = 1e10f);

#endif
