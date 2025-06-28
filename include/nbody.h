//
// Created by erick on 5/8/25.
//
#include <cstdlib>
#include<glm/glm.hpp>

#ifndef NBODY_H
#define NBODY_H

/**
 * A structure to define a body
 */
class Body {
public:
    // position
    glm::vec3 posVec;
    float x = posVec[0];
    float y = posVec[1];
    float z = posVec[2];
    // velocity
    glm::vec3 velVec;
    float vx = velVec[0];
    float vy = velVec[1];
    float vz = velVec[2];
    // mass
    float mass;
};

void simulateNBodyCPU(Body* bodies, int n, int steps, float dt = 0.01f);
void simulateNBodyCUDA(Body* h_bodies, const char* kernelFilename, int localSize, int n, float dt=0.01f);
void generateRandomBodies(Body* bodies, int n);

#endif
