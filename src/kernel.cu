//
// Created by erick on 5/23/25.
// kernel_1_global-memory.cu
//

#include <iostream>
#include <cstdlib> // For integer abs()
#include "../include/nbody.h"
#define G_CONSTANT 6.67430e-11f
#define NEAR_ZERO 0.1f
#define debug false

// universal gravitational constant
const float G = G_CONSTANT;

extern "C" __global__ void updateBodies(Body *bodies, int n, float dt, float mass, float special_mass) {
    // i is the body index (global thread index),
    // each thread handles ONE BODY
    int i = (blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x * blockDim.y +
            threadIdx.y * blockDim.x + threadIdx.x;


    // index can go no longer than the number of bodies
    if (i >= n) return;

    // for this body
    Body bi = bodies[i];

    // border conditions, initial net force is null
    float F = 0.0f, Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

    if (debug) {
        printf("(IN) Body %d: pos=(%f,%f,%f) vel=(%f,%f,%f)\n", i, bi.posVec.x, bi.posVec.y, bi.posVec.z,
               bi.velVec.x, bi.velVec.y, bi.velVec.z);
    }

    // for each other body
    for (int j = 0; j < n; ++j) {
        // skip self
        if (i == j) continue;
        // this other body (global memory access here)
        Body bj = bodies[j];

        // the distance between bodies in x, y and z
        float dx = bj.posVec.x - bi.posVec.x;
        float dy = bj.posVec.y - bi.posVec.y;
        float dz = bj.posVec.z - bi.posVec.z;

        // Euclidean distance (avoid division by zero by adding a small constant)
        float distSqr = dx * dx + dy * dy + dz * dz + NEAR_ZERO * NEAR_ZERO;

        // Newton's gravity, vectorial form
        if (abs(distSqr) > NEAR_ZERO * NEAR_ZERO) {
            // inverse of the distance
            float invDist = rsqrtf(distSqr);

            F = G * mass * mass * powf(invDist, 3.0f);
        }

        // update net force over body for x,y,z
        Fx += F * dx;
        Fy += F * dy;
        Fz += F * dz;
    }

    /** update velocity
     * if (F = m * a) and (a =  dv/dt)
     * then (F = m * dv/dt)
     * then (dv = F * dt / m)
     * then v = v + dv
     * **/
    bi.velVec.x += Fx / mass * dt;
    bi.velVec.y += Fy / mass * dt;
    bi.velVec.z += Fz / mass * dt;

    /** update position
     * v = dx/dt
     * dx = dv * dt
     * x = x + dx
     **/
    bi.posVec.x += bi.velVec.x * dt;
    bi.posVec.y += bi.velVec.y * dt;
    bi.posVec.z += bi.velVec.z * dt;

    // store the body back into GLOBAL MEMORY
    bodies[i] = bi;

    if (debug) {
        printf("(OUT) Body %d: pos=(%f,%f,%f) vel=(%f,%f,%f)\n", i,
               bi.posVec.x, bi.posVec.y, bi.posVec.z, bi.velVec.x, bi.velVec.y, bi.velVec.z);
    }
}
