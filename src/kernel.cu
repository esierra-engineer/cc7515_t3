//
// Created by erick on 5/23/25.
// kernel_1_global-memory.cu
//

#include <iostream>
#include <cstdlib> // For integer abs()
#include "../include/nbody.h"
#define G_CONSTANT 6.67430e-11f
#define debug false

// universal gravitational constant
const float G = G_CONSTANT;


extern "C" __global__ void updateBodies(Body *bodies, int n, float dt, float mass, float special_mass) {
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    // i is the body index (global thread index),
    // each thread handles ONE BODY
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // index can go no longer than the number of bodies
    if (i >= n) return;

    // for this body
    Body bi = bodies[i];

    // border conditions, initial net force is null
    float F = 0.0f;
    //float3 VecF = make_float3(0.0f, 0.0f, 0.0f);

    float mi = bi.mass < 0 ? (bi.special ? special_mass : mass) : bi.mass;

    if (debug) {
        printf("(IN) Body %d: pos=(%f,%f,%f) vel=(%f,%f,%f), mass = (%f)\n", i, bi.posVec.x, bi.posVec.y, bi.posVec.z,
               bi.velVec.x, bi.velVec.y, bi.velVec.z, mi);
    }


    // for each other body
    for (int j = 0; j < n; ++j) {
        // skip self
        if (i == j) continue;
        // this other body (global memory access here)
        Body bj = bodies[j];

        float mj = bj.mass < 0 ? (bj.special ? special_mass : mass) : bj.mass;

        // the distance between bodies in x, y and z
        float dx = bj.posVec.x - bi.posVec.x;
        float dy = bj.posVec.y - bi.posVec.y;
        float dz = bj.posVec.z - bi.posVec.z;

        // Euclidean distance (avoid division by zero by adding a small constant)
        float distSqr = dx * dx + dy * dy + dz * dz + FLT_MIN;

        // Newton's gravity, vectorial form
        // inverse of the distance
        float invDist = rsqrtf(distSqr);

        // F = G * m1 * m2 / ...
        F = G * mi * mj * powf(invDist, 3.0f);

        if (debug) {
            printf("G mi mj = %f\n", G  * mi * mj);
            printf("invDist^3 = %f\n", powf(invDist, 3.0f));
            printf("F{%d}_{%d} = %f \n", i, j, F);
        }

        // update net force over body for x,y,z
        //VecF = VecF + (F * make_float3(dx, dy, dz));

        Fx += F * dx;
        Fy += F * dy;
        Fz += F * dz;
        if (debug) {
            printf("(OUT) Body %d: dr=(%f,%f,%f)\n", i, dx, dy, dz);
            //printf("(OUT) Body %d: force=(%f,%f,%f)\n", i, glm::vec3(dx, dy, dz).x * F, glm::vec3(dx, dy, dz).y * F, glm::vec3(dx, dy, dz).z * F);
            printf("(OUT) Body %d: force=(%f,%f,%f)\n", i, Fx, Fy, Fz);
        }
    }

    /** update velocity
     * if (F = m * a) and (a =  dv/dt)
     * then (F = m * dv/dt)
     * then (dv = F * dt / m)
     * then v = v + dv
     * **/
    bi.velVec.x += Fx / mi * (dt);
    bi.velVec.y += Fy / mi * (dt);
    bi.velVec.z += Fz / mi * (dt);
    // bi.velVec += VecF * 1.0f/mi * (dt);
    // bi.posVec += bi.velVec * (dt);

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
