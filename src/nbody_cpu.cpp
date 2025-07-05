//
// Created by erick on 5/8/25.
//
#include "nbody.h"


inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

const float G = 6.67430e-11f;

void simulateNBodyCPU(Body* bodies, int n, float dt, float *mass, float* special_mass){
    float deref_mass = *mass;
    float deref_special_mass = *special_mass;

    glm::vec3 VecF;
    float F;

    for (int i = 0; i < n; ++i) {
        VecF = glm::vec3(0.0f);
        F = 0.0f;

        Body bi = bodies[i];
        float mi = bi.mass < 0 ? (bi.special ? deref_special_mass : deref_mass) : bi.mass;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            Body bj = bodies[j];
            float mj = bj.mass < 0 ? (bj.special ? deref_special_mass : deref_mass) : bj.mass;

            float dx = bj.posVec.x - bi.posVec.x;
            float dy = bj.posVec.y - bi.posVec.y;
            float dz = bj.posVec.z - bi.posVec.z;

            float distSqr = dx * dx + dy * dy + dz * dz + 1.0f;

            // inverse of the distance
            float invDist = rsqrtf(distSqr);

            // F = G * m1 * m2 / ...
            F = G * mi * mj * powf(invDist, 3.0f);
            VecF += glm::vec3(dx, dy, dz) * F;

            if (debug) {
                printf("(OUT) Body %d: dr=(%f,%f,%f)\n", i, dx, dy, dz);
                printf("(OUT) Body %d: force=(%f,%f,%f)\n", i, glm::vec3(dx, dy, dz).x * F, glm::vec3(dx, dy, dz).y * F, glm::vec3(dx, dy, dz).z * F);
                printf("(OUT) Body %d: force=(%f,%f,%f)\n", i, VecF.x, VecF.y, VecF.z);
            }
            // Fx += F * dx;
            // Fy += F * dy;
            // Fz += F * dz;
        }
        // bi.velVec.x += Fx / mi * (dt * time_scale);
        // bi.velVec.y += Fy / mi * (dt * time_scale);
        // bi.velVec.z += Fz / mi * (dt * time_scale);
        // glm::vec3 VecF = glm::vec3(Fx, Fy, Fz);

        bi.velVec += VecF * 1.0f/mi * dt;
        bi.posVec += bi.velVec * dt;

        // bi.posVec.x += bi.velVec.x * dt;
        // bi.posVec.y += bi.velVec.y * dt;
        // bi.posVec.z += bi.velVec.z * dt;

        if (debug) {
            printf("(OUT) Body %d: pos=(%f,%f,%f) vel=(%f,%f,%f)\n", i,
                   bi.posVec.x, bi.posVec.y, bi.posVec.z, bi.velVec.x, bi.velVec.y, bi.velVec.z);
        }

        bodies[i] = bi;
    }
}