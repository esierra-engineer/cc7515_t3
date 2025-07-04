//
// Created by erick on 5/8/25.
//
#include "nbody.h"
#define NEAR_ZERO 0.1f
#define DEFAULT_MASS 1e9f

inline float rsqrtf(float x) {
    return 1.0f / sqrtf(x);
}

const float G = 6.67430e-11f;

void simulateNBodyCPU(Body* bodies, int n, float dt, float *mass, float* special_mass){
    float deref_mass = *mass * DEFAULT_MASS;
    float deref_special_mass = *special_mass * DEFAULT_MASS;

    float F, Fx, Fy, Fz;

    for (int i = 0; i < n; ++i) {
        Body bi = bodies[i];
        float mi = bi.mass < 0 ? (bi.special ? deref_special_mass : deref_mass) : bi.mass / DEFAULT_MASS;
        F= 0.0, Fx = 0.0f, Fy = 0.0f, Fz = 0.0f;

        for (int j = 0; j < n; ++j) {
            if (i == j) continue;
            Body bj = bodies[j];
            float mj = bj.mass < 0 ? (bj.special ? deref_special_mass : deref_mass) : bj.mass / DEFAULT_MASS;

            float dx = bj.posVec.x - bi.posVec.x;
            float dy = bj.posVec.y - bi.posVec.y;
            float dz = bj.posVec.z - bi.posVec.z;

            float distSqr = dx * dx + dy * dy + dz * dz + NEAR_ZERO * NEAR_ZERO;

            if (abs(distSqr) > NEAR_ZERO * NEAR_ZERO) {
                // inverse of the distance
                float invDist = rsqrtf(distSqr);

                // F = G * m1 * m2 / ...
                F = G * mi * mj * powf(invDist, 3.0f);
            }

            Fx += F * dx;
            Fy += F * dy;
            Fz += F * dz;
        }
        bi.velVec.x += Fx / mi * dt;
        bi.velVec.y += Fy / mi * dt;
        bi.velVec.z += Fz / mi * dt;

        bi.posVec.x += bi.velVec.x * dt;
        bi.posVec.y += bi.velVec.y * dt;
        bi.posVec.z += bi.velVec.z * dt;

        bodies[i] = bi;
    }
}