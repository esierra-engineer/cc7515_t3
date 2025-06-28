//
// Created by erick on 5/8/25.
//
#include "nbody.h"
#include "utils.h"
#include <cuda.h>
#include <iostream>

void check(CUresult err, const char* func, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << err
                  << " \"" << func << "\" : " << errStr << "\n";
        exit(1);
    }
}

void generateRandomBodies(Body* bodies, int n) {
    for (int i = 0; i < n; ++i) {
        bodies[i].posVec = glm::vec3(
            static_cast<float>(rand()) / RAND_MAX * 2.5f,
            static_cast<float>(rand()) / RAND_MAX * 2.5f,
            static_cast<float>(rand()) / RAND_MAX * 2.5f
            );
        bodies[i].velVec = glm::vec3(0.0f);
        bodies[i].mass = 1e10f;
    }
}

CUfunction loadKernelSource(const char* filename, CUcontext* context) {
    checkCudaErrors(cuInit(0));
    CUdevice device;
    checkCudaErrors(cuDeviceGet(&device, 0));
    //CUcontext context;
    checkCudaErrors(cuCtxCreate(context, 0, device));

    // Load PTX file
    CUmodule module;
    checkCudaErrors(cuModuleLoad(&module, filename));

    // Get function
    CUfunction kernel;
    checkCudaErrors(cuModuleGetFunction(&kernel, module, "updateBodies"));

    return kernel;
}