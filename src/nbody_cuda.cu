//
// Created by erick on 5/8/25.
//
#include "nbody.h"
#include <cuda_runtime.h>
#include "utils.h"
#include <iostream>
#define DEFAULT_MASS 1e9f

/**
 * Host Function (CPU-side)
 * Copy the input data from host memory to device memory, also known as host-to-device transfer.
 * Load the GPU program and execute, caching data on-chip for performance.
 * Copy the results from device memory to host memory, also called device-to-host transfer.
 * source: https://developer.nvidia.com/blog/cuda-refresher-cuda-programming-model/

 * @param h_bodies Array of Body
 * @param kernelFilename Kernel filename
 * @param localSize Block size
 * @param n Number of bodies
 */
void simulateNBodyCUDA(Body* h_bodies, const char* kernelFilename, int localSize, int n, float dt, float* mass, float* special_mass) {
    float deref_mass = *mass * DEFAULT_MASS;
    float deref_special_mass = *special_mass * DEFAULT_MASS;

    // destination memory address pointer
    Body* d_bodies;
    // in memory size of n bodies
    size_t size = n * sizeof(Body);

    // allocate GPU memory
    cudaMalloc(&d_bodies, size);

    // copy data between host and device
    cudaMemcpy(d_bodies, h_bodies, size, cudaMemcpyHostToDevice);

    // 1D Kernel configuration
    int threadsPerBlock = localSize;
    int blocksNeeded = (n + threadsPerBlock - 1) / threadsPerBlock;

    dim3 blockDim(threadsPerBlock);
    dim3 gridDim(blocksNeeded);

    size_t sharedMemSize = threadsPerBlock * sizeof(Body);

    // Kernel
    CUcontext context;
    CUfunction kernel = loadKernelSource(kernelFilename, &context);

    // Kernel args deben ser punteros a los datos
    void* kernelArgs[] = {
        (void*) &d_bodies,
        (void*) &n,
        (void*) &dt,
        (void*) &deref_mass,
        (void*) &deref_special_mass
    };

    checkCudaErrors(
        cuLaunchKernel(
            kernel,
            gridDim.x, 1, 1,                    // grid
            blockDim.x, 1, 1,             // block
            sharedMemSize, nullptr,                        // shared memory and stream
            kernelArgs, nullptr)                // args
        );

    // wait
    cudaDeviceSynchronize();

    // retrieve the updated positions and velocities
    cudaMemcpy(h_bodies, d_bodies, size, cudaMemcpyDeviceToHost);

    // free memory
    cudaFree(d_bodies);

    cuCtxDestroy(context);
}