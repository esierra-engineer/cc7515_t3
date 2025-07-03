//
// Created by erick on 5/8/25.
//
#include "nbody.h"
#include "utils.h"

#include <csv.h>
#include <cuda.h>
#include <iostream>
#include <random>

void check(CUresult err, const char* func, const char* file, int line) {
    if (err != CUDA_SUCCESS) {
        const char* errStr;
        cuGetErrorString(err, &errStr);
        std::cerr << "CUDA error at " << file << ":" << line << " code=" << err
                  << " \"" << func << "\" : " << errStr << "\n";
        exit(1);
    }
}

void generateRandomBodies(Body* bodies, int n, int n_specials, bool from_file, char* csv_path) {
#define FACTOR 30.0f
    if (from_file) {
        io::CSVReader<8> in(csv_path);
        in.read_header(io::ignore_extra_column, "index", "xpos", "ypos", "zpos", "xvel", "yvel", "zvel", "special");
        int index, special; float xpos, ypos, zpos, xvel, yvel, zvel;
        while(in.read_row(index, xpos, ypos, zpos, xvel, yvel, zvel, special)){
            bodies[index].posVec = glm::vec3(xpos, ypos, zpos);
            bodies[index].velVec = glm::vec3(xvel, yvel, zvel);
            bodies[index].special = special;
        }
    }

    else {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-FACTOR, FACTOR);
        for (int i = 0; i < n; ++i) {
            bodies[i].posVec = glm::vec3(dist(gen), dist(gen), dist(gen));
            bodies[i].velVec = glm::vec3(0.0f);
            bodies[i].special = false;
        }
        for (int i = n; i < n + n_specials; ++i) {
            bodies[i].posVec = glm::vec3(dist(gen), dist(gen), dist(gen));
            bodies[i].velVec = glm::vec3(0.0f);
            bodies[i].special = true;
        }
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