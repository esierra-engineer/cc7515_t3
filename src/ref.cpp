//
// Created by erick on 6/23/25.
//
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

// include your Shader and Camera classes here
// ...

#include <vector>
#include <random>

// Constants
const unsigned int SCR_WIDTH = 1280, SCR_HEIGHT = 720;
const float G = 6.67430e-11f;
const int NUM_CUBES = 100;
float dt = 0.01f;
float timeScale = 1.0f;

// Globals
std::vector<glm::vec3> positions;
std::vector<glm::vec3> velocities;
std::vector<float> masses;

// Utility: random positions
std::vector<glm::vec3> generateRandomCubePositions(int N, float minX, float maxX,
                                                   float minY, float maxY,
                                                   float minZ, float maxZ) {
    std::vector<glm::vec3> vec(N);
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dx(minX, maxX),
                                             dy(minY, maxY),
                                             dz(minZ, maxZ);
    for (int i = 0; i < N; ++i)
        vec[i] = { dx(rng), dy(rng), dz(rng) };
    return vec;
}

void updatePhysics() {
    for (int i = 0; i < NUM_CUBES; ++i) {
        glm::vec3 netF(0.0f);
        for (int j = 0; j < NUM_CUBES; ++j) {
            if (i == j) continue;
            glm::vec3 r = positions[j] - positions[i];
            float d = glm::length(r) + 1e-5f;
            glm::vec3 dir = glm::normalize(r);
            float forceMag = G * masses[i] * masses[j] / (d * d);
            netF += forceMag * dir;
        }
        glm::vec3 a = netF / masses[i];
        velocities[i] += a * dt * timeScale;
    }
    for (int i = 0; i < NUM_CUBES; ++i) {
        positions[i] += velocities[i] * dt * timeScale;
    }
}

int main() {
    // GLFW + OpenGL init
    glfwInit();
    // (hint: set version…)
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "3D Gravity", NULL, NULL);
    glfwMakeContextCurrent(window);
    gladLoadGLLoader((GLADloadproc)glfwGetProcAddress);
    glEnable(GL_DEPTH_TEST);

    Shader shader("vertex.glsl", "fragment.glsl");
    Camera camera({0,0,3});

    // Initialize cubes
    positions = generateRandomCubePositions(NUM_CUBES,-100,100,-100,100,-100,100);
    velocities.assign(NUM_CUBES, glm::vec3(0.0f));
    masses.assign(NUM_CUBES, 1.0f); // or random

    // Setup cube VAO/VBO
    // … load cube vertices …

    while (!glfwWindowShouldClose(window)) {
        float currentFrame = glfwGetTime();
        // calculate dt, process input (WASD, mouse, +/- timeScale)

        updatePhysics();

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        shader.use();
        glm::mat4 proj = glm::perspective(glm::radians(camera.Zoom),
                            (float)SCR_WIDTH/SCR_HEIGHT, 0.1f, 1000.0f);
        glm::mat4 view = camera.GetViewMatrix();
        shader.setMat4("projection", proj);
        shader.setMat4("view", view);

        glBindVertexArray(cubeVAO);
        for (int i = 0; i < NUM_CUBES; ++i) {
            glm::mat4 model(1.0f);
            model = glm::translate(model, positions[i]);
            shader.setMat4("model", model);
            glDrawArrays(GL_TRIANGLES, 0, 36);
        }

        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    glfwTerminate();
    return 0;
}
