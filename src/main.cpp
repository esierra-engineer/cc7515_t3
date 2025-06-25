#include <random>
#define GLAD_GL_IMPLEMENTATION
#include <glad/gl.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "linmath.h"

#include <cstdlib>
#include <cstddef>
#include <cstdio>
#include <iostream>
#include <thread>

#include "utils.h"

typedef struct Vertex
{
    vec2 pos;
    vec3 col;
} Vertex;

// static const Vertex vertices[4] =
// {
//     { { -0.6f, -0.4f }, { 1.f, 0.f, 0.f } },
//     { {  0.6f, -0.4f }, { 0.f, 1.f, 0.f } },
//     { {   0.f,  0.6f }, { 0.f, 0.f, 1.f } },
//     { {  0.4f,  0.6f }, {0.f, 0.f, 0.f } },
// };

Vertex* getVertices(int count) {
    static std::mt19937 gen(std::random_device{}());  // solo se inicializa una vez
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    Vertex* verts = new Vertex[count];
    for (int i = 0; i < count; ++i) {
        verts[i].pos[0] = dist(gen);
        verts[i].pos[1] = dist(gen);
        verts[i].col[0] = dist(gen) * 0.5f + 0.5f; // color entre 0.0 y 1.0
        verts[i].col[1] = dist(gen) * 0.5f + 0.5f;
        verts[i].col[2] = dist(gen) * 0.5f + 0.5f;
    }
    return verts;
}

static const char* vertex_shader_text =
"#version 330\n"
"uniform mat4 MVP;\n"
"in vec3 vCol;\n"
"in vec2 vPos;\n"
"out vec3 color;\n"
"void main()\n"
"{\n"
"    gl_Position = MVP * vec4(vPos, 0.0, 1.0);\n"
"    color = vCol;\n"
"}\n";

static const char* fragment_shader_text =
"#version 330\n"
"in vec3 color;\n"
"out vec4 fragment;\n"
"uniform sampler2D tex;\n"
"in vec2 texCoord;\n"
"void main()\n"
"{\n"
"   vec2 coord = gl_PointCoord;\n"
"   fragment = vec4(color, 1.0);\n"
"}\n";

static void error_callback(int error, const char* description)
{
    fprintf(stderr, "Error: %s\n", description);
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, GLFW_TRUE);
    if (key == GLFW_KEY_E && action == GLFW_PRESS)
        std::cout << "user pressed key: " << key << "\n";
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS) {
        std::cout << "user pressed mouse key: " << button << "\n";
    }
}

int main()
{
    glfwSetErrorCallback(error_callback);

    if (!glfwInit())
        exit(EXIT_FAILURE);

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    GLFWwindow* window = glfwCreateWindow(800, 600, "N-Body Viewer", nullptr, nullptr);
    if (!window)
    {
        glfwTerminate();
        exit(EXIT_FAILURE);
    }

    glfwSetKeyCallback(window, key_callback);

    glfwSetMouseButtonCallback(window, mouse_button_callback);

    glfwMakeContextCurrent(window);
    gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);

    // NOTE: OpenGL error checks have been omitted for brevity

    GLuint vertex_buffer;
    glGenBuffers(1, &vertex_buffer);
    glBindBuffer(GL_ARRAY_BUFFER, vertex_buffer);
    // int vertexCount;
    // Vertex* dynamicVertices = getVertices(vertexCount);
    // glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertexCount, dynamicVertices, GL_STATIC_DRAW);

    const GLuint vertex_shader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex_shader, 1, &vertex_shader_text, NULL);
    glCompileShader(vertex_shader);

    const GLuint fragment_shader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment_shader, 1, &fragment_shader_text, NULL);
    glCompileShader(fragment_shader);

    glEnable(GL_PROGRAM_POINT_SIZE);
    glPointSize(15);

    const GLuint program = glCreateProgram();
    glAttachShader(program, vertex_shader);
    glAttachShader(program, fragment_shader);
    glLinkProgram(program);

    const GLint mvp_location = glGetUniformLocation(program, "MVP");
    const GLint vpos_location = glGetAttribLocation(program, "vPos");
    const GLint vcol_location = glGetAttribLocation(program, "vCol");

    GLuint vertex_array;
    glGenVertexArrays(1, &vertex_array);
    glBindVertexArray(vertex_array);
    glEnableVertexAttribArray(vpos_location);
    glVertexAttribPointer(vpos_location, 2, GL_FLOAT, GL_FALSE,
                          sizeof(Vertex), (void*) offsetof(Vertex, pos));
    glEnableVertexAttribArray(vcol_location);
    glVertexAttribPointer(vcol_location, 3, GL_FLOAT, GL_FALSE,
                          sizeof(Vertex), (void*) offsetof(Vertex, col));

    int vertexCount = 1000;

    Vertex* dynamicVertices = getVertices(vertexCount);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertexCount, dynamicVertices, GL_STATIC_DRAW);

    GLuint textureID = loadTexture("/home/erick/git/cc7515_t3/resources/wall.jpg");
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(64); // tamaño del punto


    while (!glfwWindowShouldClose(window))
    {

        int width, height;
        glfwGetFramebufferSize(window, &width, &height);
        const float ratio = width / (float) height;

        glViewport(0, 0, width, height);

        glClearColor(1,1,1,1.0);
        glClear(GL_COLOR_BUFFER_BIT);

        mat4x4 m, p, mvp;
        mat4x4_identity(m);

        mat4x4_translate_in_place(m, static_cast<float>(glfwGetTime()) * 0.1f, - static_cast<float>(glfwGetTime()) * 0.05f, 0.0f);
        mat4x4_rotate_Z(m, m, 1 * static_cast<float>(glfwGetTime()));
        mat4x4_ortho(p, -ratio, ratio, -1.f, 1.f, 1.f, -1.f);
        mat4x4_mul(mvp, p, m);

        glUseProgram(program);
        glUniformMatrix4fv(mvp_location, 1, GL_FALSE, reinterpret_cast<const GLfloat *>(&mvp));
        glBindVertexArray(vertex_array);

        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, textureID);

        // Enviar uniform al shader
        GLint texUniformLoc = glGetUniformLocation(program, "tex");
        glUniform1i(texUniformLoc, 0); // usamos GL_TEXTURE0

        // DRAW TRIANGLES
        glDrawArrays(GL_POINTS, 0, vertexCount);

        glfwSwapBuffers(window);
        glfwPollEvents();

        std::this_thread::sleep_for(std::chrono::milliseconds(500)); // Pause 0.5 seconds
        //glfwWaitEventsTimeout(1.5);
    }

    glfwDestroyWindow(window);

    glfwTerminate();
    exit(EXIT_SUCCESS);
}
