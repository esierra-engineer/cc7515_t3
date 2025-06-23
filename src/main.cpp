//
// Created by erick on 6/22/25.
//
#include <iostream>
#include <GLFW/glfw3.h>

int main()
{
    int code;
    GLFWwindow* window;

    /* Initialize the library */
    if (!glfwInit()){
        code = glfwGetError(NULL);
        std::cerr << "Could not initialize" << " " << "Error Code: " << code;
        return -1;
    }

    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(640, 480, "Hello World", NULL, NULL);

    if (!window)
    {
        glfwTerminate();
        return -1;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);



    /* Loop until the user closes the window */
    while (!glfwWindowShouldClose(window))
    {
        /* Render here */
        glClear(GL_COLOR_BUFFER_BIT);

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
    }

    glfwTerminate();
    return 0;
}