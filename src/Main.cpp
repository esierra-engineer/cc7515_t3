//------- Ignore this ----------
#include <cuda_runtime_api.h>
#include<filesystem>
namespace fs = std::filesystem;
#include <unistd.h>
//------------------------------

#include<iostream>
#include<glad/glad.h>
#include<GLFW/glfw3.h>
#include<stb/stb_image.h>
#include<glm/glm.hpp>
#include<glm/gtc/matrix_transform.hpp>
#include<glm/gtc/type_ptr.hpp>

#include "../include/Texture.h"
#include"../include/shaderClass.h"
#include"../include/VAO.h"
#include"../include/VBO.h"
#include"../include/EBO.h"
#include "../include/Camera.h"

#include "../include/sphere_vertices.h"
#include "../include/sphere_indices.h"
#include "nbody.h"
#include <cuda_gl_interop.h>
#include "imgui/imgui.h"
#define DEFAULT_DT 0.01f

const unsigned int width = 800;
const unsigned int height = 800;
constexpr auto camera_init_pos = glm::vec3(0.0f, 0.0f, 120.0f);
const unsigned int Nbodies = 100;
const double scale = 1.1;
float dt = DEFAULT_DT;

// root folder path
fs::path src_folder = "/media/storage/git/cc7515_t3/src/shaders";
fs::path resources_folder = "/media/storage/git/cc7515_t3/resources";

// GLfloat* lightVertices = sphereVertices;
// GLuint* lightIndices = sphereIndices;

GLfloat lightVertices[] =
{ //     COORDINATES     //
	-0.1f, -0.1f,  0.1f,
	-0.1f, -0.1f, -0.1f,
	 0.1f, -0.1f, -0.1f,
	 0.1f, -0.1f,  0.1f,
	-0.1f,  0.1f,  0.1f,
	-0.1f,  0.1f, -0.1f,
	 0.1f,  0.1f, -0.1f,
	 0.1f,  0.1f,  0.1f
};

GLuint lightIndices[] =
{
	0, 1, 2,
	0, 2, 3,
	0, 4, 7,
	0, 7, 3,
	3, 7, 6,
	3, 6, 2,
	2, 6, 5,
	2, 5, 1,
	1, 5, 4,
	1, 4, 0,
	4, 5, 6,
	4, 6, 7
};


void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		std::cout << "Closing...";
		glfwSetWindowShouldClose(window, true);
	}

	if (key == GLFW_KEY_I && action == GLFW_PRESS) {
		std::cout << "Increasing time step to ";
		dt += 0.01;
		std::cout << dt << "\n";
	}

	if (key == GLFW_KEY_K && action == GLFW_PRESS) {
		std::cout << "Decreasing time step to ";
		if (dt >= 0.01) dt -= 0.01;
		std::cout << dt << "\n";
	}

	if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		std::cout << "Defaulting time step to ";
		dt = DEFAULT_DT;
		std::cout << dt << "\n";
	}
}


void drawSpheres(Body* bodies, const Shader& shaderProgram, int N);

int main()
{
	// Initialize GLFW
	glfwInit();

	// Tell GLFW what version of OpenGL we are using 
	// In this case we are using OpenGL 3.3
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	// Tell GLFW we are using the CORE profile
	// So that means we only have the modern functions
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// Create a GLFWwindow object of 800 by 800 pixels, naming it "YoutubeOpenGL"
	GLFWwindow* window = glfwCreateWindow(width, height, "CC7515 Tarea 3", NULL, NULL);
	// Error check if the window fails to create
	if (window == NULL)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	// Introduce the window into the current context
	glfwMakeContextCurrent(window);

	//Load GLAD so it configures OpenGL
	gladLoadGL();
	// Specify the viewport of OpenGL in the Window
	// In this case the viewport goes from x = 0, y = 0, to x = width, y = height
	glViewport(0, 0, width, height);

	// Generates Shader object using shaders default.vert and default.frag
	fs::path file = "default.vert";
	fs::path default_vert_path = src_folder / file;

	file = "default.frag";
	fs::path default_frag_path = src_folder / file;

	Shader shaderProgram((default_vert_path.c_str()), (default_frag_path.c_str()));
	// Generates Vertex Array Object and binds it
	VAO VAO1;
	VAO1.Bind();
	// Generates Vertex Buffer Object and links it to vertices
	VBO VBO1(sphereVertices, sizeof(sphereVertices));
	// Generates Element Buffer Object and links it to indices
	EBO EBO1(sphereIndices, sizeof(sphereIndices));
	// Links VBO attributes such as coordinates and colors to VAO
	VAO1.LinkAttrib(VBO1, 0, 3, GL_FLOAT, 11 * sizeof(float), nullptr);
	VAO1.LinkAttrib(VBO1, 1, 3, GL_FLOAT, 11 * sizeof(float), reinterpret_cast<void *>(3 * sizeof(float)));
	VAO1.LinkAttrib(VBO1, 2, 2, GL_FLOAT, 11 * sizeof(float), reinterpret_cast<void *>(6 * sizeof(float)));
	VAO1.LinkAttrib(VBO1, 3, 3, GL_FLOAT, 11 * sizeof(float), reinterpret_cast<void *>(8 * sizeof(float)));
	// Unbind all to prevent accidentally modifying them
	VAO1.Unbind();
	VBO1.Unbind();
	EBO1.Unbind();


	// Shader for light cube
	file = "light.vert";
	fs::path light_vert_path = src_folder / file;

	file = "light.frag";
	fs::path light_frag_path = src_folder / file;

	Shader lightShader((light_vert_path.string().c_str()), (light_frag_path.string().c_str()));
	// Generates Vertex Array Object and binds it
	VAO lightVAO;
	lightVAO.Bind();
	// Generates Vertex Buffer Object and links it to vertices
	VBO lightVBO(lightVertices, sizeof(lightVertices));
	// Generates Element Buffer Object and links it to indices
	EBO lightEBO(lightIndices, sizeof(lightIndices));
	// Links VBO attributes such as coordinates and colors to VAO
	lightVAO.LinkAttrib(lightVBO, 0, 3, GL_FLOAT, 3 * sizeof(float), nullptr);
	// Unbind all to prevent accidentally modifying them
	lightVAO.Unbind();
	lightVBO.Unbind();
	lightEBO.Unbind();

	glm::vec4 lightColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
	glm::vec3 lightPos = glm::vec3(0.5f, 0.5f, 0.5f);
	glm::mat4 lightModel = glm::mat4(1.0f);
	lightModel = glm::translate(lightModel, lightPos);

	glm::vec3 spherePos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::mat4 sphereModel = glm::mat4(1.0f);
	sphereModel = glm::translate(sphereModel, spherePos);


	lightShader.Activate();
	glUniformMatrix4fv(glGetUniformLocation(lightShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(lightModel));
	glUniform4f(glGetUniformLocation(lightShader.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	shaderProgram.Activate();
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(sphereModel));
	glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
	glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);

	// Texture
	file = "football.png";
	fs::path texture_path = resources_folder / file;
	Texture brickTex((texture_path).c_str(), GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE);
	brickTex.texUnit(shaderProgram, "tex0", 0);



	// Enables the Depth Buffer
	glEnable(GL_DEPTH_TEST);

	// Creates camera object
	Camera camera(width, height, camera_init_pos);

	// create Bodies vector
	Body* bodies = new Body[Nbodies];
	// give random positions
	generateRandomBodies(bodies, Nbodies);

	std::string kernel_filename = "kernel_1_global-memory_1D.ptx";
	int local_size = 32;

	// CUDA interop
	cudaGraphicsResource_t cudaVBO;
	cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO1.ID, cudaGraphicsMapFlagsWriteDiscard);

	// Main while loop
	while (!glfwWindowShouldClose(window))
	{
		// Specify the color of the background
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		// Clean the back buffer and depth buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Handles camera inputs
		camera.Inputs(window);
		// Updates and exports the camera matrix to the Vertex Shader
		camera.updateMatrix(45.0f, 0.1f, 100.0f);


		// Tells OpenGL which Shader Program we want to use
		shaderProgram.Activate();
		// Exports the camera Position to the Fragment Shader for specular lighting
		glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);
		// Export the camMatrix to the Vertex Shader of the pyramid
		camera.Matrix(shaderProgram, "camMatrix");
		// Binds texture so that is appears in rendering
		brickTex.Bind();
		// Bind the VAO so OpenGL knows to use it
		VAO1.Bind();
		// CUDA interop
		Body* devicePtr;
		size_t size;
		cudaGraphicsMapResources(1, &cudaVBO);
		cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&devicePtr), &size, cudaVBO);
		// update positions
		simulateNBodyCUDA(bodies, kernel_filename.c_str(), local_size, Nbodies, dt);

		cudaGraphicsUnmapResources(1, &cudaVBO);
		// Draw primitives, number of indices, datatype of indices, index of indices
		drawSpheres(bodies, shaderProgram, Nbodies);

		// Tells OpenGL which Shader Program we want to use
		lightShader.Activate();
		// Export the camMatrix to the Vertex Shader of the light cube
		camera.Matrix(lightShader, "camMatrix");
		// Bind the VAO so OpenGL knows to use it
		lightVAO.Bind();
		// Draw primitives, number of indices, datatype of indices, index of indices
		glDrawElements(GL_TRIANGLES, sizeof(lightIndices) / sizeof(int), GL_UNSIGNED_INT, 0);

		// Window Test
		// Create a window called "My First Tool", with a menu bar.
		bool my_tool_active = 1;
		ImGui::Begin("My First Tool", &my_tool_active, ImGuiWindowFlags_MenuBar);
		if (ImGui::BeginMenuBar())
		{
			if (ImGui::BeginMenu("File"))
			{
				if (ImGui::MenuItem("Open..", "Ctrl+O")) { /* Do stuff */ }
				if (ImGui::MenuItem("Save", "Ctrl+S"))   { /* Do stuff */ }
				if (ImGui::MenuItem("Close", "Ctrl+W"))  { my_tool_active = false; }
				ImGui::EndMenu();
			}
			ImGui::EndMenuBar();
		}

		// Edit a color stored as 4 floats
		float my_color[4];
		ImGui::ColorEdit4("Color", my_color);

		// Generate samples and plot them
		float samples[100];
		for (int n = 0; n < 100; n++)
			samples[n] = sinf(n * 0.2f + ImGui::GetTime() * 1.5f);
		ImGui::PlotLines("Samples", samples, 100);

		// Display contents in a scrolling region
		ImGui::TextColored(ImVec4(1,1,0,1), "Important Stuff");
		ImGui::BeginChild("Scrolling");
		for (int n = 0; n < 50; n++)
			ImGui::Text("%04d: Some text", n);
		ImGui::EndChild();
		ImGui::End();

		// listen to key events
		glfwSetKeyCallback(window, key_callback);

		// Swap the back buffer with the front buffer
		glfwSwapBuffers(window);
		// Take care of all GLFW events
		glfwPollEvents();
	}

	// Delete all the objects we've created
	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
	brickTex.Delete();
	shaderProgram.Delete();
	lightVAO.Delete();
	lightVBO.Delete();
	lightEBO.Delete();
	lightShader.Delete();
	// Delete window before ending the program
	glfwDestroyWindow(window);
	// Terminate GLFW before ending the program
	glfwTerminate();
	return 0;
}

void drawSpheres(Body* bodies, const Shader& shaderProgram, int N) {
	for (int i = 0; i < N; ++i) {
		if (debug) {
			std::cout << "[drawSpheres] Animating element in index " << i << ":\n";

		std::cout << "[drawSpheres] from position" <<
				" x = " <<
				bodies[i].posVec.x <<
				" y = " <<
				bodies[i].posVec.y <<
				" z = " <<
				bodies[i].posVec.z << "\n";
		}

		glm::mat4 model = glm::mat4(1.0f);

		glm::vec3 newPos = bodies[i].posVec;

		if (debug) {
			std::cout << "[drawSpheres] to position" <<
			" x = " <<
			newPos.x <<
				" y = " <<
					newPos.y <<
						" z = " <<
							newPos.z << "\n";
		}

		model = glm::translate(model, newPos);
		model = glm::scale(model, glm::vec3(scale, scale, scale));

		glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(model));
		glDrawElements(GL_TRIANGLES, sizeof(sphereIndices) / sizeof(GLuint), GL_UNSIGNED_INT, nullptr);
	}
}