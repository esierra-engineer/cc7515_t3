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

#include "Texture.h"
#include"shaderClass.h"
#include"VAO.h"
#include"VBO.h"
#include"EBO.h"
#include "Camera.h"

#include "sphere_vertices.h"
#include "sphere_indices.h"
#include "nbody.h"
#include <cuda_gl_interop.h>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <nlohmann/json.hpp>
#include "csv.h"

#define DEFAULT_DT 0.01f
#define DEFAULT_N_SPECIAL_BODIES 0
#define SHOW_CONF_AT_START false
#define CONF_READ_FILE false
#define MASS_SCALE 1.0e3f


using json = nlohmann::json;

// Declare variables
unsigned int width;
unsigned int height;
glm::vec3 camera_init_pos(0.0f);
float particleSize;
float specialParticleSize;
float sm;
float m;
float massFactor = 1.0f;
float specialMassFactor = 1.0f;
std::string kernel_filename;
int local_size;
float sourceLightColor[4] = {0.0f, 0.0f, 0.0f, 0.0f};
int useGPU;
bool stop;
bool read_particles_from_file = CONF_READ_FILE;
std::string particles_conf_file;


Body* bodies;
int numBodies;
int specialBodies;
float dt;
bool showConf = SHOW_CONF_AT_START;

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

Camera* cam;
json* jsoncfg;
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (key == GLFW_KEY_RIGHT_CONTROL && action == GLFW_PRESS) {
		stop = !stop;
		std::cout << (stop ? "Pausing" : "Resuming") << " movement\n";
	}

	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
		std::cout << "Closing...\n";
		glfwSetWindowShouldClose(window, true);
	}

	if (key == GLFW_KEY_I && action == GLFW_PRESS) {
		dt += 0.01f;
		std::cout << "Increasing time step to " << dt << "\n";
	}

	if (key == GLFW_KEY_K && action == GLFW_PRESS) {
		if (dt >= 0.01f) dt -= 0.01f;
		std::cout << "Decreasing time step to " << dt << "\n";
	}

	if (key == GLFW_KEY_R && action == GLFW_PRESS) {
		if (json cfg = *jsoncfg; cfg.contains("dt")) dt = cfg["dt"];
		else dt = DEFAULT_DT;
		std::cout << "Resetting time step to " << dt << "\n";
	}

	if (key == GLFW_KEY_C && action == GLFW_PRESS) {
		std::cout << (showConf ? "Closing" : "Opening") << " conf. Window... \n";
		showConf = !showConf;
	}
	// CAMERA
	if (!showConf) cam->KeyboardInputs(window, key, action);
}

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	if (showConf) return;
	cam->KeyboardInputs(window, button, action);
	cam->MouseInputs(window);

}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	if (showConf) return;
	cam->MouseInputs(window);
}

void drawSpheres(Body* bodies, const Shader& shaderProgram, int N, int offset = 0);

void showConfWindow();

void createDrawBodies(Body* drawBodies, Body* bodies, int numBodies, int specialBodies){
	for (int i = 0 ; i < numBodies ; ++i) {
		drawBodies[i] = bodies[i];
	}
	for (int i = numBodies ; i < numBodies + specialBodies ; ++i) {
		drawBodies[i] = bodies[DEFAULT_N_BODIES + i - numBodies];
	}
}


void drawSpheres(Body* bodies, const Shader& shaderProgram, int N, int offset) {
	using namespace glm;
	for (int i = offset; i < offset + N; ++i) {
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

		mat4 model = mat4(1.0f);

		vec3 newPos = bodies[i].posVec;

		if (debug) {
			std::cout << "[drawSpheres] to position" <<
			" x = " <<
			newPos.x <<
				" y = " <<
					newPos.y <<
						" z = " <<
							newPos.z << "\n";
		}

		model = translate(model, newPos);
		model = (bodies[i].special ? scale(model, vec3(specialParticleSize)) : scale(model, vec3(particleSize)));

		glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, value_ptr(model));
		glDrawElements(GL_TRIANGLES, sizeof(sphereIndices) / sizeof(GLuint), GL_UNSIGNED_INT, nullptr);
	}
}

int main(int argc, char** argv)
{
	if (argc<2) {
		std::cerr << "Please specify json path";
		return 1;
	}

	std::string json_path(argv[1]);

	std::ifstream f(json_path);
	json configJson = json::parse(f);
	jsoncfg = &configJson;

	// Assign values from JSON
	width = configJson["width"];
	height = configJson["height"];
	camera_init_pos = glm::vec3(
		configJson["camera_init_pos"]["x"],
		configJson["camera_init_pos"]["y"],
		configJson["camera_init_pos"]["z"]
	);
	particleSize = configJson["particleSize"];
	specialMassFactor = configJson["specialMass"];
	massFactor = configJson["mass"];
	kernel_filename = configJson["kernel_filename"];
	fs::path texture_path = configJson["particleTextureFile"];
	fs::path texture_path_special = configJson["specialParticleTextureFile"];
	local_size = configJson["local_size"];

	for (int i = 0; i < 4; ++i) {
		sourceLightColor[i] = configJson["sourceLightColor"][i];
	}

	useGPU = configJson["useGPU"] ? 1 : 0;
	stop = configJson["stop"];
	read_particles_from_file = configJson["read_particles_from_file"];
	if (read_particles_from_file) particles_conf_file = configJson["particles_conf_file"];

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
	//
	glfwSetKeyCallback(window, key_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	// Introduce the window into the current context
	glfwMakeContextCurrent(window);

	//Load GLAD so it configures OpenGL
	gladLoadGL();
	// Specify the viewport of OpenGL in the Window
	// In this case the viewport goes from x = 0, y = 0, to x = width, y = height
	glViewport(0, 0, width, height);

	// Generates Shader object using shaders default.vert and default.frag
	fs::path default_vert_path = "shaders/default.vert";

	fs::path default_frag_path = "shaders/default.frag";

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

	VAO VAO2;
	VAO2.Bind();
	// Generates Vertex Buffer Object and links it to vertices
	VBO VBO2(sphereVertices, sizeof(sphereVertices));
	// Generates Element Buffer Object and links it to indices
	EBO EBO2(sphereIndices, sizeof(sphereIndices));
	// Links VBO attributes such as coordinates and colors to VAO
	VAO2.LinkAttrib(VBO2, 0, 3, GL_FLOAT, 11 * sizeof(float), nullptr);
	VAO2.LinkAttrib(VBO2, 1, 3, GL_FLOAT, 11 * sizeof(float), reinterpret_cast<void *>(3 * sizeof(float)));
	VAO2.LinkAttrib(VBO2, 2, 2, GL_FLOAT, 11 * sizeof(float), reinterpret_cast<void *>(6 * sizeof(float)));
	VAO2.LinkAttrib(VBO2, 3, 3, GL_FLOAT, 11 * sizeof(float), reinterpret_cast<void *>(8 * sizeof(float)));
	// Unbind all to prevent accidentally modifying them
	VAO2.Unbind();
	VBO2.Unbind();
	EBO2.Unbind();


	// Shader for light cube
	fs::path light_vert_path = "shaders/light.vert";

	fs::path light_frag_path = "shaders/light.frag";

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


	glm::vec3 lightPos = glm::vec3(0.5f, 0.5f, 0.5f);
	glm::mat4 lightModel = glm::mat4(1.0f);
	lightModel = glm::translate(lightModel, lightPos);

	glm::vec3 spherePos = glm::vec3(0.0f, 0.0f, 0.0f);
	glm::mat4 sphereModel = glm::mat4(1.0f);
	sphereModel = glm::translate(sphereModel, spherePos);

	lightShader.Activate();
	glUniformMatrix4fv(glGetUniformLocation(lightShader.ID, "model"), 1, GL_FALSE, glm::value_ptr(lightModel));

	shaderProgram.Activate();
	glUniformMatrix4fv(glGetUniformLocation(shaderProgram.ID, "model"), 1, GL_FALSE, glm::value_ptr(sphereModel));
	glUniform3f(glGetUniformLocation(shaderProgram.ID, "lightPos"), lightPos.x, lightPos.y, lightPos.z);

	// Texture
	//fs::path texture_path = "resources/football.png";
	//fs::path texture_path_special = "resources/grass.png";
	Texture brickTex(texture_path.c_str(), GL_TEXTURE_2D, GL_TEXTURE0, GL_RGBA, GL_UNSIGNED_BYTE);
	Texture specialTex(texture_path_special.c_str(), GL_TEXTURE_2D, GL_TEXTURE1, GL_RGBA, GL_UNSIGNED_BYTE);
	brickTex.texUnit(shaderProgram, "tex0", 1);
	//specialTex.texUnit(shaderProgram, "tex1", 1);

	// Enables the Depth Buffer
	glEnable(GL_DEPTH_TEST);

	// Creates camera object
	Camera camera(width, height, camera_init_pos);
	cam = &camera;

	// CUDA interop
	cudaGraphicsResource_t cudaVBO;
	cudaGraphicsGLRegisterBuffer(&cudaVBO, VBO1.ID, cudaGraphicsMapFlagsWriteDiscard);

	// imgui
	// Setup Dear ImGui context
	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGuiIO& io = ImGui::GetIO();
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
	io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

	// Setup Platform/Renderer backends
	ImGui_ImplGlfw_InitForOpenGL(window, true);          // Second param install_callback=true will install GLFW callbacks and chain to existing ones.
	ImGui_ImplOpenGL3_Init();

	// create Bodies vector
	bodies = new Body[DEFAULT_N_BODIES + DEFAULT_N_BODIES];
	// give random positions
	generateBodies(bodies, DEFAULT_N_BODIES, DEFAULT_N_BODIES, read_particles_from_file, particles_conf_file);

	if (configJson.contains("numBodies"))
		numBodies = configJson["numBodies"];
	else
		numBodies = DEFAULT_N_BODIES;

	if (configJson.contains("specialBodies"))
		specialBodies = configJson["specialBodies"];
	else
		specialBodies = DEFAULT_N_SPECIAL_BODIES;

	if (configJson.contains("dt")) dt = configJson["dt"];
	else dt = DEFAULT_DT;

	if (configJson.contains("specialParticleSize")) specialParticleSize = configJson["specialParticleSize"];
	else specialParticleSize = 1.0f * particleSize;

	Body* drawBodies;
	drawBodies = new Body[DEFAULT_N_BODIES + DEFAULT_N_BODIES];

	createDrawBodies(drawBodies, bodies, numBodies, specialBodies);

	// Main while loop
	while (!glfwWindowShouldClose(window))
	{
		// Specify the color of the background
		glClearColor(0.07f, 0.13f, 0.17f, 1.0f);
		// Clean the back buffer and depth buffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		// Updates and exports the camera matrix to the Vertex Shader
		camera.updateMatrix(45.0f, 0.1f, 100.0f);

		// light color
		glm::vec4 lightColor = glm::vec4(sourceLightColor[0], sourceLightColor[1], sourceLightColor[2], sourceLightColor[3]);
		// Tells OpenGL which Shader Program we want to use
		shaderProgram.Activate();
		// Exports the camera Position to the Fragment Shader for specular lighting
		glUniform3f(glGetUniformLocation(shaderProgram.ID, "camPos"), camera.Position.x, camera.Position.y, camera.Position.z);
		glUniform4f(glGetUniformLocation(shaderProgram.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
		// Export the camMatrix to the Vertex Shader of the pyramid
		camera.Matrix(shaderProgram, "camMatrix");
		// Binds texture so that is appears in rendering
		brickTex.Bind();
		// Bind the VAO so OpenGL knows to use it
		VAO1.Bind();
		// Draw primitives, number of indices, datatype of indices, index of indices
		drawSpheres(bodies, shaderProgram, numBodies, 0);

		specialTex.Bind();
		VAO2.Bind();
		drawSpheres(bodies, shaderProgram, specialBodies, DEFAULT_N_BODIES);

		// Tells OpenGL which Shader Program we want to use
		lightShader.Activate();
		glUniform4f(glGetUniformLocation(lightShader.ID, "lightColor"), lightColor.x, lightColor.y, lightColor.z, lightColor.w);
		// Export the camMatrix to the Vertex Shader of the light cube
		camera.Matrix(lightShader, "camMatrix");
		// Bind the VAO so OpenGL knows to use it
		lightVAO.Bind();
		// Draw primitives, number of indices, datatype of indices, index of indices
		glDrawElements(GL_TRIANGLES, sizeof(lightIndices) / sizeof(int), GL_UNSIGNED_INT, 0);

		// Take care of all GLFW events
		glfwPollEvents();

		// Window Test
		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();
		ImGui::BeginMainMenuBar();
		ImGui::Text("Simulation Status: ");
		std::string showText;
		if (stop) showText = "STOPPED";
		else showText = "RUNNING";
		ImGui::Text(showText.c_str());
		std::string btnText = stop ? "Resume" : "Pause";
		if (ImGui::Button(btnText.c_str())) {
			stop = !stop;
		}
		ImGui::EndMainMenuBar();
		// Start the Dear ImGui frame
		if (showConf) {
			//ImGui::ShowDemoWindow();
			ImGui::Begin("Configuration");
			ImGui::Text("Simulation Engine");
			ImGui::RadioButton("GPU", &useGPU, 1);
			ImGui::SameLine();
			ImGui::RadioButton("CPU", &useGPU, 0);

			ImGui::SliderFloat("Time Step", &dt, DEFAULT_DT, 100.0f);
			ImGui::SliderInt("Bodies", &numBodies, 1, DEFAULT_N_BODIES);
			ImGui::SliderInt("Special Bodies", &specialBodies, 0, DEFAULT_N_BODIES);
			ImGui::SliderFloat("Normal Mass", &massFactor, 1, 1e3f, "%.0f");
			ImGui::SliderFloat("Special Mass", &specialMassFactor, 1, 1e3f, "%.0f");
			ImGui::SliderFloat("Particle Scale", &particleSize, 0.1f, 10.0f, "%.1f");
			ImGui::SliderFloat("Special Particle Scale", &specialParticleSize, 0.1f, 10.0f, "%.1f");
			ImGui::ColorEdit4("Light Color", sourceLightColor);
			if (ImGui::Button("Reset")) {
				generateBodies(bodies, numBodies, specialBodies, read_particles_from_file, particles_conf_file);
				showConf = !showConf;
			}
			ImGui::End();
		}

		createDrawBodies(drawBodies, bodies, numBodies, specialBodies);

		m = MASS_SCALE * massFactor;
		sm = MASS_SCALE * specialMassFactor;

		// update positions
		if (useGPU & !stop) {
			// CUDA interop
			Body* devicePtr;
			size_t size;
			cudaGraphicsMapResources(1, &cudaVBO);
			cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void **>(&devicePtr), &size, cudaVBO);
			simulateNBodyCUDA(drawBodies, kernel_filename.c_str(), local_size, numBodies + specialBodies, dt, &m, &sm);
			// unmap resources
			cudaGraphicsUnmapResources(1, &cudaVBO);
		} else if (!(useGPU || stop)) {
			simulateNBodyCPU(drawBodies, numBodies + specialBodies, dt, &m, &sm);
		}

		for (int i = 0 ; i < numBodies ; ++i) {
			bodies[i] = drawBodies[i];
		}
		for (int i = numBodies ; i < numBodies + specialBodies ; ++i) {
			bodies[DEFAULT_N_BODIES + i - numBodies] = drawBodies[i];
		}

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

		// Swap the back buffer with the front buffer
		glfwSwapBuffers(window);

	}

	// Delete all the objects we've created
	VAO1.Delete();
	VBO1.Delete();
	EBO1.Delete();
	VAO2.Delete();
	VBO2.Delete();
	EBO2.Delete();
	brickTex.Delete();
	specialTex.Delete();
	shaderProgram.Delete();
	lightVAO.Delete();
	lightVBO.Delete();
	lightEBO.Delete();
	lightShader.Delete();
	// Delete window before ending the program
	glfwDestroyWindow(window);
	// Terminate GLFW before ending the program
	glfwTerminate();
	// imgui
	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	return 0;
}
