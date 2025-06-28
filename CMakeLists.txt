cmake_minimum_required(VERSION 3.31)
project(CC7515_T3)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories("Libraries/include")
#include_directories("include")

# Add executable with all .cpp and .c files
add_executable(CC7515_T3
        "src/Camera.cpp"
        "src/EBO.cpp"
        "src/glad.c"
        "src/Main.cpp"
        "src/shaderClass.cpp"
        "src/stb.cpp"
        "src/Texture.cpp"
        "src/VAO.cpp"
        "src/VBO.cpp"
)

# Link GLFW
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

target_link_libraries(CC7515_T3 PRIVATE glfw OpenGL::GL)
#target_link_directories(CC7515_T3 PRIVATE "include")