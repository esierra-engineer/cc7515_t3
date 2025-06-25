cmake_minimum_required(VERSION 3.31)
project(cc7515_t3)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/include)
include_directories(resources)


add_executable(cc7515_t3
        src/main.cpp
        src/gl.c
        src/include/glad/glad.h
)

add_executable(ref src/ref.cpp)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(cc7515_t3 glfw)
target_link_libraries(ref PRIVATE glfw)

find_package(glm CONFIG REQUIRED)
target_link_libraries(ref PRIVATE glm::glm)

find_package(OpenGL REQUIRED)
target_link_libraries(cc7515_t3 OpenGL::GL)
