cmake_minimum_required(VERSION 3.31)
project(cc7515_t3)

set(CMAKE_CXX_STANDARD 14)

include_directories(src/include)


add_executable(cc7515_t3
        src/main.cpp)

find_package(glfw3 3.3 REQUIRED)
target_link_libraries(cc7515_t3 glfw)

find_package(OpenGL REQUIRED)
target_link_libraries(cc7515_t3 OpenGL::GL)
