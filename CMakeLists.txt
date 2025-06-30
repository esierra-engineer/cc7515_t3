cmake_minimum_required(VERSION 3.25)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES native)
project(CC7515_T3 LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link GLFW
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# Kernel PTX generation
set(KERNEL_SRCS
        ${CMAKE_SOURCE_DIR}/src/kernel.cu
)

set(PTX_FILES "")
foreach(kernel ${KERNEL_SRCS})
    get_filename_component(kernel_name ${kernel} NAME_WE)
    set(ptx_file ${CMAKE_BINARY_DIR}/${kernel_name}.ptx)

    add_custom_command(
            OUTPUT ${ptx_file}
            COMMAND ${CMAKE_CUDA_COMPILER} --ptx ${kernel} -o ${ptx_file}
            DEPENDS ${kernel}
            COMMENT "Compiling PTX from ${kernel}"
    )

    list(APPEND PTX_FILES ${ptx_file})
endforeach()

add_custom_target(compile_ptx ALL DEPENDS ${PTX_FILES})

set(IMGUI_SRC
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/imgui.cpp
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/imgui_draw.cpp
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/imgui_tables.cpp
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/imgui_widgets.cpp
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/imgui_demo.cpp
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/backends/imgui_impl_glfw.cpp
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/backends/imgui_impl_opengl3.cpp
)


# Add executable with all .cpp and .c files
set(SOURCES
        ${CMAKE_SOURCE_DIR}/src/Camera.cpp
        ${CMAKE_SOURCE_DIR}/src/EBO.cpp
        ${CMAKE_SOURCE_DIR}/src/glad.c
        ${CMAKE_SOURCE_DIR}/src/Main.cpp
        ${CMAKE_SOURCE_DIR}/src/shaderClass.cpp
        ${CMAKE_SOURCE_DIR}/src/stb.cpp
        ${CMAKE_SOURCE_DIR}/src/Texture.cpp
        ${CMAKE_SOURCE_DIR}/src/VAO.cpp
        ${CMAKE_SOURCE_DIR}/src/VBO.cpp
        ${CMAKE_SOURCE_DIR}/src/utils.cpp
        ${CMAKE_SOURCE_DIR}/src/nbody_cuda.cu
)

add_executable(CC7515_T3
        ${SOURCES}
        ${IMGUI_SRC}
)

target_link_libraries(CC7515_T3 PRIVATE glfw OpenGL::GL cuda)

target_link_directories(CC7515_T3 PRIVATE
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui
)

target_include_directories(CC7515_T3 PRIVATE
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui
        ${CMAKE_SOURCE_DIR}/Libraries/include/imgui/backends
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/Libraries/include
)

# CUDA-specific target properties
set_target_properties(CC7515_T3 PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        CUDA_ARCHITECTURES 52 60 70 75 86
)

# Compiler options
target_compile_options(CC7515_T3 PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
        $<$<COMPILE_LANGUAGE:CXX>:-O2>
)

# Ensure PTX is built before main executable
add_dependencies(CC7515_T3 compile_ptx)

# Copy shaders to build directory
file(COPY ${CMAKE_SOURCE_DIR}/src/shaders
        DESTINATION ${CMAKE_BINARY_DIR})

# Copy texture to build directory
file(COPY ${CMAKE_SOURCE_DIR}/resources
        DESTINATION ${CMAKE_BINARY_DIR})