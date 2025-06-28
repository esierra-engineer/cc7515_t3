cmake_minimum_required(VERSION 3.31)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_ARCHITECTURES native)
project(CC7515_T3 LANGUAGES CXX CUDA C)

set(CMAKE_CXX_STANDARD 17)

# Include directories
include_directories("include")
include_directories("Libraries/include")
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

# Link GLFW
find_package(glfw3 REQUIRED)
find_package(OpenGL REQUIRED)

# Kernel PTX generation
set(KERNEL_SRCS
        ${CMAKE_SOURCE_DIR}/src/kernel_1_global-memory_1D.cu
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
        "src/utils.cpp"
        "src/nbody_cuda.cu"
)

target_link_libraries(CC7515_T3 PRIVATE glfw OpenGL::GL cuda)
target_link_directories(CC7515_T3 PRIVATE "include")

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