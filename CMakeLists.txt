# CMakeLists.txt

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

project(cuDPEFIM
    VERSION 1.0
    LANGUAGES CXX CUDA
)

# Set C++ and CUDA Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Specify CUDA Architecture
set(CMAKE_CUDA_ARCHITECTURES 86)

# Automatically collect all .cpp and .cu files in src/
file(GLOB SRC_FILES "src/*.cpp" "src/*.cu")


# Create executable, including main.cu
add_executable(${PROJECT_NAME} main.cu ${SRC_FILES})

# Include directories for your project and ouroboros
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Add -O3 optimization to both host (CXX) and device (CUDA) code.
target_compile_options(${PROJECT_NAME} PRIVATE
    # Host compiler flags (C++):
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
    # Device compiler flags (CUDA):
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --expt-relaxed-constexpr -Xcompiler=-O3 -Xcompiler=-DVERBOSE -lcudart -lcudadevrt>
)

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)


# Dynamic Memory Manager for GPU
include(cmake/CPM.cmake)
CPMAddPackage(
  NAME gallatin
  GITHUB_REPOSITORY saltsystemslab/gallatin
  GIT_TAG origin/main
)

target_link_libraries(${PROJECT_NAME} PRIVATE gallatin)
