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
set(CMAKE_CUDA_ARCHITECTURES 75)

# Automatically collect all .cpp and .cu files in src/
file(GLOB SRC_FILES "src/*.cpp" "src/*.cu")

# Create executable, including main.cu
add_executable(${PROJECT_NAME} main.cu ${SRC_FILES})

# Include directories for your project and dependencies
target_include_directories(${PROJECT_NAME} PRIVATE
    ${PROJECT_SOURCE_DIR}/include
)

# Set Debug and Release specific flags
target_compile_options(${PROJECT_NAME} PRIVATE
    # Host compiler flags (C++):
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:-O0 -g>  # Debug: No optimization, include debug symbols
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:-O3>    # Release: High optimization
    
    # Device compiler flags (CUDA):
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-O0 -G -g --expt-relaxed-constexpr -Xcompiler=-g>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3 --expt-relaxed-constexpr>
)

# Define DEBUG macro for Debug builds
target_compile_definitions(${PROJECT_NAME} PRIVATE 
    $<$<CONFIG:Debug>:DEBUG>
)

# Set CUDA properties
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# Ensure debugging info is available
set_target_properties(${PROJECT_NAME} PROPERTIES
    CUDA_DEBUG_INFO ON
)

# Link CUDA cufile library
find_library(CUFILE_LIBRARY cufile PATHS /usr/local/cuda-12.6/lib64)
target_link_libraries(${PROJECT_NAME} PRIVATE ${CUFILE_LIBRARY})

# Include CPM for dependency management
include(cmake/CPM.cmake)
CPMAddPackage(
  NAME gallatin
  GITHUB_REPOSITORY saltsystemslab/gallatin
  GIT_TAG origin/main
)

# Link external library gallatin
target_link_libraries(${PROJECT_NAME} PRIVATE gallatin)
