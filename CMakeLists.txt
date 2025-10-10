cmake_minimum_required(VERSION 3.20)
project(cuDPEFIM LANGUAGES CUDA CXX)

# 1. Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. Build Type Configuration (Default to max-optimized Release)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# 3. CUDA Architecture & Compilation Settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# --- LTO DISABLED ---
# Device Link-Time Optimization is disabled to prevent toolchain conflicts.
# set(CMAKE_CUDA_DEVICE_LINK_TIME_OPTIMIZATION ON) # <--- REMOVED

# IMPORTANT: Target your specific GPU architecture.
# Find your architecture: https://developer.nvidia.com/cuda-gpus
# Examples: 86 (Ampere), 89 (Ada Lovelace), 90 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES 86) # <--- CHANGE THIS to your GPU's compute capability

# 4. Find Source Files & Create Executable
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
add_executable(cuDPEFIM ${SRC_FILES})

# 5. Global Compiler & Linker Optimizations
# set(HOST_WARNING_FLAGS "-Wall -Wextra -Wpedantic")
set(CUDA_WARNING_FLAGS "-Xcompiler=${HOST_WARNING_FLAGS}")
target_compile_options(cuDPEFIM PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${HOST_WARNING_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_WARNING_FLAGS}>
)

# 6. Build-Type Specific Flags for MAXIMUM PERFORMANCE
if(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Configuring for Release")

  # --- Host (CPU) Compiler Optimizations ---
  set(HOST_OPTIMIZATION_FLAGS
    "-O3"                            # Highest level of optimization
    "-DNDEBUG"                       # Disable assertions and debug code
    "-march=native"                  # Generate instructions for the current CPU
    # "-flto"                          # <--- REMOVED to fix build error
    "-fno-strict-aliasing"           # Less strict aliasing rules, can improve performance
    "-ffast-math"                    # Aggressive floating-point optimizations
  )

  # --- Device (GPU) Compiler Optimizations ---
  set(CUDA_OPTIMIZATION_FLAGS
    "-O3"                            # Highest optimization level for NVCC
    "--use_fast_math"
    "--extra-device-vectorization"
    "--expt-extended-lambda"
    "--expt-relaxed-constexpr"
    "-lineinfo"                      # Add line info for profilers (e.g., Nsight)
    "-Xptxas=-O3,-v,-dlcm=ca"         # Max PTXAS optimization, verbose, L1 cache
    # Host compiler flags passed through NVCC (removed -flto)
    "-Xcompiler=-O3,-march=native,-fno-rtti"
  )

  target_compile_options(cuDPEFIM PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${HOST_OPTIMIZATION_FLAGS}>
    $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_OPTIMIZATION_FLAGS}>
  )

elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Configuring for Debugging")
  target_compile_definitions(cuDPEFIM PRIVATE DEBUG)
  target_compile_options(cuDPEFIM PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O0 -g>
    $<$<COMPILE_LANGUAGE:CUDA>:-O0 -g -G>
  )
endif()

# 7. Link Libraries
target_link_libraries(cuDPEFIM PRIVATE cuda)

# 8. Final Output (Cleaned for clarity)
message(STATUS "--------------------------------------------------------------------")
message(STATUS "Build Configuration for cuDPEFIM:")
message(STATUS "  - Target GPU Arch: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "  - Build Type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "--------------------------------------------------------------------")