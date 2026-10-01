cmake_minimum_required(VERSION 3.20)
project(GHUIM LANGUAGES CUDA CXX)

# 1. Standards
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 2. Build Type Configuration (default to Release for single-config generators)
if(NOT CMAKE_CONFIGURATION_TYPES)
  set(default_build_type "Release")
  if(NOT CMAKE_BUILD_TYPE)
    message(STATUS "Setting build type to '${default_build_type}' (use -DCMAKE_BUILD_TYPE=<type> to override)")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the build type." FORCE)
  endif()
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
else()
  message(STATUS "Multi-config generator detected. Available configurations: ${CMAKE_CONFIGURATION_TYPES}")
endif()

# 3. CUDA Architecture & Compilation Settings
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Find your architecture: https://developer.nvidia.com/cuda-gpus
# Examples: 86 (Ampere), 89 (Ada Lovelace), 90 (Hopper)
set(CMAKE_CUDA_ARCHITECTURES 90) #

# 4. Find Source Files & Create Executable
include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB_RECURSE SRC_FILES src/*.cpp src/*.cu)
add_executable(GHUIM ${SRC_FILES})
set_target_properties(GHUIM PROPERTIES OUTPUT_NAME "G-HUIM")

# 5. Global Compiler & Linker Optimizations
# set(HOST_WARNING_FLAGS "-Wall -Wextra -Wpedantic")
set(CUDA_WARNING_FLAGS "-Xcompiler=${HOST_WARNING_FLAGS}")
target_compile_options(GHUIM PRIVATE
  $<$<COMPILE_LANGUAGE:CXX>:${HOST_WARNING_FLAGS}>
  $<$<COMPILE_LANGUAGE:CUDA>:${CUDA_WARNING_FLAGS}>
)

# 6. Build-Type Specific Flags (applied via generator expressions)
set(HOST_RELEASE_FLAGS
  -O3
  -DNDEBUG
  -march=native
  -fno-strict-aliasing
  -ffast-math
)

set(CUDA_RELEASE_FLAGS
  -O3
  --use_fast_math
  --extra-device-vectorization
  --expt-extended-lambda
  --expt-relaxed-constexpr
  -lineinfo
  -Xptxas=-O3,-v,-dlcm=ca
  -Xcompiler=-O3,-march=native,-fno-rtti
)

set(HOST_DEBUG_FLAGS -O0 -g)
set(CUDA_DEBUG_FLAGS -O0 -g -G)

target_compile_definitions(GHUIM PRIVATE
  $<$<CONFIG:Debug>:DEBUG>
)

target_compile_options(GHUIM PRIVATE
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CXX>>:${HOST_RELEASE_FLAGS}>
  $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:${CUDA_RELEASE_FLAGS}>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CXX>>:${HOST_DEBUG_FLAGS}>
  $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:${CUDA_DEBUG_FLAGS}>
)

# 7. Link Libraries
target_link_libraries(GHUIM PRIVATE cuda)

# 8. Final Output (Cleaned for clarity)
message(STATUS "--------------------------------------------------------------------")
message(STATUS "Build Configuration for G-HUIM:")
message(STATUS "  - Target GPU Arch: ${CMAKE_CUDA_ARCHITECTURES}")
if(CMAKE_CONFIGURATION_TYPES)
  message(STATUS "  - Configurations:  ${CMAKE_CONFIGURATION_TYPES}")
else()
  message(STATUS "  - Build Type:      ${CMAKE_BUILD_TYPE}")
endif()
message(STATUS "--------------------------------------------------------------------")