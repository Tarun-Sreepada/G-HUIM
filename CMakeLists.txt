cmake_minimum_required(VERSION 3.20)
project(cuDPEFIM LANGUAGES CXX CUDA)

# ─── 1) C++ / CUDA standards ───────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD      20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD     20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── 2) Default build type ────────────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# ─── 3) CUDA settings ─────────────────────────────────────────────────────────
# Enable separable compilation & target your GPU archs:
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89 90)

# ─── 4) Includes & source files ───────────────────────────────────────────────
include_directories(
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
  ${CMAKE_SOURCE_DIR}/include
)

# find liburing
find_library(LIBURING_LIB NAMES uring REQUIRED)
find_path  (LIBURING_INCLUDE_DIR NAMES liburing.h REQUIRED)
include_directories(${LIBURING_INCLUDE_DIR})

find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
  set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif()

# find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP enabled")
endif()

file(GLOB_RECURSE SRC_CPP_FILES src/*.cpp)
file(GLOB_RECURSE SRC_CU_FILES  src/*.cu)

add_executable(cuDPEFIM
  ${SRC_CPP_FILES}
  ${SRC_CU_FILES}
)

# ─── 5) Common warning flags ──────────────────────────────────────────────────
set(BASE_CXX_WARN_FLAGS    "-Wall -Wextra -Wpedantic")
set(BASE_CUDA_WARN_FLAGS   "-Xcompiler=-Wall,-Wextra,-Wpedantic")

# ─── 6) Build‐type–specific flags ─────────────────────────────────────────────
if(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Configuring Debug flags")
  # C++
  set(CMAKE_CXX_FLAGS_DEBUG   "${BASE_CXX_WARN_FLAGS} -O0 -g")
  # CUDA
  set(CMAKE_CUDA_FLAGS_DEBUG  "${BASE_CUDA_WARN_FLAGS} -O0 -g -lineinfo")
  target_compile_definitions(cuDPEFIM PRIVATE DEBUG)

else()  # Release or RelWithDebInfo
  message(STATUS "Configuring Release flags")

  # --- C++ release flags (mega-fast) ---
  set(CMAKE_CXX_FLAGS_RELEASE "${BASE_CXX_WARN_FLAGS} -Ofast -g0 -march=native -flto -DNDEBUG")

  # --- CUDA release flags ---
  set(CMAKE_CUDA_FLAGS_RELEASE "${BASE_CUDA_WARN_FLAGS} -O3 --use_fast_math --extra-device-vectorization --expt-relaxed-constexpr --expt-extended-lambda --disable-warnings -lineinfo -Xptxas=-O3,-v,-dlcm=ca -Xptxas=-maxrregcount=64 -Xcompiler=-march=native -Xcompiler=-funroll-loops -Xcompiler=-fno-rtti -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 -gencode arch=compute_90,code=sm_90")
endif()

# ─── 7) Link libraries ────────────────────────────────────────────────────────
target_link_libraries(cuDPEFIM
  PRIVATE
    ${LIBURING_LIB}
    cuda
    cudart
    OpenMP::OpenMP_CXX
)

# ─── 8) Print out final flags ─────────────────────────────────────────────────
message(STATUS "  CXX Release Flags:   ${CMAKE_CXX_FLAGS_RELEASE}")
message(STATUS "  CUDA Release Flags:  ${CMAKE_CUDA_FLAGS_RELEASE}")
message(STATUS "  Linker Flags:        ${CMAKE_EXE_LINKER_FLAGS_RELEASE}")



message(STATUS "--------------------------------------------------------------------")
message(STATUS "cuDPEFIM:")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "To compile, run:")
message(STATUS "  make -j$(nproc)")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "Usage Instructions for cuDPEFIM:")
message(STATUS "--------------------------------------------------------------------")
message(STATUS "To display help:")
message(STATUS "  ./cuDPEFIM -h")
message(STATUS "--------------------------------------------------------------------")
