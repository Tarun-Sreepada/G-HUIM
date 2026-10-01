cmake_minimum_required(VERSION 3.20)

# Minimum supported GPU: RTX 3000 series (Ampere, sm_86). Kernels use sm_80+ warp intrinsics;
# the embedded compute_86 PTX is JIT-compiled for newer GPUs.
set(CMAKE_CUDA_ARCHITECTURES 90)
project(GHUIM LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

include(FetchContent)
# Pin dependency commits for reproducible builds (CLI11 2.6.2, spdlog 1.16.0).
FetchContent_Declare(CLI11
    GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
    GIT_TAG 37bb6edc5317e99af72ef48405e65d9ca5218861)
FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog.git
    GIT_TAG 486b55554f11c9cccc913e11a87085b2a91f706f)
FetchContent_MakeAvailable(CLI11 spdlog)
find_package(CUDAToolkit REQUIRED)

add_library(ghuim_app STATIC
    src/app/config.cpp
    src/app/cli.cpp
    src/app/logging.cpp
    src/data/parse.cpp
    src/data/dataset.cpp)
target_include_directories(ghuim_app PUBLIC include)
target_link_libraries(ghuim_app PUBLIC CLI11::CLI11 spdlog::spdlog)
target_compile_options(ghuim_app PRIVATE -Wall -Wextra -Wpedantic)

add_library(ghuim_core STATIC
    src/mine/miner.cpp
    src/mine/output.cpp
    src/mine/primitives.cu
    src/mine/bfs/small_kernels.cu
    src/mine/bfs/huge_kernels.cu
    src/mine/bfs/schedule.cu
    src/mine/bfs/occupancy.cu
    src/mine/bfs/root_image.cu
    src/mine/bfs/bfs_miner.cu)
target_include_directories(ghuim_core PUBLIC include)
target_link_libraries(ghuim_core PUBLIC CUDA::cudart spdlog::spdlog)
target_compile_options(ghuim_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)
target_compile_definitions(ghuim_core PRIVATE $<$<CONFIG:Debug>:DEBUG>)

add_executable(GHUIM src/main.cpp)
set_target_properties(GHUIM PROPERTIES OUTPUT_NAME "G-HUIM")
target_link_libraries(GHUIM PRIVATE ghuim_app ghuim_core)
set_target_properties(GHUIM PROPERTIES LINKER_LANGUAGE CUDA)

# CPU EFIM baseline.
find_package(Threads REQUIRED)
add_executable(efim efim/main.cpp efim/efim.cpp efim/reader.cpp efim/worker.cpp)
target_link_libraries(efim PRIVATE Threads::Threads)
target_compile_options(efim PRIVATE -Wall -Wextra $<$<CONFIG:Release>:-O3 -march=native>)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cli_help_without_gpu COMMAND GHUIM --help)
    set_tests_properties(cli_help_without_gpu PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
endif()
