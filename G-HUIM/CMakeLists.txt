# G-HUIM: the GPU miner. Builds standalone (cmake -S G-HUIM) or via the root CMakeLists.txt.
cmake_minimum_required(VERSION 3.24)

# Build for the GPU(s) in this machine unless -DCMAKE_CUDA_ARCHITECTURES=... says otherwise.
# Kernels use sm_80+ warp intrinsics (Ampere or newer).
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CUDAARCHS})
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()
project(GHUIM LANGUAGES C CXX CUDA)
message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
foreach(arch IN LISTS CMAKE_CUDA_ARCHITECTURES_NATIVE)
    if(arch MATCHES "^([0-9]+)" AND CMAKE_MATCH_1 LESS 80)
        message(FATAL_ERROR "Detected GPU architecture ${arch}; the kernels need sm_80 or newer")
    endif()
endforeach()

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
if(NOT TARGET CLI11::CLI11)
    FetchContent_Declare(CLI11
        GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
        GIT_TAG 37bb6edc5317e99af72ef48405e65d9ca5218861)
    FetchContent_MakeAvailable(CLI11)
endif()
if(NOT TARGET spdlog::spdlog)
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG 486b55554f11c9cccc913e11a87085b2a91f706f)
    FetchContent_MakeAvailable(spdlog)
endif()
find_package(CUDAToolkit REQUIRED)

add_library(ghuim_app STATIC
    src/app/config.cpp
    src/app/cli.cpp
    src/app/logging.cpp
    src/data/parse.cpp
    src/data/dataset.cpp)
target_include_directories(ghuim_app PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ghuim_app PUBLIC CLI11::CLI11 spdlog::spdlog)
target_compile_options(ghuim_app PRIVATE -Wall -Wextra -Wpedantic)

add_library(ghuim_core STATIC
    src/mine/miner.cpp
    src/mine/output.cpp
    src/mine/primitives.cu
    src/mine/bfs/small_kernels.cu
    src/mine/bfs/vertical_kernels.cu
    src/mine/bfs/huge_kernels.cu
    src/mine/bfs/schedule.cu
    src/mine/bfs/occupancy.cu
    src/mine/bfs/root_image.cu
    src/mine/bfs/bfs_miner.cu
    src/mine/bfs/host/auto_tune.cu
    src/mine/bfs/host/budget.cu
    src/mine/bfs/host/itemsets.cu
    src/mine/bfs/host/level_miner/allocate.cu
    src/mine/bfs/host/level_miner/seed_first_level.cu
    src/mine/bfs/host/level_miner/reset_control.cu
    src/mine/bfs/host/level_miner/plan_chunks.cu
    src/mine/bfs/host/level_miner/mine_chunk.cu
    src/mine/bfs/host/level_miner/mine_level.cu
    src/mine/bfs/host/level_miner/run.cu
    src/mine/dfs/schedule.cu
    src/mine/dfs/baseline/kernels.cu
    src/mine/dfs/optimized/small.cu
    src/mine/dfs/optimized/huge.cu
    src/mine/dfs/optimized/occupancy.cu
    src/mine/dfs/bins.cu
    src/mine/dfs/root_image.cu
    src/mine/dfs/arena_gc.cu
    src/mine/dfs/dfs_miner.cu)
target_include_directories(ghuim_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(ghuim_core PUBLIC CUDA::cudart spdlog::spdlog)
# -lineinfo keeps source lines in optimized SASS, so Nsight Compute can attribute stalls to code.
# -DGHUIM_PROFILE=ON also prints registers, spills and shared memory per kernel at build time.
option(GHUIM_PROFILE "Print per-kernel resource usage while compiling" OFF)
target_compile_options(ghuim_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
    $<$<AND:$<BOOL:${GHUIM_PROFILE}>,$<COMPILE_LANGUAGE:CUDA>>:--resource-usage>
    $<$<AND:$<CONFIG:Release>,$<COMPILE_LANGUAGE:CUDA>>:-O3>
    $<$<AND:$<CONFIG:Debug>,$<COMPILE_LANGUAGE:CUDA>>:-G>)
target_compile_definitions(ghuim_core PRIVATE $<$<CONFIG:Debug>:DEBUG>)

add_executable(GHUIM
    src/main.cpp
    src/app/steps/timing.cpp
    src/app/steps/parse_command_line.cpp
    src/app/steps/start_cuda_context.cpp
    src/app/steps/load_data.cpp
    src/app/steps/wait_for_context.cpp
    src/app/steps/mine_data.cpp
    src/app/steps/report_totals.cpp)
set_target_properties(GHUIM PROPERTIES OUTPUT_NAME "G-HUIM")
target_link_libraries(GHUIM PRIVATE ghuim_app ghuim_core)
set_target_properties(GHUIM PROPERTIES LINKER_LANGUAGE CUDA)

include(CTest)
if(BUILD_TESTING)
    add_test(NAME cli_help_without_gpu COMMAND GHUIM --help)
    set_tests_properties(cli_help_without_gpu PROPERTIES ENVIRONMENT "CUDA_VISIBLE_DEVICES=")
endif()
