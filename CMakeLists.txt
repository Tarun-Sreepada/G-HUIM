# Two independent miners, each with its own CMakeLists.txt, include/ and src/:
#   REFIM/   CPU miner  -> build/REFIM
#   G-HUIM/  GPU miner  -> build/G-HUIM
cmake_minimum_required(VERSION 3.24)
project(HUIM LANGUAGES NONE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
include(CTest)
# -DHUIM_BUILD_REFIM=OFF or -DHUIM_BUILD_GHUIM=OFF builds only the other miner (REFIM alone needs no CUDA).
option(HUIM_BUILD_REFIM "Build the REFIM CPU miner" ON)
option(HUIM_BUILD_GHUIM "Build the G-HUIM GPU miner" ON)
if(NOT HUIM_BUILD_REFIM AND NOT HUIM_BUILD_GHUIM)
    message(FATAL_ERROR "HUIM_BUILD_REFIM and HUIM_BUILD_GHUIM are both OFF: nothing to build")
endif()

# Binary dirs are suffixed so they do not collide with the executables build/REFIM, build/G-HUIM.
if(HUIM_BUILD_REFIM)
    add_subdirectory(REFIM obj-refim)
endif()
if(HUIM_BUILD_GHUIM)
    add_subdirectory(G-HUIM obj-ghuim)
endif()
