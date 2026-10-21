# Two independent miners, each with its own CMakeLists.txt, include/ and src/:
#   REFIM/   CPU miner  -> build/REFIM
#   G-HUIM/  GPU miner  -> build/G-HUIM
cmake_minimum_required(VERSION 3.24)
project(HUIM LANGUAGES NONE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
include(CTest)
# Binary dirs are suffixed so they do not collide with the executables build/REFIM, build/G-HUIM.
add_subdirectory(REFIM obj-refim)
add_subdirectory(G-HUIM obj-ghuim)
