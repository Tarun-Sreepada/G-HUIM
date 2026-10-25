# Three miners, each with its own CMakeLists.txt:
#   REFIM/   CPU miner                       -> build/REFIM
#   G-HUIM/  GPU miner (CUDA)                -> build/G-HUIM
#   GREFIM/  every CPU core + GPU (Vulkan)   -> build/GREFIM   (links REFIM's library)
cmake_minimum_required(VERSION 3.24)
project(HUIM LANGUAGES NONE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
include(CTest)
# -DHUIM_BUILD_<MINER>=OFF leaves a miner out (only G-HUIM needs CUDA, only GREFIM Vulkan).
option(HUIM_BUILD_REFIM "Build the REFIM CPU miner" ON)
option(HUIM_BUILD_GHUIM "Build the G-HUIM GPU miner" ON)
option(HUIM_BUILD_GREFIM "Build the GREFIM CPU + GPU miner" ON)
if(NOT HUIM_BUILD_REFIM AND NOT HUIM_BUILD_GHUIM AND NOT HUIM_BUILD_GREFIM)
    message(FATAL_ERROR "HUIM_BUILD_REFIM, HUIM_BUILD_GHUIM and HUIM_BUILD_GREFIM are all OFF: nothing to build")
endif()

# Binary dirs are suffixed so they do not collide with the executables build/REFIM, build/G-HUIM.
if(HUIM_BUILD_REFIM OR HUIM_BUILD_GREFIM)
    add_subdirectory(REFIM obj-refim)  # GREFIM links its library
    if(NOT HUIM_BUILD_REFIM)
        set_target_properties(REFIM PROPERTIES EXCLUDE_FROM_ALL ON)
    endif()
endif()
if(HUIM_BUILD_GHUIM)
    add_subdirectory(G-HUIM obj-ghuim)
endif()
if(HUIM_BUILD_GREFIM)
    add_subdirectory(GREFIM obj-grefim)
endif()
