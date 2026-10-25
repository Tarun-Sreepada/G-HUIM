# GREFIM: REFIM on every CPU core plus every GPU through Vulkan compute. Builds via the root CMakeLists.txt
# (it links REFIM's refim_core) -> build/GREFIM. Needs the Vulkan headers and loader, and glslc to compile the
# shaders to SPIR-V; no CUDA or ROCm.
cmake_minimum_required(VERSION 3.24)
project(GREFIM LANGUAGES C CXX)

if(NOT TARGET refim_core)
    message(FATAL_ERROR "GREFIM builds from the root CMakeLists.txt, after REFIM")
endif()
find_package(Vulkan REQUIRED COMPONENTS glslc)

# Shaders -> SPIR-V -> one generated header.
set(GREFIM_SHADERS expand)
set(GREFIM_SPIRV "")
foreach(shader IN LISTS GREFIM_SHADERS)
    set(src ${CMAKE_CURRENT_SOURCE_DIR}/shaders/${shader}.comp)
    set(spv ${CMAKE_CURRENT_BINARY_DIR}/shaders/${shader}.spv)
    add_custom_command(OUTPUT ${spv}
        COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_BINARY_DIR}/shaders
        COMMAND ${Vulkan_GLSLC_EXECUTABLE} --target-env=vulkan1.2 -O ${src} -o ${spv}
        DEPENDS ${src}
        COMMENT "glslc ${shader}.comp" VERBATIM)
    list(APPEND GREFIM_SPIRV ${spv})
endforeach()
set(GREFIM_SHADER_HEADER ${CMAKE_CURRENT_BINARY_DIR}/generated/shaders.hpp)
string(REPLACE ";" "," GREFIM_SPIRV_ARG "${GREFIM_SPIRV}")
add_custom_command(OUTPUT ${GREFIM_SHADER_HEADER}
    COMMAND ${CMAKE_COMMAND} -DINPUTS=${GREFIM_SPIRV_ARG} -DOUTPUT=${GREFIM_SHADER_HEADER} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    DEPENDS ${GREFIM_SPIRV} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_spirv.cmake
    COMMENT "Embedding SPIR-V" VERBATIM)

# Vulkan structs are set up as {sType, pNext} with the rest zero: -Wno-missing-field-initializers.
set(GREFIM_COMPILE_OPTIONS -Wall -Wextra -Wno-missing-field-initializers $<$<CONFIG:Release>:-O3 -march=native>)

add_executable(GREFIM
    src/main.cpp
    src/vk/context.cpp
    src/gpu/engine.cpp
    src/sched/pool.cpp
    ${GREFIM_SHADER_HEADER})
target_include_directories(GREFIM PRIVATE src ${CMAKE_CURRENT_BINARY_DIR}/generated)
target_link_libraries(GREFIM PRIVATE refim_core Vulkan::Vulkan)
target_compile_options(GREFIM PRIVATE ${GREFIM_COMPILE_OPTIONS})

# The share sees REFIM's databases, so like REFIM's search it is built once per width pair (REFIM_WIDTHS).
foreach(widths IN LISTS REFIM_WIDTHS)
    string(REGEX MATCH "^([0-9]+):([0-9]+)$" _ ${widths})
    set(variant grefim_share_k${CMAKE_MATCH_1}_u${CMAKE_MATCH_2})
    add_library(${variant} OBJECT src/share.cpp)
    target_include_directories(${variant} PRIVATE src)
    target_link_libraries(${variant} PRIVATE refim_core Vulkan::Vulkan)
    target_compile_options(${variant} PRIVATE ${GREFIM_COMPILE_OPTIONS})
    target_compile_definitions(${variant} PRIVATE REFIM_KEY_BITS=${CMAKE_MATCH_1} REFIM_UTIL_BITS=${CMAKE_MATCH_2})
    target_link_libraries(GREFIM PRIVATE ${variant})
endforeach()
