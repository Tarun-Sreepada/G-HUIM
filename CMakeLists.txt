cmake_minimum_required(VERSION 3.20)
project(cuDPEFIM LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type to Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89;90") # Adjust according to your GPU architecture

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES} ${CMAKE_SOURCE_DIR}/include)

# Find liburing
find_library(LIBURING_LIB NAMES uring REQUIRED)
find_path(LIBURING_INCLUDE_DIR NAMES liburing.h REQUIRED)
include_directories(${LIBURING_INCLUDE_DIR})

# Find OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found")
endif()

file(GLOB_RECURSE SRC_CPP_FILES src/*.cpp)
file(GLOB_RECURSE SRC_CU_FILES src/*.cu)

add_executable(cuDPEFIM ${SRC_CPP_FILES} ${SRC_CU_FILES})

# Set target properties
set_target_properties(cuDPEFIM PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION ON
)

# Common flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# Debug flags
if(CMAKE_BUILD_TYPE MATCHES "[Dd]ebug")
    message(STATUS "Building in Debug mode")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -g -Wall -Wextra -Wpedantic")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -g -lineinfo")
    target_compile_definitions(cuDPEFIM PRIVATE DEBUG)
else()
    message(STATUS "Building in Release mode (optimized)")
    # Native arch tuning, LTO, fast-math, aggressive inlining, no debug
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -flto -funroll-loops -DNDEBUG")
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 -use_fast_math -lineinfo -Xcompiler=-march=native")
    
    # Optional: make sure LTO is supported
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if(result)
        set_property(TARGET cuDPEFIM PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "LTO not supported: ${output}")
    endif()
endif()


# Link necessary libraries
target_link_libraries(cuDPEFIM PRIVATE ${LIBURING_LIB} cuda cudart OpenMP::OpenMP_CXX) # Link liburing, CUDA, and OpenMP



message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CXX Flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "CUDA Flags: ${CMAKE_CUDA_FLAGS}")


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
message(STATUS "  ./cuDPEFIM --help")
message(STATUS "--------------------------------------------------------------------")
