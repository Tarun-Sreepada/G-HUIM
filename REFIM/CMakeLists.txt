# REFIM: the CPU miner (refined EFIM). Builds standalone (cmake -S REFIM) or via the root CMakeLists.txt.
cmake_minimum_required(VERSION 3.24)
project(REFIM LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
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
find_package(Threads REQUIRED)

# Sources: io/ reads the dataset and writes itemsets, mine/ is the search, sys/ holds NUMA,
# allocation and threading helpers.
add_executable(refim
    src/main.cpp
    src/app/options.cpp
    src/app/logging.cpp
    src/app/run.cpp
    src/io/reader.cpp
    src/io/reader/text.cpp
    src/io/reader/name_table.cpp
    src/io/reader/read_all.cpp
    src/io/reader/split_into_chunks.cpp
    src/io/reader/parse_chunks.cpp
    src/io/reader/merge_names.cpp
    src/io/reader/rank_items.cpp
    src/io/reader/rename_transactions.cpp
    src/io/reader/check_utilities.cpp
    src/io/reader/concatenate.cpp
    src/io/reader/sort_transactions.cpp
    src/io/reader/build_root.cpp
    src/io/reader/find_primary.cpp
    src/io/output.cpp
    src/mine/refim.cpp
    src/mine/worker/worker.cpp
    src/mine/worker/expand.cpp
    src/mine/worker/leaf_utility.cpp
    src/mine/worker/project_child.cpp
    src/mine/worker/record_pattern.cpp
    src/mine/worker/select_extensions.cpp
    src/mine/worker/compact_child.cpp
    src/mine/worker/descend.cpp
    src/mine/simd.cpp
    src/sys/alloc.cpp
    src/sys/numa.cpp)
target_include_directories(refim PRIVATE include)
target_link_libraries(refim PRIVATE Threads::Threads CLI11::CLI11 spdlog::spdlog)
target_compile_options(refim PRIVATE -Wall -Wextra $<$<CONFIG:Release>:-O3 -march=native>)

# refim build option: link-time optimisation.
option(REFIM_LTO "Link-time optimisation for refim" ON)
if(CMAKE_BUILD_TYPE STREQUAL "Release" AND REFIM_LTO)
    target_compile_options(refim PRIVATE -flto=auto -fno-fat-lto-objects)
    target_link_options(refim PRIVATE -flto=auto)
endif()
