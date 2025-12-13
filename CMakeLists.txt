# The minimum CMake version with sufficient support for list transformations
cmake_minimum_required(VERSION 3.27)

# Make the FetchContent module available for adding dependencies
include(FetchContent)

# Name of the project
project(attention-hlslib)

# Declare the Vitis HLS Integration as dependency to be populated via
# FetchContent pulling from GitHub
FetchContent_Declare(
    # This is the project name and the name of the dependency
    cmake-hls
    # Pull from the GitHub repository
    GIT_REPOSITORY  https://github.com/iksnagreb/cmake-hls.git
    # Pin a specific git commit hash, should be more stable than pinning a tag
    GIT_TAG         35b738f9dc3e8125b2db5b9d19984632198d1e14
    # Only include the CMake scripts/modules, without this the demo project will
    # be registered as build targets as well
    SOURCE_SUBDIR   cmake
)

# Make the declared dependencies available
FetchContent_MakeAvailable(cmake-hls)

# Test for the necessary version of the CMake HLS integration
cmake_hls_minimum_required(VERSION 0.0.0.0)

# Use C++14 standard (should be the one supported by Vitis HLS)
set(CMAKE_CXX_STANDARD 14)

# Depends on FINN HLS library
include_directories($ENV{FINN_HLSLIB})
# Local attention library
include_directories(.)

# Enable constexpr in Vitis library
add_compile_definitions(HLS_CONSTEXPR_ENABLE)
