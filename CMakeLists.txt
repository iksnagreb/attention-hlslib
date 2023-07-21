# This is my local version, prior version might do as well...
cmake_minimum_required(VERSION 3.22)

# Name of the project
project(attention-hlslib)
# Use C++14 standard (should be the one supported by Vitis HLS)
set(CMAKE_CXX_STANDARD 14)

# Depends of Vitis HLS library
include_directories($ENV{VITIS_HLS_PATH}/include)
# Depends on FINN HLS library
include_directories($ENV{FINN_HLSLIB})
# Local attention library
include_directories(.)

# Use CTest for testing
include(CTest)

# Tests the flatten operation
add_executable(test_flatten tests/test_flatten.cpp flatten.hpp)
add_test(NAME test_flatten COMMAND test_flatten)

# Tests the swizzle operation
add_executable(test_swizzle tests/test_swizzle.cpp swizzle.hpp)
add_test(NAME test_swizzle COMMAND test_swizzle)
