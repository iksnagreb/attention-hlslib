# Use dynamically linked Boost
set (Boost_USE_STATIC_LIBS OFF)
# Use the unit testing framework of Boost
find_package (Boost REQUIRED COMPONENTS unit_test_framework)
# Add Boost Testing tot he include paths
include_directories (${Boost_INCLUDE_DIRS})

# Add Boost tests main executable collecting  and running all tests
add_executable(run_tests
        tests_main.cpp
        test_flatten.cpp
        test_swizzle.cpp
        test_stream_tiler.cpp
        test_matmul.cpp
        test_attention.cpp)
# Dynamically link the boost libraries to the test executable
target_link_libraries(run_tests ${Boost_LIBRARIES})

# Register the test executable as a runnable target
add_custom_target(test DEPENDS run_tests COMMAND run_tests)
