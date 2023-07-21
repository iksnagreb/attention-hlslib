// EXIT_SUCCESS macro and assert
#include <cstdlib>
#include <cassert>
// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Swizzle bit-vectors SIMD and PE order
#include "swizzle.hpp"

// Runs the test of the swizzle operation
void test_swizzle() {
    // Test vector of 4 groups of 3 elements each
    //  Note: 4 bit per element to easily specify in hexadecimal
    ap_uint<48> test_vector = 0x321321321321;
    // Swizzle of the test vector should result in 3 groups of 4 element each
    assert(
        (swizzle</*per group*/3, /*groups*/4>(test_vector)) == 0x333322221111
    );
    // The inverse of swizzle is swizzle with flipped arguments
    assert(
        (swizzle<4, 3>(swizzle<3, 4>(test_vector))) == test_vector
    );
    // Swizzles preserves the width of the bit-vector
    assert(decltype(swizzle<3, 4>(test_vector))::width == 48);
}

// Program entrypoint
int main(int, char**) {
    // Run the test (might fail via assertion)
    test_swizzle();
    // No error, exit with status code "ok"
    return EXIT_SUCCESS;
}
