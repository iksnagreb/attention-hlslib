// EXIT_SUCCESS macro and assert
#include <cstdlib>
#include <cassert>
// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Flatten array of bit-vectors
#include "flatten.hpp"

// Runs the test of the flatten operation
void test_flatten() {
    // Create a buffer of some datatype
    //  Note: Use 4 bit to easily specify numbers in hexadecimal
    ap_uint<4> array[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    // When flattened, each element of the array will be one nibble of the flat
    // bit-vector
    assert(flatten<9>(array) == 0x987654321);
    // The width of the flat bit-vector must be N x the bit-width per element
    assert(decltype(flatten<9>(array))::width == 3);
}

// Program entrypoint
int main(int, char**) {
    // Run the test (might fail via assertion)
    test_flatten();
    // No error, exit with status code "ok"
    return EXIT_SUCCESS;
}