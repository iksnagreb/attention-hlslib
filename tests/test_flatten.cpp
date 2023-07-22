// Setup Boost Tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Flatten array of bit-vectors
#include "flatten.hpp"

// Runs the test of the flatten operation
BOOST_AUTO_TEST_CASE(test_flatten) {
    // Create a buffer of some datatype
    //  Note: Use 4 bit to easily specify numbers in hexadecimal
    ap_uint<4> array[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    // When flattened, each element of the array will be one nibble of the flat
    // bit-vector
    BOOST_CHECK(flatten<9>(array) == 0x987654321);
    // The width of the flat bit-vector must be N x the bit-width per element
    BOOST_CHECK(decltype(flatten<9>(array))::width == 36);
}
