// Setup Boost Tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Swizzle bit-vectors SIMD and PE order
#include "swizzle.hpp"

// Runs the test of the swizzle operation
BOOST_AUTO_TEST_CASE(test_swizzle) {
    // Test vector of 4 groups of 3 elements each
    //  Note: 4 bit per element to easily specify in hexadecimal
    ap_uint<48> test_vector = 0x321321321321;
    // Swizzle of the test vector should result in 3 groups of 4 element each
    BOOST_CHECK(
        (swizzle</*per group*/3, /*groups*/4>(test_vector)) == 0x333322221111
    );
    // The inverse of swizzle is swizzle with flipped arguments
    BOOST_CHECK(
        (swizzle<4, 3>(swizzle<3, 4>(test_vector))) == test_vector
    );
    // Swizzles preserves the width of the bit-vector
    BOOST_CHECK(decltype(swizzle<3, 4>(test_vector))::width == 48);
}
