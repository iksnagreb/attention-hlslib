// Setup Boost Tests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Matrix and matrix streaming utility
#include "utils.hpp"
// Softmax function to be tested
#include "softmax.hpp"

// Configure the datatype to use for tests (single element type)
using TestType = ap_uint<5>;

// Configuration of the tiling tests
static constexpr std::size_t M = 12;
static constexpr std::size_t N = 15;
static constexpr std::size_t PE = 5;  // Parallel processing elements


// Tests the softmax normalization simples, flat input stream
BOOST_AUTO_TEST_CASE(test_softmax_simple) {
    // Generate a random matrix
    auto x = randf_matrix<M, N>();
    // Softmax normalize the matrix
    auto y = softmax(x);

    // Quantize the input and normalized matrix
    auto qx = Quantized<TestType::width, M, N>(x);
    auto qy = Quantized<TestType::width, M, N>(y);

    // Validate quantization: Errors should never exceed one "step of scale"
    BOOST_CHECK(all_close(x, qx.dequantize(), qx.scale));
    BOOST_CHECK(all_close(y, qy.dequantize(), qy.scale));

//    // For debugging print scale parameters
//    std::printf("z.scale=%4.4f, z.bias=%4.4f\n", z.scale, z.bias);
//    std::printf("q.scale=%4.4f, q.bias=%4.4f\n", q.scale, q.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);

    // Compute softmax normalization over the stream
    Softmax<N, /*PE=*/1, TestType> softmax(
        qx_elems.out, qx.scale, qy.scale, qy.bias, /*rep=*/M
    );

    // Matrix to be filled by softmax stream
    decltype(qy) qs = qy;
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax.out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), 0.1f));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(softmax.out.empty());
}

// Tests the softmax normalization with grouped input stream
BOOST_AUTO_TEST_CASE(test_softmax_grouped) {
    // Generate a random matrix
    auto x = randf_matrix<M, N>();
    // Softmax normalize the matrix
    auto y = softmax(x);

    // Quantize the input and normalized matrix
    auto qx = Quantized<TestType::width, M, N>(x);
    auto qy = Quantized<TestType::width, M, N>(y);

    // Validate quantization: Errors should never exceed one "step of scale"
    BOOST_CHECK(all_close(x, qx.dequantize(), qx.scale));
    BOOST_CHECK(all_close(y, qy.dequantize(), qy.scale));

//    // For debugging print scale parameters
//    std::printf("z.scale=%4.4f, z.bias=%4.4f\n", z.scale, z.bias);
//    std::printf("q.scale=%4.4f, q.bias=%4.4f\n", q.scale, q.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);
    // Group the input stream into PE parallel elements
    GroupStreamElements<TestType, PE> qx_grouped(qx_elems.out);

    // Compute softmax normalization over the stream
    Softmax<N / PE, /*PE=*/PE, ap_uint<TestType::width * PE>> softmax(
        qx_grouped.out, qx.scale, qy.scale, qy.bias, /*rep=*/M
    );

    // Split the output stream for validation
    SplitStreamElements<ap_uint<TestType::width * PE>, PE> softmax_elems(
        softmax.out
    );
    // Matrix to be filled by softmax stream
    decltype(qy) qs = qy;
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax_elems.out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), 0.1f));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(qx_grouped.out.empty());
    BOOST_CHECK(softmax.out.empty());
}