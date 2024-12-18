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


// Tests the softmax normalization simple, flat input stream
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
//    std::printf("qx.scale=%4.4f, qx.bias=%4.4f\n", qx.scale, qx.bias);
//    std::printf("qy.scale=%4.4f, qy.bias=%4.4f\n", qy.scale, qy.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);

    // Scale factor for converting integer to float for computing the softmax
    // and converting back to the same integer range
    float scale = 1.0f / (
        (ap_uint<TestType::width + 1>{1} << TestType::width) - 1
    );

    // Instantiate a softmax normalization function with quantization scales
    // inferred above
    Softmax<N, /*PE=*/1, TestType, TestType, QuantActivation<TestType>, M>
        softmax{QuantActivation<TestType>{scale, 0.0}, qx.scale};

    // Normalize all rows of the output matrix
    decltype(softmax)::OStream softmax_out;
    softmax(qx_elems.out, softmax_out);

    // Matrix to be filled by softmax stream
    decltype(qy) qs{{}, scale, 0.0};
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax_out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), scale));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(softmax_out.empty());
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
//    std::printf("qx.scale=%4.4f, qx.bias=%4.4f\n", qx.scale, qx.bias);
//    std::printf("qy.scale=%4.4f, qy.bias=%4.4f\n", qy.scale, qy.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);
    // Group the input stream into PE parallel elements
    GroupStreamElements<TestType, PE> qx_grouped(qx_elems.out);

    // Scale factor for converting integer to float for computing the softmax
    // and converting back to the same integer range
    float scale = 1.0f / (
        (ap_uint<TestType::width + 1>{1} << TestType::width) - 1
    );

    // Instantiate a softmax normalization function with quantization scales
    // inferred above
    Softmax<N / PE, /*PE=*/PE, TestType, TestType, QuantActivation<TestType>, M>
        softmax{QuantActivation<TestType>{scale, 0.0}, qx.scale};

    // Normalize all rows of the output matrix
    decltype(softmax)::OStream softmax_out;
    softmax(qx_grouped.out, softmax_out);

    // Split the output stream for validation
    SplitStreamElements<ap_uint<TestType::width * PE>, PE> softmax_elems(
        softmax_out
    );
    // Matrix to be filled by softmax stream
    decltype(qy) qs{{}, scale, 0.0};
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax_elems.out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), scale));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(qx_grouped.out.empty());
    BOOST_CHECK(softmax_out.empty());
}

// Tests the softmax normalization simple, flat input stream with causal
// attention mask applied
BOOST_AUTO_TEST_CASE(test_softmax_causal_simple) {
    // Generate a random matrix
    auto x = randf_matrix<M, N>();
    // Generate a causal mask
    auto m = causal_mask<M, N>();
    // Softmax normalize the matrix
    auto y = softmax(x + m);

    // Quantize the input and normalized matrix
    auto qx = Quantized<TestType::width, M, N>(x);
    auto qy = Quantized<TestType::width, M, N>(y);

    // Validate quantization: Errors should never exceed one "step of scale"
    BOOST_CHECK(all_close(x, qx.dequantize(), qx.scale));
    BOOST_CHECK(all_close(y, qy.dequantize(), qy.scale));

//    // For debugging print scale parameters
//    std::printf("qx.scale=%4.4f, qx.bias=%4.4f\n", qx.scale, qx.bias);
//    std::printf("qy.scale=%4.4f, qy.bias=%4.4f\n", qy.scale, qy.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);

    // Scale factor for converting integer to float for computing the softmax
    // and converting back to the same integer range
    float scale = 1.0f / (
        (ap_uint<TestType::width + 1>{1} << TestType::width) - 1
    );

    // Instantiate a softmax normalization function with quantization scales
    // inferred above
    Softmax<N, /*PE=*/1, TestType, TestType, QuantActivation<TestType>, M>
        softmax{QuantActivation<TestType>{scale, 0.0}, qx.scale};

    // Normalize all rows of the output matrix
    decltype(softmax)::OStream softmax_out;
    softmax(qx_elems.out, softmax_out, attention::mask::CAUSAL);

    // Matrix to be filled by softmax stream
    decltype(qy) qs{{}, scale, 0.0};
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax_out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), scale));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(softmax_out.empty());
}

// Tests the softmax normalization simple, flat input stream with random
// attention mask applied
BOOST_AUTO_TEST_CASE(test_softmax_masked_simple) {
    // Generate a random matrix
    auto x = randf_matrix<M, N>();
    // Generate a random attention mask
    auto m = randf_mask<M, N>();
    // Softmax normalize the matrix
    auto y = softmax(x + m);

    // Quantize the input and normalized matrix
    auto qx = Quantized<TestType::width, M, N>(x);
    auto qy = Quantized<TestType::width, M, N>(y);

    // Prepare a binary attention mask
    attention::mask::Const<N, 1, M> m_binary;
    // Iterate the indices in row-major order
    for(unsigned i = 0; i < M; ++i) {
        for(unsigned j = 0; j < N; ++j) {
            // Mapping 0 <=> 0, 1 <=> -inf
            m_binary[i][j] = m[i][j] == -INFINITY;
        }
    }

    // Validate quantization: Errors should never exceed one "step of scale"
    BOOST_CHECK(all_close(x, qx.dequantize(), qx.scale));
    BOOST_CHECK(all_close(y, qy.dequantize(), qy.scale));

//    // For debugging print scale parameters
//    std::printf("qx.scale=%4.4f, qx.bias=%4.4f\n", qx.scale, qx.bias);
//    std::printf("qy.scale=%4.4f, qy.bias=%4.4f\n", qy.scale, qy.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);

    // Scale factor for converting integer to float for computing the softmax
    // and converting back to the same integer range
    float scale = 1.0f / (
        (ap_uint<TestType::width + 1>{1} << TestType::width) - 1
    );

    // Instantiate a softmax normalization function with quantization scales
    // inferred above
    Softmax<N, /*PE=*/1, TestType, TestType, QuantActivation<TestType>, M>
        softmax{QuantActivation<TestType>{scale, 0.0}, qx.scale};

    // Normalize all rows of the output matrix
    decltype(softmax)::OStream softmax_out;
    softmax(qx_elems.out, softmax_out, m_binary);

    // Matrix to be filled by softmax stream
    decltype(qy) qs{{}, scale, 0.0};
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax_out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), scale));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(softmax_out.empty());
}

// Tests the softmax normalization with grouped input stream and causal
// attention mask applied
BOOST_AUTO_TEST_CASE(test_softmax_causal_grouped) {
    // Generate a random matrix
    auto x = randf_matrix<M, N>();
    // Generate a causal mask
    auto m = causal_mask<M, N>();
    // Softmax normalize the matrix
    auto y = softmax(x + m);

    // Quantize the input and normalized matrix
    auto qx = Quantized<TestType::width, M, N>(x);
    auto qy = Quantized<TestType::width, M, N>(y);

    // Validate quantization: Errors should never exceed one "step of scale"
    BOOST_CHECK(all_close(x, qx.dequantize(), qx.scale));
    BOOST_CHECK(all_close(y, qy.dequantize(), qy.scale));

//    // For debugging print scale parameters
//    std::printf("qx.scale=%4.4f, qx.bias=%4.4f\n", qx.scale, qx.bias);
//    std::printf("qy.scale=%4.4f, qy.bias=%4.4f\n", qy.scale, qy.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);
    // Group the input stream into PE parallel elements
    GroupStreamElements<TestType, PE> qx_grouped(qx_elems.out);

    // Scale factor for converting integer to float for computing the softmax
    // and converting back to the same integer range
    float scale = 1.0f / (
        (ap_uint<TestType::width + 1>{1} << TestType::width) - 1
    );

    // Instantiate a softmax normalization function with quantization scales
    // inferred above
    Softmax<N / PE, /*PE=*/PE, TestType, TestType, QuantActivation<TestType>, M>
        softmax{QuantActivation<TestType>{scale, 0.0}, qx.scale};

    // Normalize all rows of the output matrix
    decltype(softmax)::OStream softmax_out;
    softmax(qx_grouped.out, softmax_out, attention::mask::CAUSAL);

    // Split the output stream for validation
    SplitStreamElements<ap_uint<TestType::width * PE>, PE> softmax_elems(
        softmax_out
    );
    // Matrix to be filled by softmax stream
    decltype(qy) qs{{}, scale, 0.0};
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax_elems.out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), scale));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(softmax_out.empty());
}

// Tests the softmax normalization with grouped input stream and causal
// attention mask applied
BOOST_AUTO_TEST_CASE(test_softmax_masked_grouped) {
    // Generate a random matrix
    auto x = randf_matrix<M, N>();
    // Generate a random attention mask
    auto m = randf_mask<M, N>();
    // Softmax normalize the matrix
    auto y = softmax(x + m);

    // Quantize the input and normalized matrix
    auto qx = Quantized<TestType::width, M, N>(x);
    auto qy = Quantized<TestType::width, M, N>(y);

    // Prepare a binary attention mask
    attention::mask::Const<N / PE, PE, M> m_binary;
    // Iterate the indices in row-major order
    for(unsigned i = 0; i < M; ++i) {
        for(unsigned j = 0; j < N / PE; ++j) {
            for(int pe = 0; pe < PE; ++pe)
            // Mapping 0 <=> 0, 1 <=> -inf
            m_binary[i][j][pe] = m[i][j * PE + pe] == -INFINITY;
        }
    }

    // Validate quantization: Errors should never exceed one "step of scale"
    BOOST_CHECK(all_close(x, qx.dequantize(), qx.scale));
    BOOST_CHECK(all_close(y, qy.dequantize(), qy.scale));

//    // For debugging print scale parameters
//    std::printf("qx.scale=%4.4f, qx.bias=%4.4f\n", qx.scale, qx.bias);
//    std::printf("qy.scale=%4.4f, qy.bias=%4.4f\n", qy.scale, qy.bias);

    // Generate streams of the quantized matrices
    RowMajorMatrixStreamer<TestType> qx_elems(qx.z);
    // Group the input stream into PE parallel elements
    GroupStreamElements<TestType, PE> qx_grouped(qx_elems.out);

    // Scale factor for converting integer to float for computing the softmax
    // and converting back to the same integer range
    float scale = 1.0f / (
        (ap_uint<TestType::width + 1>{1} << TestType::width) - 1
    );

    // Instantiate a softmax normalization function with quantization scales
    // inferred above
    Softmax<N / PE, /*PE=*/PE, TestType, TestType, QuantActivation<TestType>, M>
        softmax{QuantActivation<TestType>{scale, 0.0}, qx.scale};

    // Normalize all rows of the output matrix
    decltype(softmax)::OStream softmax_out;
    softmax(qx_grouped.out, softmax_out, m_binary);

    // Split the output stream for validation
    SplitStreamElements<ap_uint<TestType::width * PE>, PE> softmax_elems(
        softmax_out
    );
    // Matrix to be filled by softmax stream
    decltype(qy) qs{{}, scale, 0.0};
    // Read the softmax output back into memory
    RowMajorMatrixStreamer<TestType>::read(softmax_elems.out, qs.z);

    // Validate results by checking whether dequantized values are within
    // tolerance: Errors should never exceed one "step of scale"
    //  TODO: Apparently this can happen due to floating point imprecision...
    BOOST_CHECK(all_close(qy.dequantize(), qs.dequantize(), scale));

    // Sanity checks: All streams should be empty by now
    BOOST_CHECK(qx_elems.out.empty());
    BOOST_CHECK(softmax_out.empty());
}
