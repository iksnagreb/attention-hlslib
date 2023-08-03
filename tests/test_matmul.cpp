// Setup Boost Tests
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Matrix and matrix streaming utility
#include "utils.hpp"
// StreamTiler used to feed the matmul operation
#include "stream_tiler.hpp"
// StreamedMatMul to be tested
#include "matmul.hpp"

// Configure the datatype to use for tests (single element type)
using TestType = ap_uint<4>;

// Configuration of the tiling tests
static constexpr std::size_t L =  9;
static constexpr std::size_t M = 12;
static constexpr std::size_t N = 15;
static constexpr std::size_t R =  4;  // Tile rows
static constexpr std::size_t C =  3;  // Tile cols
static constexpr std::size_t TH = M / R; // Tile height
static constexpr std::size_t TW = N / C; // Tile width


// Tests matmul of two elementwise streams without adaptation
BOOST_AUTO_TEST_CASE(test_matmul_elementwise_no_adaptation) {
    // Generate two random input matrices
    auto matrix_a = rand_matrix<TestType, L, M>();
    auto matrix_b = rand_matrix<TestType, M, N>();
    // Compute the result of multiplying matrix a width b
    auto matrix_c = matmul(matrix_a, matrix_b);

    // Get the result type of the matmul output
    using OutType = std::decay_t<decltype((matrix_c[0][0]))>;

    // Create input streams of matrix a (left hand side, in row-major order) and
    // matrix b (right hand side, in col-major order)
    RowMajorMatrixStreamer<TestType> stream_a(matrix_a);
    ColMajorMatrixStreamer<TestType> stream_b(matrix_b, /*rep=*/L);

    // Generate the stream of the expected result in row-major order (as it
    // should be produced by the matmul operator)
    RowMajorMatrixStreamer<OutType> stream_c(matrix_c);

    // Create the streamed matmul operator to be tested
    // @formatter:off
    using MatMul = MatMul<M, N, 1, 1, TestType, TestType>; MatMul matmul;
    // @formatter:on
    // Output stream to be filled by the matmul operator
    MatMul::OutStream matmul_out;
    // Apply the matmul operator to the input streams
    matmul(stream_a.out, stream_b.out, matmul_out, L);

    // Compare the streamed matmul to the ground-truth
    BOOST_CHECK(all_equal(matmul_out, stream_c.out));

    // The two input streams must be empty
    //  Note: Checking for empty outputs is covered by all_equal() above
    BOOST_CHECK(stream_a.out.empty());
    BOOST_CHECK(stream_b.out.empty());
}

// Tests matmul of two elementwise streams with adaptation of right hand side
// from row-major to col-major order
BOOST_AUTO_TEST_CASE(test_matmul_elementwise_row2col_adaptation) {
    // Generate two random input matrices
    auto matrix_a = rand_matrix<TestType, L, M>();
    auto matrix_b = rand_matrix<TestType, M, N>();
    // Compute the result of multiplying matrix a width b
    auto matrix_c = matmul(matrix_a, matrix_b);

    // Get the result type of the matmul output
    using OutType = std::decay_t<decltype((matrix_c[0][0]))>;

    // Create input streams of matrix a (left hand side, in row-major order) and
    // matrix b (right hand side, in col-major order)
    RowMajorMatrixStreamer<TestType> stream_a(matrix_a);
    RowMajorMatrixStreamer<TestType> stream_b(matrix_b);

    // Generate the stream of the expected result in row-major order (as it
    // should be produced by the matmul operator)
    RowMajorMatrixStreamer<OutType> stream_c(matrix_c);

    // Adapt the order of the right hand side stream to col-major order
    Row2ColAdapter<M, N, TestType> adapted_b(stream_b.out, Transpose<1>{}, L);

    // Create the streamed matmul operator to be tested
    // @formatter:off
    using MatMul = MatMul<M, N, 1, 1, TestType, TestType>; MatMul matmul;
    // @formatter:on
    // Output stream to be filled by the matmul operator
    MatMul::OutStream matmul_out;
    // Apply the matmul operator to the input streams
    matmul(stream_a.out, adapted_b.out, matmul_out, L);

    // Compare the streamed matmul to the ground-truth
    BOOST_CHECK(all_equal(matmul_out, stream_c.out));

    // The two input streams must be empty
    //  Note: Checking for empty outputs is covered by all_equal() above
    BOOST_CHECK(stream_a.out.empty());
    BOOST_CHECK(stream_b.out.empty());

    // The adapted and repeating stream must be empty as well
    BOOST_CHECK(adapted_b.out.empty());
}

// Tests matmul of tiled input streams without adaptation
BOOST_AUTO_TEST_CASE(test_matmul_tiled_no_adaptation) {
    // Generate two random input matrices
    auto matrix_a = rand_matrix<TestType, L, M>();
    auto matrix_b = rand_matrix<TestType, M, N>();
    // Compute the result of multiplying matrix a width b
    auto matrix_c = matmul(matrix_a, matrix_b);

    // Get the result type of the matmul output
    using OutType = std::decay_t<decltype((matrix_c[0][0]))>;

    // Create input streams of matrix a (left hand side, in row-major order) and
    // matrix b (right hand side, in col-major order)
    RowMajorMatrixStreamer<TestType> stream_a(matrix_a);
    ColMajorMatrixStreamer<TestType> stream_b(matrix_b);

    // Group the elements such that there are C groups per row/col
    GroupStreamElements<TestType, M / R> grouped_a(stream_a.out);
    GroupStreamElements<TestType, M / R> grouped_b(stream_b.out);

    // Generate the stream of the expected result in row-major order (as it
    // should be produced by the matmul operator)
    RowMajorMatrixStreamer<OutType> stream_c(matrix_c);
    // Group the elements such that there are R groups per row
    GroupStreamElements<OutType, N / C> grouped_c(stream_c.out);


    // Derive the chunk type of grouped elements
    using LhsChunk = decltype(grouped_a.out.read());
    using RhsChunk = decltype(grouped_b.out.read());

    // Collect the right hand side groups in tiled
    Col2ColStreamTiler<R, C, N / C, RhsChunk> tiled_b(grouped_b.out, L);

    // Create the streamed matmul operator to be tested
    // @formatter:off
    using MatMul = MatMul<R, C, TH, TW, TestType, TestType>; MatMul matmul;
    // @formatter:on
    // Output stream to be filled by the matmul operator
    MatMul::OutStream matmul_out;
    // Apply the matmul operator to the input streams
    matmul(grouped_a.out, tiled_b.out, matmul_out, L);

    // Compare the streamed matmul to the ground-truth
    BOOST_CHECK(all_equal(matmul_out, grouped_c.out));

    // The two input streams and the original output stream must be empty
    BOOST_CHECK(stream_a.out.empty());
    BOOST_CHECK(stream_b.out.empty());
    BOOST_CHECK(stream_c.out.empty());

    // The grouped streams must be empty
    BOOST_CHECK(grouped_a.out.empty());
    BOOST_CHECK(grouped_b.out.empty());
    BOOST_CHECK(grouped_c.out.empty());

    // The tile stream must be empty, i.e. all tiles must have been consumed by
    // the matmul
    BOOST_CHECK(tiled_b.out.empty());
}

// Tests matmul of tiled input streams with adaptation of right hand side from
// row-major to col-major order
BOOST_AUTO_TEST_CASE(test_matmul_tiled_row2col_adaptation) {
    // Generate two random input matrices
    auto matrix_a = rand_matrix<TestType, L, M>();
    auto matrix_b = rand_matrix<TestType, M, N>();
    // Compute the result of multiplying matrix a width b
    auto matrix_c = matmul(matrix_a, matrix_b);

    // Get the result type of the matmul output
    using OutType = std::decay_t<decltype((matrix_c[0][0]))>;

    // Create input streams of matrix a (left hand side, in row-major order) and
    // matrix b (right hand side, in col-major order)
    RowMajorMatrixStreamer<TestType> stream_a(matrix_a);
    RowMajorMatrixStreamer<TestType> stream_b(matrix_b);

    // Group the elements such that there are C groups per row/col
    GroupStreamElements<TestType, M / R> grouped_a(stream_a.out);
    GroupStreamElements<TestType, N / C> grouped_b(stream_b.out);

    // Generate the stream of the expected result in row-major order (as it
    // should be produced by the matmul operator)
    RowMajorMatrixStreamer<OutType> stream_c(matrix_c);
    // Group the elements such that there are R groups per row
    GroupStreamElements<OutType, N / C> grouped_c(stream_c.out);


    // Derive the chunk type of grouped elements
    using LhsChunk = decltype(grouped_a.out.read());
    using RhsChunk = decltype(grouped_b.out.read());

    // Collect the right hand side groups in tiled
    Row2ColStreamTiler<R, C, M / R, RhsChunk> tiled_b(
        grouped_b.out, Transpose<N / C>{}, L
    );

    // Create the streamed matmul operator to be tested
    // @formatter:off
    using MatMul = MatMul<R, C, TH, TW, TestType, TestType>; MatMul matmul;
    // @formatter:on
    // Output stream to be filled by the matmul operator
    MatMul::OutStream matmul_out;
    // Apply the matmul operator to the input streams
    matmul(grouped_a.out, tiled_b.out, matmul_out, L);

    // Compare the streamed matmul to the ground-truth
    BOOST_CHECK(all_equal(matmul_out, grouped_c.out));

    // The two input streams and the original output stream must be empty
    BOOST_CHECK(stream_a.out.empty());
    BOOST_CHECK(stream_b.out.empty());
    BOOST_CHECK(stream_c.out.empty());

    // The grouped streams must be empty
    BOOST_CHECK(grouped_a.out.empty());
    BOOST_CHECK(grouped_b.out.empty());
    BOOST_CHECK(grouped_c.out.empty());

    // The tile stream must be empty, i.e. all tiles must have been consumed by
    // the matmul
    BOOST_CHECK(tiled_b.out.empty());
}