// EXIT_SUCCESS macro and assert
#include <cstdlib>
#include <cassert>

// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Matrix and matrix streaming utility
#include "matrix.hpp"
// StreamTiler to be tested
#include "stream_tiler.hpp"

// Configure the datatype to use for tests (single element type)
using TestType = ap_uint<4>;

// Configuration of the tiling tests
static constexpr std::size_t M = 12;
static constexpr std::size_t N = 15;
static constexpr std::size_t R =  4;  // Tile rows
static constexpr std::size_t C =  3;  // Tile cols

// Tests row-major to col-major adapter
void test_row2col_adapter() {
    // Generate a random matrix
    auto matrix = rand_matrix<TestType, M, N>();

    // Stream the matrix into two streams: Row-Major and Col-Major order
    RowMajorMatrixStreamer<TestType> row_stream(matrix);
    ColMajorMatrixStreamer<TestType> col_stream(matrix);

    // Adapt the row-major stream to col-major order
    Row2ColAdapter<M, N, TestType> adapted_stream(row_stream.out);

    // Adapting the row-major stream to col-major order should be identical to
    // directly streaming in col-major order
    assert(all_equal(adapted_stream.out, col_stream.out));
}

// Tests col-major to row-major adapter
void test_col2row_adapter() {
    // Generate a random matrix
    auto matrix = rand_matrix<TestType, M, N>();

    // Stream the matrix into two streams: Row-Major and Col-Major order
    RowMajorMatrixStreamer<TestType> row_stream(matrix);
    ColMajorMatrixStreamer<TestType> col_stream(matrix);

    // Adapt the col-major stream to row-major order
    Col2RowAdapter<M, N, TestType> adapted_stream(col_stream.out);

    // Adapting the col-major stream to row-major order should be identical to
    // directly streaming in row-major order
    assert(all_equal(adapted_stream.out, row_stream.out));
}

// Tests tiling a row-major order matrix stream into a row-major oder stream of
// tiles
void test_row2row_stream_tiler() {
    // Generate a random matrix
    auto matrix = rand_matrix<TestType, M, N>();
    // Tile and flatten the matrix
    auto tiles = flatten_tiles(tile_matrix<R, C>(matrix));
    // Derive datatype of a tile
    using Tile = std::decay_t<decltype(tiles[0][0])>;
    // Stream the tiles in row-major order: This will serve as the ground-truth
    RowMajorMatrixStreamer<Tile> tile_stream(tiles);

    // Stream the matrix elements in row-major order
    RowMajorMatrixStreamer<TestType> elem_stream(matrix);
    // Group the elements such that there are C groups per row
    GroupStreamElements<TestType, N / C> group_stream(elem_stream.out);

    // Derive the chunk type of grouped elements
    using Chunk = decltype(group_stream.out.read());
    // Tile the group stream such that there are R tiles per column
    Row2RowStreamTiler<R, C, M / R, Chunk> tiler(group_stream.out);

    // Validate the tiler output against the ground-truth
    assert(all_equal(tiler.out, tile_stream.out));
}

// Tests tiling a row-major order matrix stream into a col-major oder stream of
// tiles
void test_row2col_stream_tiler() {
    // Generate a random matrix
    auto matrix = rand_matrix<TestType, M, N>();
    // Tiles are transposed to col-major as well
    auto tiles = flatten_tiles(tile_matrix<R, C>(matrix, TransposeTile{}));
    // Derive datatype of a tile
    using Tile = std::decay_t<decltype(tiles[0][0])>;
    // Stream the tiles in col-major order: This will serve as the ground-truth
    ColMajorMatrixStreamer<Tile> tile_stream(tiles);

    // Stream the matrix elements in row-major order
    RowMajorMatrixStreamer<TestType> elem_stream(matrix);
    // Group the elements such that there are C groups per row
    GroupStreamElements<TestType, N / C> group_stream(elem_stream.out);

    // Derive the chunk type of grouped elements
    using Chunk = decltype(group_stream.out.read());
    // Tile the group stream such that there are R tiles per column
    Row2ColStreamTiler<R, C, M / R, Chunk> tiler(
        // Transpose each tile to adapt from row-major to col-major as well
        group_stream.out, Transpose<N / C>{}
    );

    // Validate the tiler output against the ground-truth
    assert(all_equal(tiler.out, tile_stream.out));
}

// Tests tiling a col-major order matrix stream into a col-major oder stream of
// tiles
void test_col2col_stream_tiler() {
    // Generate a random matrix
    auto matrix = rand_matrix<TestType, M, N>();
    // Tile and flatten the matrix; Tiles are transposed to col-major as well
    auto tiles = flatten_tiles(tile_matrix<R, C>(matrix, TransposeTile{}));
    // Derive datatype of a tile
    using Tile = std::decay_t<decltype(tiles[0][0])>;
    // Stream the tiles in col-major order: This will serve as the ground-truth
    ColMajorMatrixStreamer<Tile> tile_stream(tiles);

    // Stream the matrix elements in col-major order
    ColMajorMatrixStreamer<TestType> elem_stream(matrix);
    // Group the elements such that there are R groups per col
    GroupStreamElements<TestType, M / R> group_stream(elem_stream.out);

    // Derive the chunk type of grouped elements
    using Chunk = decltype(group_stream.out.read());
    // Tile the group stream such that there are R tiles per column
    Col2ColStreamTiler<R, C, N / C, Chunk> tiler(group_stream.out);

    // Validate the tiler output against the ground-truth
    assert(all_equal(tiler.out, tile_stream.out));
}

// Tests tiling a col-major order matrix stream into a row-major oder stream of
// tiles
void test_col2row_stream_tiler() {
    // Generate a random matrix
    auto matrix = rand_matrix<TestType, M, N>();
    // Tile and flatten the matrix
    auto tiles = flatten_tiles(tile_matrix<R, C>(matrix));
    // Derive datatype of a tile
    using Tile = std::decay_t<decltype(tiles[0][0])>;
    // Stream the tiles in row-major order: This will serve as the ground-truth
    RowMajorMatrixStreamer<Tile> tile_stream(tiles);

    // Stream the matrix elements in col-major order
    ColMajorMatrixStreamer<TestType> elem_stream(matrix);
    // Group the elements such that there are R groups per col
    GroupStreamElements<TestType, M / R> group_stream(elem_stream.out);

    // Derive the chunk type of grouped elements
    using Chunk = decltype(group_stream.out.read());
    // Tile the group stream such that there are C tiles per row
    Col2RowStreamTiler<R, C, N / C, Chunk> tiler(
        // Transpose each tile to adapt from col-major to row-major as well
        group_stream.out, Transpose<M / R>{}
    );

    // Validate the tiler output against the ground-truth
    assert(all_equal(tiler.out, tile_stream.out));
}

// Program entrypoint
int main(int, char**) {
    // Test plain adaptation of matrix order (no tiling or grouping of elements)
    test_row2col_adapter();
    test_col2row_adapter();

    // Test actual tiling os streams in all order combinations
    test_row2row_stream_tiler();
    test_row2col_stream_tiler();
    test_col2col_stream_tiler();
    test_col2row_stream_tiler();

    // No error, exit with status code "ok"
    return EXIT_SUCCESS;
}
