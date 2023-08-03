#ifndef STREAM_TILER_HPP
#define STREAM_TILER_HPP

// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>
// HLS arbitrary precision types
#include <ap_int.h>

// Slicing of bit vectors
#include "interpret.hpp"
// Swizzle (transpose) of bit vectors
#include "swizzle.hpp"

// Specifies the direction in which a streamed matrix is processed
template<std::size_t TileRows, std::size_t TileCols, std::size_t N = 1>
    struct RowMajor {
        // Count the row and column indices of each tile and the chunks within a
        // tile
        std::size_t tr = 0, tc = 0, n = 0;

        // Note: The order of this is equivalent to the following nested loops
        //
        //  for(std::size_t tr = 0; tr < TileRows; ++tr)
        //      for(std::size_t n = 0; n < N; ++n)
        //          for(std::size_t tc = 0; tc < TileCols; ++tc)

        // Advance to the next index set in row-major order
        void next() {
// This is just nested index increment and should be inlined
#pragma HLS INLINE
            // Row-Major order means the tile columns runs fastest
            ++tc;
            // If it wraps around, one row is done
            if(tc == TileCols) {
                // Reset to the start of the next row
                tc = 0;
                // Count the next chunk before advancing to a full tile
                ++n;
                // If this wraps around, a complete row of tiles is done
                if(n == N) {
                    // Reset to the first row of the next tile
                    n = 0;
                    // Advance to the next tile row
                    ++tr;
                    // If this wraps around, all tiles are done
                    if(tr == TileRows) {
                        // Reset to the start
                        tr = 0;
                    }
                }
            }
        }
    };

// Specifies the direction in which a streamed matrix is processed
template<std::size_t TileRows, std::size_t TileCols, std::size_t N = 1>
    struct ColMajor {
        // Count the row and column indices of each tile and the chunks within a
        // tile
        std::size_t tr = 0, tc = 0, n = 0;

        // Note: The order of this is equivalent to the following nested loops
        //
        //  for(std::size_t tc = 0; tc < TileCols; ++tc)
        //      for(std::size_t n = 0; n < N; ++n)
        //          for(std::size_t tr = 0; tr < TileRows; ++tr)

        // Advance to the next index set in col-major order
        void next() {
// This is just nested index increment and should be inlined
#pragma HLS INLINE
            // Col-Major order means the tile rows run fastest
            ++tr;
            // If it wraps around, one column is done
            if(tr == TileRows) {
                // Reset to the start of the next column
                tr = 0;
                // Count the next chunk before advancing to a full tile
                ++n;
                // If this wraps around, a complete column of tiles is done
                if(n == N) {
                    // Reset to the first column of the next tile
                    n = 0;
                    // Advance to the next tile column
                    ++tc;
                    // If this wraps around, all tiles are done
                    if(tc == TileCols) {
                        // Reset to the start
                        tc = 0;
                    }
                }
            }
        }
    };

// This is just used as a tag to indicate transposed stream tiling
// @formatter:off
template<std::size_t>
    class Transpose {  };
// @formatter:on

// Receives matrix as a stream of chunks (containing S elements each) in IOrder
// and produces a stream of tiles (containing N x S elements) in OOrder.
// @formatter:off
template<
    // Order in which the matrix chunks are streamed in (RowMajor or ColMajor)
    template<std::size_t...> class IOrder,
    // Order in which the matrix tiles are produced (RowMajor or ColMajor)
    template<std::size_t...> class OOrder,
    // Datatype of each individual elements
    class Elem,
    // Specifies whether tiles need to be transposed
    bool Transpose,
    // Number of tile rows on the right hand side
    std::size_t TileRows,
    // Number of tile cols on the right hand side
    std::size_t TileCols,
    // Number of grouped input elements, i.e. input parallelism
    std::size_t GroupSize,
    // Number of input groups making up a tile
    std::size_t NumGroups
>
// @formatter:on
    struct StreamTiler {
        // Bitvector type representing a tile
        //  Each tile collects tile-width x tile-height elements
        using Tile = ap_uint<Elem::width * GroupSize * NumGroups>;
        // Bitvector type representing a tile-height parallel input chunk
        using Chunk = ap_uint<Elem::width * GroupSize>;

        // Output data stream of ordered matrix tiles
        hls::stream<Tile> out;

        // Produces a tile stream with NumGroups x GroupSize layout (GroupSize
        // dimension is innermost) if NOT transposed=true
        //
        //  Example: GroupSize=3, NumGroups=4; Each chunk: | 1 2 3 |
        //      Each tile will have layout like: | 123 123 123 123 |
        //
        // Produces a tile stream with GroupSize x NumGroups layout (NumGroups
        // dimension is innermost) if transposed=true
        //
        //  Example: GroupSize=3, NumGroups=4; Each chunk: | 1 2 3 |
        //      Each tile will have layout like: | 1111 2222 3333 |
        explicit StreamTiler(
            hls::stream<Chunk> &in, const std::size_t rep = 1) {
// Allow functions and loops to overlap in the following
#pragma HLS dataflow
            // Completely buffer the whole input matrix organized as tiles
            // @formatter:off
            Tile buffer[TileRows][TileCols]; read2buffer(in, buffer);
            // @formatter:on

// Set depth of the tile stream to fit the entire input stream length
#pragma HLS stream variable=out depth=rep * TileRows * TileCols

            // Iterate tile indices according to the specified output order
            OOrder<TileRows, TileCols, 1> o_index;
            // Repeatedly iterate over all tiles
            for(std::size_t i = 0; i < rep * TileRows * TileCols; ++i) {
                // Send the next tile into the output stream
                out.write(maybe_transpose(buffer[o_index.tr][o_index.tc]));
                // Next tile index
                o_index.next();
            }
        }

    private:
        // If specified via template arguments, this transposes a tile. If not,
        // this will simply pass through the input.
        static Tile maybe_transpose(const Tile &tile) {
            // Transpose flag is set as template argument, this should be a
            // constexpr and might be optimized away (does HLS work this way?)
            if(Transpose) {
                // Transpose the tile by swizzling the GroupSize and NumGroups
                // dimension
                return swizzle<GroupSize, NumGroups>(tile);
            }
            // Pass-Through
            return tile;
        }

        // Datatype of the internal buffer holding the complete matrix of tiles
        using Buffer = Tile[TileRows][TileCols];

        // Shared implementation of stream to buffer reading
        void read2buffer(hls::stream<Chunk> &in, Buffer &buffer) {
// This is just stream reading logic, should be inlined
#pragma HLS INLINE
            // Iterate tile and chunk indices according to the specified input
            // order
            IOrder<TileRows, TileCols, NumGroups> i_index;
            // It takes N cycles to see all chunks of a tile and there are in
            // total TileRows x TileCols tiles to complete the matrix
            for(std::size_t i = 0; i < NumGroups * TileRows * TileCols; ++i) {
                // Current chunk index within a tile
                std::size_t n = i_index.n;
                // Reference to the tile to be filled next
                Tile &tile = buffer[i_index.tr][i_index.tc];
                // Read the next chunk of data from the input stream into the
                // tiled buffer
                tile((n + 1) * Chunk::width - 1, n * Chunk::width) = in.read();
                // Next tile index
                i_index.next();
            }
        }
    };

// StreamTiler receiving and producing in column-major order
template<class Type, std::size_t... Sizes>
    using Col2ColStreamTiler = StreamTiler<
        ColMajor, ColMajor, Type, /*Transpose=*/false, Sizes...
    >;

// StreamTiler receiving in row-major order and producing in column-major order
template<class Type, std::size_t... Sizes>
    using Row2ColStreamTiler = StreamTiler<
        // Changing the order requires transposing each tile, i.e. change the
        // order of the tile as well, not just the order of tiles
        RowMajor, ColMajor, Type, /*Transpose=*/true, Sizes...
    >;

// StreamTiler receiving and producing in row-major order
template<class Type, std::size_t... Sizes>
    using Row2RowStreamTiler = StreamTiler<
        RowMajor, RowMajor, Type, /*Transpose=*/false, Sizes...
    >;

// StreamTiler receiving in column-major order and producing in row-major order
template<class Type, std::size_t... Sizes>
    using Col2RowStreamTiler = StreamTiler<
        // Changing the order requires transposing each tile, i.e. change the
        // order of the tile as well, not just the order of tiles
        ColMajor, RowMajor, Type, /*Transpose=*/true, Sizes...
    >;

// Adapts a stream of chunks from row-major to column-major order
//  Note: Does not really create tiles, merely buffers all chunks to change the
//  order.
template<class Type, std::size_t TileRows, std::size_t TileCols>
    using Row2ColAdapter = Row2ColStreamTiler<Type, TileRows, TileCols, 1, 1>;

// Adapts a stream of chunks from column-major to row-major order
//  Note: Does not really create tiles, merely buffers all chunks to change the
//  order.
template<class Type, std::size_t TileRows, std::size_t TileCols>
    using Col2RowAdapter = Col2RowStreamTiler<Type, TileRows, TileCols, 1, 1>;

#endif // STREAM_TILER_HPP
