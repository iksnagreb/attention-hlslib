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
    // Number of tile rows on the right hand side
    std::size_t TileRows,
    // Number of tile cols on the right hand side
    std::size_t TileCols,
    // Number of chunks per tile
    std::size_t N,
    // Datatype of each chunk (contains S elements)
    class Type
>
// @formatter:on
    struct StreamTiler {
        // Derive the datatype of a tile:
        //  Each tile collects N chunks of Type::width elements
        using Tile = ap_uint<Type::width * N>;

        // Output data stream of ordered matrix tiles
        hls::stream<Tile> out;

        // Produces a tile stream with N x S layout (S dimension is innermost)
        //  Note: S is implicitly specified by the Type::width
        //
        //  Example: S=3, N=4; Enumerate elements in each chunk: | 1 2 3 |
        //      Each tile will have layout like: | 123 123 123 123 |
        explicit StreamTiler(hls::stream<Type> &in, const std::size_t rep = 1) {
// Allow functions and loops to overlap in the following
#pragma HLS dataflow
            // Completely buffer the whole input matrix organized as tiles
            // @formatter:off
            Tile buffer[TileRows][TileCols]; read2buffer(in, buffer);
            // @formatter:on

            // TODO: This is rather sequential...
// Set depth of the tile stream to fit the entire input stream length
#pragma HLS stream variable=out depth=rep * TileRows * TileCols

            // Iterate tile indices according to the specified output order
            OOrder<TileRows, TileCols, 1> o_index;
            // Repeatedly iterate over all tiles
            for(std::size_t i = 0; i < rep * TileRows * TileCols; ++i) {
                // Send the next tile into the output stream
                out.write(buffer[o_index.tr][o_index.tc]);
                // Next tile index
                o_index.next();
            }
        }

        // Produces a tile stream with S x N layout (N dimension is innermost)
        //  Note: For transpose (swizzle) it is necessary to have S explicitly
        //
        //  Example: S=3, N=4; Enumerate elements in each chunk: | 1 2 3 |
        //      Each tile will have layout like: | 1111 2222 3333 |
        template<std::size_t S>
            StreamTiler(
                hls::stream<Type> &in, Transpose<S>, std::size_t rep = 1) {
// Allow functions and loops to overlap in the following
#pragma HLS dataflow
                // Completely buffer the whole input matrix organized as tiles
                // @formatter:off
                Tile buffer[TileRows][TileCols]; read2buffer(in, buffer);
                // @formatter:on

                // TODO: This is rather sequential...
// Set depth of the tile stream to fit the entire input stream length
#pragma HLS stream variable=out depth=rep * TileRows * TileCols

                // Iterate tile indices according to the specified output order
                OOrder<TileRows, TileCols, 1> o_index;
                // Repeatedly iterate over all tiles
                for(std::size_t i = 0; i < rep * TileRows * TileCols; ++i) {
                    // Send the next tile into the output stream
                    //  Note: swizzle transposes each tile
                    out.write(swizzle<S, N>(buffer[o_index.tr][o_index.tc]));
                    // Next tile index
                    o_index.next();
                }
            }

    private:
        // Datatype of the internal buffer holding the complete matrix of tiles
        using Buffer = Tile[TileRows][TileCols];

        // Shared implementation of stream to buffer reading
        void read2buffer(hls::stream<Type> &in, Buffer &buffer) {
// This is just stream reading logic, should be inlined
#pragma HLS INLINE
            // Iterate tile and chunk indices according to the specified input
            // order
            IOrder<TileRows, TileCols, N> i_index;
            // It takes N cycles to see all chunks of a tile and there are in
            // total TileRows x TileCols tiles to complete the matrix
            for(std::size_t i = 0; i < N * TileRows * TileCols; ++i) {
                // Current chunk index within a tile
                std::size_t n = i_index.n;
                // Reference to the tile to be filled next
                Tile &tile = buffer[i_index.tr][i_index.tc];
                // Read the next chunk of data from the input stream into the
                // tiled buffer
                tile((n + 1) * Type::width - 1, n * Type::width) = in.read();
                // Next tile index
                i_index.next();
            }
        }
    };

// StreamTiler receiving and producing in column-major order
template<std::size_t TileRows, std::size_t TileCols, std::size_t N, class Type>
    using Col2ColStreamTiler = StreamTiler<
        ColMajor, ColMajor, TileRows, TileCols, N, Type
    >;

// StreamTiler receiving in row-major order and producing in column-major order
template<std::size_t TileRows, std::size_t TileCols, std::size_t N, class Type>
    using Row2ColStreamTiler = StreamTiler<
        RowMajor, ColMajor, TileRows, TileCols, N, Type
    >;

// StreamTiler receiving and producing in row-major order
template<std::size_t TileRows, std::size_t TileCols, std::size_t N, class Type>
    using Row2RowStreamTiler = StreamTiler<
        RowMajor, RowMajor, TileRows, TileCols, N, Type
    >;

// StreamTiler receiving in column-major order and producing in row-major order
template<std::size_t TileRows, std::size_t TileCols, std::size_t N, class Type>
    using Col2RowStreamTiler = StreamTiler<
        ColMajor, RowMajor, TileRows, TileCols, N, Type
    >;

// Adapts a stream of chunks from row-major to column-major order
//  Note: Does not really create tiles, merely buffers all chunks to change the
//  order.
template<std::size_t TileRows, std::size_t TileCols, class Type>
    using Row2ColAdapter = Row2ColStreamTiler<TileRows, TileCols, 1, Type>;

// Adapts a stream of chunks from column-major to row-major order
//  Note: Does not really create tiles, merely buffers all chunks to change the
//  order.
template<std::size_t TileRows, std::size_t TileCols, class Type>
    using Col2RowAdapter = Col2RowStreamTiler<TileRows, TileCols, 1, Type>;

#endif // STREAM_TILER_HPP
