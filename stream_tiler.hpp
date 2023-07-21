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
template<unsigned SF, unsigned NF, unsigned N = 1>
    struct RowMajor {
        // Count synapse fold and neuron fold tile indices
        unsigned sf = 0, nf = 0, n = 0;

        // Note: The order of this is equivalent to the following nested loops
        //
        //  for(unsigned sf = 0; sf < SF; ++sf)
        //      for(unsigned n = 0; n < N; ++n)
        //          for(unsigned nf = 0; nf < NF; ++nf)

        // Advance to the next index set in row-major order
        void next() {
// This is just nested index increment and should be inlined
#pragma HLS INLINE
            // Row-Major order means the neuron fold runs fastest
            ++nf;
            // If it wraps around, one row is done
            if (nf == NF) {
                // Reset to the next row
                nf = 0;
                // Count the next chunk before advancing to a full synapse fold
                ++n;
                // If this wraps around, a complete row of tiles is done
                if (n == N) {
                    // Reset to the first row of the next tile
                    n = 0;
                    // Advance to the next synapse fold
                    ++sf;
                    // If this wraps around, all tiles are done
                    if (sf == SF) {
                        // Reset to the start
                        sf = 0;
                    }
                }
            }
        }
    };

// Specifies the direction in which a streamed matrix is processed
template<unsigned SF, unsigned NF, unsigned N = 1>
    struct ColMajor {
        // Count synapse fold and neuron fold tile indices
        unsigned sf = 0, nf = 0, n = 0;

        // Note: The order of this is equivalent to the following nested loops
        //
        //  for(unsigned nf = 0; nf < NF; ++nf)
        //      for(unsigned n = 0; n < N; ++n)
        //          for(unsigned sf = 0; sf < SF; ++sf)

        // Advance to the next index set in col-major order
        void next() {
// This is just nested index increment and should be inlined
#pragma HLS INLINE
            // Col-Major order means the synapse fold runs fastest
            ++sf;
            // If it wraps around, one column is done
            if (sf == SF) {
                // Reset to the next column
                sf = 0;
                // Count the next chunk before advancing to a full neuron fold
                ++n;
                // If this wraps around, a complete column of tiles is done
                if (n == N) {
                    // Reset to the first column of the next tile
                    n = 0;
                    // Advance to the next neuron fold
                    ++nf;
                    // If this wraps around, all tiles are done
                    if (nf == NF) {
                        // Reset to the start
                        nf = 0;
                    }
                }
            }
        }
    };

// This is just used as a tag to indicate transposed stream tiling
// @formatter:off
template<unsigned>
    class Transpose {  };
// @formatter:on

// Receives matrix as a stream of chunks (containing S elements each) in IOrder
// and produces a stream of tiles (containing N x S elements) in OOrder.
// @formatter:off
template<
    // Order in which the matrix chunks are streamed in (RowMajor or ColMajor)
    template<unsigned...> class IOrder,
    // Order in which the matrix tiles are produced (RowMajor or ColMajor)
    template<unsigned...> class OOrder,
    // Synapse-Fold: Number of tiles per tile column
    unsigned SF,
    // Neuron-Fold: Number of tiles per tile row
    unsigned NF,
    // Number of chunks per tile
    unsigned N,
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
        explicit StreamTiler(hls::stream<Type> &in, const unsigned rep = 1) {
            // Completely buffer the whole input matrix organized as tiles
            // @formatter:off
            Tile buffer[SF][NF]; read2buffer(in, buffer);
            // @formatter:on

            // TODO: This is rather sequential...

            // Iterate tile indices according to the specified output order
            OOrder<SF, NF, 1> o_index;
            // Repeatedly iterate over all tiles
            for (unsigned i = 0; i < rep * SF * NF; ++i) {
                // Send the next tile into the output stream
                out.write(buffer[o_index.sf][o_index.nf]);
                // Next tile index
                o_index.next();
            }
        }

        // Produces a tile stream with S x N layout (N dimension is innermost)
        //  Note: For transpose (swizzle) it is necessary to have S explicitly
        //
        //  Example: S=3, N=4; Enumerate elements in each chunk: | 1 2 3 |
        //      Each tile will have layout like: | 1111 2222 3333 |
        template<unsigned S>
            StreamTiler(
                hls::stream<Type> &in, Transpose<S>, const unsigned rep = 1) {
                // Completely buffer the whole input matrix organized as tiles
                // @formatter:off
                Tile buffer[SF][NF]; read2buffer(in, buffer);
                // @formatter:on

                // TODO: This is rather sequential...

                // Iterate tile indices according to the specified output order
                OOrder<SF, NF, 1> o_index;
                // Repeatedly iterate over all tiles
                for (unsigned i = 0; i < rep * SF * NF; ++i) {
                    // Send the next tile into the output stream
                    //  Note: swizzle transposes each tile
                    out.write(swizzle<S, N>(buffer[o_index.sf][o_index.nf]));
                    // Next tile index
                    o_index.next();
                }
            }

    private:
        // Datatype of the internal buffer holding the complete matrix of tiles
        using Buffer = Tile[SF][NF];

        // Shared implementation of stream to buffer reading
        void read2buffer(hls::stream<Type> &in, Buffer &buffer) {
// This is just stream reading logic, should be inlined
#pragma HLS INLINE
            // Iterate tile and chunk indices according to the specified input
            // order
            IOrder<SF, NF, N> i_index;
            // It takes N cycles to see all chunks of a tile and there are in
            // total SF x NF tiles to complete the matrix
            for (unsigned i = 0; i < N * SF * NF; ++i) {
                // Current chunk index within a tile
                unsigned n = i_index.n;
                // Reference to the tile to be filled next
                Tile &tile = buffer[i_index.sf][i_index.nf];
                // Read the next chunk of data from the input stream into the
                // tiled buffer
                tile((n + 1) * Type::width - 1, n * Type::width) = in.read();
                // Next tile index
                i_index.next();
            }
        }
    };

// StreamTiler receiving and producing in column-major order
template<unsigned SF, unsigned NF, unsigned N, class Type>
    using Col2ColStreamTiler = StreamTiler<ColMajor, ColMajor, SF, NF, N, Type>;

// StreamTiler receiving in row-major order and producing in column-major order
template<unsigned SF, unsigned NF, unsigned N, class Type>
    using Row2ColStreamTiler = StreamTiler<RowMajor, ColMajor, SF, NF, N, Type>;

// StreamTiler receiving and producing in row-major order
template<unsigned SF, unsigned NF, unsigned N, class Type>
    using Row2RowStreamTiler = StreamTiler<RowMajor, RowMajor, SF, NF, N, Type>;

// StreamTiler receiving in column-major order and producing in row-major order
template<unsigned SF, unsigned NF, unsigned N, class Type>
    using Col2RowStreamTiler = StreamTiler<ColMajor, RowMajor, SF, NF, N, Type>;

// Adapts a stream of chunks from row-major to column-major order
//  Note: Does not really create tiles, merely buffers all chunks to change the
//  order.
template<unsigned SF, unsigned NF, class Type>
    using Row2ColAdapter = Row2ColStreamTiler<SF, NF, 1, Type>;

// Adapts a stream of chunks from column-major to row-major order
//  Note: Does not really create tiles, merely buffers all chunks to change the
//  order.
template<unsigned SF, unsigned NF, class Type>
    using Col2RowAdapter = Col2RowStreamTiler<SF, NF, 1, Type>;

#endif // STREAM_TILER_HPP
