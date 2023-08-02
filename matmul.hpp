#ifndef MATMUL_HPP
#define MATMUL_HPP

// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>
// HLS arbitrary precision types: ap_uint class template
#include <ap_int.h>

// Slicing of bit vectors
#include "interpret.hpp"
// Multiply accumulate unit
#include "mac.hpp"
// Use the MACInfo to derive accumulator datatype
#include "mac_info.hpp"
// Flattening of buffers to bit vectors
#include "flatten.hpp"

// Computes a matrix-matrix multiplication where the second matrix is a tiled
// input stream of SF x NF tiles of size SIMD x PE.
//
// Naming Convention:
//  Element: A single element of some ap_uint<> type
//  Chunk: SIMD x Element
//  Tile: PE x Chunk = PE x SIMD x Element
//
// The left hand side is expected in row-major order while the tiles of the
// right hand side are expected in column-major order.
//
// Internally, one complete row of the left hand side is buffered at any time,
// while no buffering of the right hand side is performed. The input generator
// of the right hand side needs to take care of buffering and/or repeating
// the tile stream until all rows of the output are produced.
template<
    unsigned SF, unsigned NF, unsigned SIMD, unsigned PE, class Lhs, class Rhs
>
    struct TiledStreamMatMul {
        // Synapse fold x SIMD widths gives the size of the common dimension
        static constexpr unsigned COMMON_DIM = SF * SIMD;

        // Type alias for convenience
        using RhsTile = Rhs;  // Note: Right hand side arrives as tiles

        // Type alias for convenience
        using LhsChunk = Lhs;  // Note: Left hand side arrives as chunks
        // Derive the datatype of a right hand side chunk
        using RhsChunk = ap_uint<RhsTile::width / PE>;

        // Derive the element types of left and right hand side
        using LhsElement = ap_uint<Lhs::width / SIMD>;
        using RhsElement = ap_uint<Rhs::width / SIMD / PE>;  // Note: Is tiled

        // Accumulators must be wide enough to fit summation of Lhs x Rhs
        // products along the common dim
        using AccType =
            typename MACInfo<COMMON_DIM, LhsElement, RhsElement>::AccType;

        // Expose the datatype of the output stream
        using OutType = ap_uint<PE * AccType::width>;

        // Output stream of PE parallel elements of the row-wise result matrix
        hls::stream<OutType> out;

        // Receives two input matrices as (tiled) streams and fills the output
        // stream
        TiledStreamMatMul(
            hls::stream<Lhs> &lhs, hls::stream<Rhs> &rhs, const unsigned len) {
            // Need to buffer a complete row of the left hand side matrix
            Lhs lhs_buffer[SF];
            // Accumulate PE elements in parallel
            AccType acc[PE];

            // Actual loop variables mapping to the tile indices sf and nf
            unsigned sf = 0, nf = 0;

// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=out depth=len * NF

            // Process in a flat loop over all parallel output elements
            for(unsigned i = 0; i < len * NF * SF; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
                // Currently processed chunk of the left hand side
                LhsChunk lhs_chunk;
                // A zero neuron fold tile index along the PE dimension
                // indicates the start of a new row
                if(nf == 0) {
                    // Read the next SIMD wide chunk of the left hand side
                    lhs_chunk = lhs.read();
                    // Store the chunk into the buffer for reuse
                    lhs_buffer[sf] = lhs_chunk;
                }
                // Still processing from the same row of input
                else {
                    // The next chunk of the left hand side is already buffered
                    lhs_chunk = lhs_buffer[sf];
                }

                // Start of a new tile column
                if(sf == 0) {
                    // Clear all accumulators for the next set of PEs
                    for(unsigned pe = 0; pe < PE; ++pe) {
// Clear all PE accumulators in parallel
#pragma HLS UNROLL
                        // Clear/Reset by setting to zero
                        acc[pe] = AccType{0};
                    }
                }

                // Read the next right hand side tile from the stream and slice
                // into PE chunks
                auto const rhs_chunks = Slice<RhsChunk>{}(rhs.read());

                // Multiply chunks of lhs and rhs vectors in PE parallel
                // elements
                for(unsigned pe = 0; pe < PE; ++pe) {
// Processing elements are processed in parallel
#pragma HLS UNROLL
                    // Slice the lhs chunk into SIMD elements
                    auto const lhs_sliced = Slice<LhsElement>{}(
                        lhs_chunk
                    );
                    // Slice each of the rhs chunks into SIMD elements
                    auto const rhs_sliced = Slice<RhsElement>{}(
                        rhs_chunks(pe, 0)
                    );

                    // Process SIMD elements from each input chunk
                    // TODO: Maybe adapt to mac/mul implementation
                    for(unsigned simd = 0; simd < SIMD; ++simd) {
// Input elements are processed in parallel
#pragma HLS UNROLL
                        // Compute partial MAC to the accumulator
                        // TODO: Add resource type specification
                        acc[pe] += lhs_sliced(simd, 0) * rhs_sliced(simd, 0);
                    }
                }

                // Each iteration processes a new SIMD wide chunk along the
                // synapse fold sf
                ++sf;
                // Once the synapse fold wraps around, a complete tile columns
                // of the rhs has been multiplied with one row of the lhs
                if(sf == SF) {
                    // Reset loop index for next round
                    sf = 0;
                    // Flatten the accumulator along the PE dimension and send
                    // to output stream
                    out.write(flatten<PE>(acc));
                    // Advance to the next folded output neuron
                    ++nf;
                    // If this wraps around, a complete row has been produced
                    if(nf == NF) {
                        // Reset to the first folded neuron of the next row
                        nf = 0;
                    }
                }
            }
        }
    };

#endif // MATMUL_HPP
