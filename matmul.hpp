#ifndef MATMUL_HPP
#define MATMUL_HPP

// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>
// HLS arbitrary precision types: ap_uint class template
#include <ap_int.h>

// Slicing of bit vectors
#include "interpret.hpp"
// FINN HLSLIB activation functions: e.g. pass-through and thresholds
#include "activations.hpp"
// Multiply accumulate unit
#include "mac.hpp"
// Use the MACInfo to derive accumulator datatype
#include "mac_info.hpp"
// Flattening of buffers to bit vectors
#include "flatten.hpp"

// Tests for activation being the PassThroughActivation at compile time
//  Note: Defaults to false for generic activations
template<class Type>
    static constexpr bool is_pass_through = false;

// Specialization of the activation actually being the PassThroughActivation
template<class Elem>
    static constexpr bool is_pass_through<PassThroughActivation<Elem>> = true;

// Tests whether the activation-input-output pair is valid, i.e. whether calling
// Activation on the accumulator type yields the specified output type.
template<class Activation, class Acc, class Out>
    static constexpr bool is_valid_activation =
        std::is_same<Out, decltype(Activation{}.activate(0, 0, Acc{}))>::value;


// Decluttered matrix-matrix multiplication where the second matrix is a tiled
// input stream
template<
    // Number of tile rows on the right hand side
    std::size_t TileRows,
    // Number of tile cols on the right hand side
    std::size_t TileCols,
    // Height of each tile (in number of elements), i.e. input parallelism
    std::size_t TH,
    // Width of each tile (in number of elements), i.e. output parallelism
    std::size_t TW,
    // Datatype of a single input element on the left hand side
    class LhsElem,
    // Datatype of a single input element on the right hand side
    class RhsElem,
    // Datatype of a single accumulator element
    //  Note: Defaults to fit MAC along the common dimension without overflows
    class AccElem = typename MACInfo<TileRows * TH, LhsElem, RhsElem>::AccType,
    // Datatype of a single output element
    //  Note: Defaults to fit MAC along the common dimension without overflows
    class OutElem = typename MACInfo<TileRows * TH, LhsElem, RhsElem>::AccType,
    // Type of activation function to use
    class Activation = PassThroughActivation<AccElem>
>
    struct MatMul {
//        // Test whether the activation-type combination is valid
//        static_assert(
//            is_valid_activation<Activation, AccElem, OutElem>,
//                "Invalid Activation: Would require type cast to OutElem"
//        );

        // Inputs arte chunked for input parallelism along the common dimension
        using LhsChunk = ap_uint<LhsElem::width * TH>;
        using RhsChunk = ap_uint<RhsElem::width * TH>;
        // Outputs are chunked for output parallelism along the width of each
        // tile
        using OutChunk = ap_uint<OutElem::width * TW>;

        // Grouping of chunks along the width yields tiles on the right hand
        // side
        using RhsTile = ap_uint<RhsChunk::width * TW>;

        // Type aliases for all the streams involved
        using LhsStream = hls::stream<LhsChunk>;
        using RhsStream = hls::stream<RhsTile>;
        using OutStream = hls::stream<OutChunk>;

        // Activation function which potentially contains parameters
        Activation activation;

        // Sets up the matmul operator by initializing the activation
        //  Note: For default-constructible activations this can resemble a
        //  default constructor, i.e. no argument, as well.
        explicit MatMul(const Activation &activation = {})
            : activation{activation} {

        }

        // Multiplies the left hand side and the right hand side matrices
        // provided as streams producing an output stream
        void operator() (
            LhsStream &lhs, RhsStream &rhs, OutStream &out, std::size_t len) {
            // Buffer a complete row of the left hand side in TH parallel chunks
            LhsChunk lhs_buffer[TileRows];
            // Accumulator of TW parallel chunks
            AccElem acc[TW];
            // Activations of TW parallel chunks
            OutElem act[TW];

            // Actual loop variables mapping to the tile row index (tr) and tile
            // column index (tc)
            std::size_t tr = 0, tc = 0;

            // Process everything in one flat loop for pipelining. Iterate all
            // tiles of the right hand side repeatedly for each row of the left
            // hand side.
            for(std::size_t i = 0; i < len * TileRows * TileCols; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp

                // Get the left hand side chunk processed in this iteration
                LhsChunk lhs_chunk;
                // A zero tile column index indicates the start of a new row
                if(tc == 0) {
                    // Read the next chunk of the left hand side and store into
                    // the buffer for reuse
                    lhs_buffer[tr] = lhs_chunk = lhs.read();
                }
                // Still processing from the same row of input
                else {
                    // The next chunk of the left hand side is already buffered
                    lhs_chunk = lhs_buffer[tr];
                }

                // Start of a new tile column
                if(tr == 0) {
                    // Clear all accumulators for the start of the next column
                    for(std::size_t pe = 0; pe < TW; ++pe) {
// Clear all accumulators in parallel
#pragma HLS UNROLL
                        // Clear the accumulators by initializing according to
                        // the activation function
                        //  Note: Currently this allows for implicit type cast
                        acc[pe] = activation.init(tc, pe);
                    }
                }

                // Read the next right hand side tile from the stream and slice
                // into TW chunks
                auto const rhs_chunks = Slice<RhsChunk>{}(rhs.read());

                // Each tile yields partial results of TW MAC operations, these
                // are the processing elements
                for(std::size_t pe = 0; pe < TW; ++pe) {
// Unroll all MACs to operate in parallel, there is no dependence between MACs
#pragma HLS UNROLL
                    // Slice the lhs chunk into elements
                    auto const lhs_sliced = Slice<LhsElem>{}(lhs_chunk);
                    // Slice each of the rhs chunks into elements
                    auto const rhs_sliced = Slice<RhsElem>{}(rhs_chunks(pe, 0));

                    // Each of the processing elements receives TH parallel
                    // input elements
                    // TODO: Maybe adapt to mac/mul implementation
                    for(std::size_t simd = 0; simd < TH; ++simd) {
// Process all input elements in parallel
#pragma HLS UNROLL
                        // Compute partial MAC to the accumulator
                        // TODO: Add resource type specification
                        acc[pe] += lhs_sliced(simd, 0) * rhs_sliced(simd, 0);
                    }
                }

                // Each iteration processes a new TH wide chunk, i.e. from the
                // next tile row
                ++tr;
                // Once the tile row index wraps around, a complete tile column
                // of the rhs has been multiplied with one row of the lhs
                if(tr == TileRows) {
                    // Reset loop index for next round
                    tr = 0;

                    // Compute the activation function over all TW accumulation
                    // results
                    for(std::size_t pe = 0; pe < TW; ++pe) {
// There is no dependence between parallel accumulators and activations, thus
// this can be unrolled
#pragma HLS UNROLL
                        // Elementwise activation function from accumulator to
                        // activations buffer
                        //  Note: Currently this allows for implicit type cast
                        act[pe] = activation.activate(tc, pe, acc[pe]);
                    }

                    // Flatten the activations output along the tile width, i.e.
                    // parallel output elements (PE), and send to output stream
                    out.write(flatten<TW>(act));
                    // Advance to the next parallel output chunk, i.e. tile
                    // column
                    ++tc;
                    // If this wraps around, a complete row of output has been
                    // produced, i.e. all tiles have been consumed once.
                    if(tc == TileCols) {
                        // Reset to the first tile of the next row
                        tc = 0;
                    }
                }
            }
        }
    };

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
