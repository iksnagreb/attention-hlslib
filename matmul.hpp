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


// Computes a matrix-matrix multiplication where the second matrix is a tiled
// input stream of TileRows x TileCols tiles of size TH x TW elements.
//
// Naming Convention:
//  *Elem: A single element of some ap_uint<> type
//  *Chunk: TH x Elem (for inputs), TW x Elem (for outputs)
//  *Tile: TW x Chunk = TW x TH x Elem (only relevant for Rhs input)
//
// The left hand side is expected in row-major order while the tiles of the
// right hand side are expected in column-major order.
//
// Internally, one complete row of the left hand side is buffered at any time,
// while no buffering of the right hand side is performed. The input generator
// of the right hand side needs to take care of buffering and/or repeating
// the tile stream until all rows of the output are produced.
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
    class Activation = PassThroughActivation<AccElem>,
    // Resource type for the hardware implementation of the MAC block
    class ResourceType = ap_resource_dflt
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
        void operator()(
            LhsStream &lhs, RhsStream &rhs, OutStream &out, std::size_t len) {
            // Buffer a complete row of the left hand side in TH parallel chunks
            LhsChunk lhs_buffer[TileRows];
// Completely partition the left hand side row array along all dimensions to
// avoid conflicts when accessing all in parallel
#pragma HLS ARRAY_PARTITION variable=lhs_buffer complete dim=0
            // Accumulator of TW parallel chunks
            AccElem acc[TW];
// Completely partition the accumulator array along all dimensions to avoid
// conflicts when accessing all in parallel
#pragma HLS ARRAY_PARTITION variable=acc complete dim=0
            // Activations of TW parallel chunks
            OutElem act[TW];
// Completely partition the activations array along all dimensions to avoid
// conflicts when accessing all in parallel
#pragma HLS ARRAY_PARTITION variable=act complete dim=0

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
                    for(std::size_t simd = 0; simd < TH; ++simd) {
// Process all input elements in parallel
#pragma HLS UNROLL
                        // Compute partial MAC to the accumulator
                        acc[pe] += mul(
                            lhs_sliced(simd, 0),
                            rhs_sliced(simd, 0),
                            ResourceType{}
                        );
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
                    out.write(flatten(act));
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

#endif // MATMUL_HPP
