#ifndef ATTENTION_HLSLIB_SOFTMAX_HPP
#define ATTENTION_HLSLIB_SOFTMAX_HPP

// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>
// HLS arbitrary precision types: ap_uint class template
#include <ap_int.h>
// HLS math function, e.g. hls::exp
// TODO: Somehow the functions inside cannot be linked by gcc...
// #include <hls_math.h>
// C/C++ math library
#include <cmath>
// FINN HLS-Lib utility functions
#include <utils.hpp>
// Slicing of bit vectors
#include <interpret.hpp>
// FINN HLSLIB activation functions: e.g. pass-through and thresholds
#include <activations.hpp>

// Flattening of buffers to bit vectors
#include "flatten.hpp"

// Wrap config containers in a namespace to not clutter the global namespace
// with generic terms like "Shapes" and "Types"...
namespace attention {
namespace mask {
// No attention mask: More of a tag type, does not contain any data
// or functionality
struct None {
    // Empty, just a tag type
};
// Static instance of the tag for convenient tag-dispatching
[[maybe_unused]] static constexpr None NONE;

// Causal attention mask: More of a tag type, does not contain any data
// or functionality
struct Causal {
    // Empty, just a tag type
};
// Static instance of the tag for convenient tag-dispatching
[[maybe_unused]] static constexpr Causal CAUSAL;

// Constant attention mask: The mask is stored as a packed binary matrix
// where 0 <=> 0, 1 <=> -inf
template<
    // Number of input groups making up a complete feature map to be normalized
    std::size_t NumGroups,
    // Number of grouped input elements, i.e. input parallelism
    std::size_t GroupSize,
    // Number of rows of the attention matrix or repetitions of the softmax
    // operator for batches
    std::size_t NumRows = 1
>
    using Const = ap_uint<GroupSize>[NumRows][NumGroups];

// Input attention mask: The input mask is provided via a streaming
// interface, grouping GroupSize mask bits  where 0 <=> 0, 1 <=> -inf
template<
    // Number of input groups making up a complete feature map to be normalized
    std::size_t /*NumGroups*/,
    // Number of grouped input elements, i.e. input parallelism
    std::size_t GroupSize,
    // Number of rows of the attention matrix or repetitions of the softmax
    // operator for batches
    std::size_t /*NumRows*/ = 1
>
    using Input = hls::stream<ap_uint<GroupSize>/*,DEPTH=NumRows * NumGroups*/>;
}
}

// Streaming softmax with GroupSize parallelism handling and task-level
// parallelism
template<
    // Number of input groups making up a complete feature map to be normalized
    std::size_t NumGroups,
    // Number of grouped input elements, i.e. input parallelism
    std::size_t GroupSize,
    // Datatype of single input elements
    class IType,
    // Datatype of single output elements
    class OType,
    // Output activation function
    class Activation = PassThroughActivation<float>,
    // Number of rows of the attention matrix or repetitions of the softmax
    // operator for batches
    std::size_t NumRows = 1
>
    struct Softmax {
        // Activation function which potentially contains parameter: e.g.
        // thresholds which need to be initialized at construction/compile time
        Activation activation;

        // Dequantizer input scale factor for converting from integer to float
        // representation.
        //  Note: No dequantizer bias necessary as softmax is shift invariant
        const float dequant = 1.0;

        // Short names to the input and output streams of parallel elements
        using IStream = hls::stream<ap_uint<GroupSize * IType::width>>;
        using OStream = hls::stream<ap_uint<GroupSize * OType::width>>;

        // Total length of the feature map
        static constexpr std::size_t Len = NumGroups * GroupSize;

        // Receives repeated streams of not-normalized values and produces a
        // softmax normalized output stream
        template<class Mask = attention::mask::None>
            void operator()(IStream &in, OStream &out, Mask &&mask = Mask{}) {
// Use task-level pipelining in the following, allowing the loops to overlap
#pragma HLS dataflow

                // Type used to convert exponentiated elements to integers for
                // accumulation
                //  Note: 24 bits is a kind of arbitrary choice...
                using ZType = ap_uint<24>;

                // Maximum possible value of input elements
                IType max_x =
                    (ap_uint<IType::width + 1>{1} << IType::width) - 1;
                // Maximum possible value of intermediate elements
                ZType max_z =
                    (ap_uint<ZType::width + 1>{1} << ZType::width) - 1;
                // Maximum possible value of exponential of the input elements
                float max_e = std::ceil(std::exp(dequant * max_x));
                // Scale factor to convert from float to the intermediate format
                float scale = (float) max_z / max_e;

                // Structure packing all state information which needs to be
                // communicated from one loop to the other
                struct State {
                    // Maximum value per row, i.e., per feature map and the
                    // count of these values for overflow handling
                    // @formatter:off
                    IType max_value; ap_uint<clog2(Len + 1)> max_count = 0;
                    // @formatter:on

                    // The total over the exponentiated feature map, i.e., the
                    // sum of exp(x) along the row in integer arithmetic
                    //  Note: Needs to fit Len additions of ZType::width sized
                    //  elements + 1 overflow bit
                    ap_uint<ZType::width + Len + 1> total = 0;

                    // Track whether there has been an overflow accumulating the
                    // total
                    bool overflow = false;
                };

                // Structure packing the elementwise input and intermediate
                // stream connecting the two loops
                struct Value {
                    // Original input type elements as separate elements
                    IType ix[GroupSize];
                    // Elementwise floating point exponential of the inputs
                    float ex[GroupSize];
                };

                // State buffer between the two loops
                hls::stream<State> state_buffer;
// This buffer needs to hold one state per repetition, i.e., per row
#pragma HLS stream variable=state_buffer depth=NumRows

                // Value buffer between the two loops
                hls::stream<Value> value_buffer;
// This buffer needs to hold one value pack per group of elements
#pragma HLS stream variable=value_buffer depth=NumRows * NumGroups

                // Scope of the first loop to have simple short names for locals
                {
                    // Local variables for building up state and values in the
                    // first loop iteration
                    State state;
                    Value value;

                    // Operate as long as there are elements in the input stream
                    [[maybe_unused]] sm_loop1:
                    for(std::size_t i = 0; i < NumRows * NumGroups; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
                        // Read and slice the next group from the input stream
                        auto buffer = Slice<IType>{}(in.read());
                        // Get the attention mask bits corresponding to this
                        // group. This might be "Void", depending on the mask
                        // type.
                        auto mask_bits = maybe_mask(mask, i);

                        // Process the GroupSize elements in parallel
                        for(std::size_t pe = 0; pe < GroupSize; ++pe) {
// Inner loop should be unrolled
#pragma HLS UNROLL
                            // Will hold the input element to be processed or
                            // zero if masked
                            IType x = 0;
                            // Will hold the exponential of the input element of
                            // zero if masked
                            float ex = 0.0;

                            // Optionally allows for masked attention
                            if(!is_masked(mask, mask_bits, i, pe)) {
                                // Get the input element at the pe index out of
                                // the packed group of parallel input elements
                                x = buffer(pe, 0);
                                // Convert to float, scale and compute
                                // exponential function
                                ex = std::exp(dequant * float(x));

                                // Keep track of the maximum value encountered
                                //  Note: Overflow handling only if not-masked
                                if(state.max_value < x ||
                                   state.max_count == 0) {
                                    // New maximum, occurred once
                                    state.max_value = x;
                                    state.max_count = 1;
                                } else if(state.max_value == x) {
                                    // Got the old maximum again
                                    state.max_count++;
                                }
                            }

                            // Accumulate the exponential for normalizing in the
                            // second loop: Convert from float to integer to
                            // optimize latency
                            //  TODO: I do not really understand why the extra
                            //   bit is necessary, maybe to account for
                            //   rounding?
                            state.total += ap_uint<ZType::width + 1>{
                                std::round((ex * scale))
                            };

                            // Detect overflow by checking the highest bit
                            if(state.total.test(ZType::width + Len)) {
                                state.overflow = true;
                            }

                            // Insert the elements into the value pack
                            value.ix[pe] = x;
                            value.ex[pe] = ex;
                        }

                        // For each group, forward the value pack to the next
                        // loop to continue processing
                        value_buffer.write(value);
                        // Forward the state at the end of each row, i.e., for
                        // each completion of a feature map
                        if(((i + 1) % NumGroups) == 0) {
                            // Send state to be consumed by the next loop
                            state_buffer.write(state);
                            // Reset the maximum tracking
                            state.max_count = 0;
                            // Reset the accumulator
                            state.total = 0;
                            // Reset the overflow state
                            state.overflow = false;
                        }
                    }
                }

                // Scope of the second loop to have simple short names for
                // locals
                {
                    // Local variables for holding state and values during the
                    // loop iteration
                    State state;
                    Value value;

                    // Denominator shared by whole feature map
                    float den;

                    // Operate as long as there are elements in the input stream
                    [[maybe_unused]] sm_loop2:
                    for(std::size_t i = 0; i < NumRows * NumGroups; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
                        // Receive new state at the start of a row, i.e., for
                        // each new feature map
                        if((i % NumGroups) == 0) {
                            // Receive state from connecting buffer
                            state = state_buffer.read();
                            // Update the denominator, which is shared across
                            // the whole feature map by default, normalize by
                            // the total
                            //  Note: Vitis reports "unsafe type casting from
                            //  type 'size_t' to type 'float'" here, but why?
                            den = float(state.total) / scale;
                            // If there was an overflow, use the count of
                            // maximal values instead
                            if(state.overflow) {
                                den = float(state.max_count);
                            }
                        }

                        // Receive the next group of values form the
                        // intermediate buffer
                        value = value_buffer.read();

                        // Buffer collecting GroupSize elements
                        OType buffer[GroupSize];

                        // Process the GroupSize elements in parallel
                        for(std::size_t pe = 0; pe < GroupSize; ++pe) {
// Inner loop should be unrolled
#pragma HLS UNROLL
                            // Numerator to be normalized
                            float num;

                            // Overflow handling
                            if(state.overflow) {
                                // In case of an overflow, distribute equal
                                // weight to all occurrences of the maximum
                                // value, such that the weights still sum to
                                // one.
                                num =
                                    value.ix[pe] == state.max_value ? 1.0 : 0.0;
                            } else {
                                // In case of no overflow, normalize the
                                // exponential values by the accumulated total
                                num = value.ex[pe];
                            }

                            // Read next element into the buffer and scale to
                            // cover the right output range
                            buffer[pe] = activation.activate(
                                i % NumGroups, pe, num / den
                            );
                        }

                        // Feed the output stream with flattened buffer
                        out.write(flatten<GroupSize>(buffer));
                    }
                }
            }

        // Implementation details of masked attention, mostly tag dispatching
        // depending on the mask type to unify the interface above.
    private:
        // None mask does not manifest as a real mask object
        [[maybe_unused]] auto maybe_mask(attention::mask::None, std::size_t) {
            // Local dummy type serving as "void"
            // @formatter:off
            struct Void {};
            return Void {};
            // @formatter:on
        }

        // Causal mask does not manifest as a real mask object
        [[maybe_unused]] auto maybe_mask(attention::mask::Causal, std::size_t) {
            // Local dummy type serving as "void"
            // @formatter:off
            struct Void {};
            return Void {};
            // @formatter:on
        }

        // Make the constant mask type available using shorthand notation
        using Const = attention::mask::Const<NumGroups, GroupSize, NumRows>;

        // Constant mask can be accessed from memory
        [[maybe_unused]] auto maybe_mask(const Const &mask, std::size_t i) {
            // Convert flat group index to row and col index
            return mask[i / NumGroups][i % NumGroups];
        }

        // Make the input mask available using shorthand notation
        using Input = attention::mask::Input<NumGroups, GroupSize, NumRows>;

        // Input mask is read from input stream
        [[maybe_unused]] auto maybe_mask(Input &mask, std::size_t) {
            return mask.read();
        }

        // Tests whether the element at (i,pe) should be masked
        template<class Mask, class Bits>
            [[maybe_unused]] bool is_masked(
                Mask, Bits, std::size_t, std::size_t) {
                // By default, do not mask anything
                return false;
            }

        // Causal mask does not manifest as bits, only depends on the i and pe
        // indices
        template<class Bits>
            [[maybe_unused]] bool is_masked(
                attention::mask::Causal, Bits, std::size_t i, std::size_t pe) {
                // Compare column and row index to see whether the element is
                // above the main diagonal
                return GroupSize * (i % NumGroups) + pe > i / NumGroups;
            }

        // Constant and input masks manifest as bit-vectors, we do not care from
        // which type these originate
        template<class Mask>
            [[maybe_unused]] bool is_masked(
                Mask &&, ap_uint<GroupSize> bits, std::size_t, std::size_t pe) {
                // Get the mask bit at pe index, 1 <=> -inf which means "masked"
                return bits.test(pe);
            }
    };

#endif //ATTENTION_HLSLIB_SOFTMAX_HPP
