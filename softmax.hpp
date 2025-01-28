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

// Numeric limits of arbitrary precision datatypes
#include "limits.hpp"
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

        // Type used to convert exponentiated elements to integers for
        // accumulation
        //  Note: 24 bits is a kind of arbitrary choice...
        using ZType = ap_uint<24>;
        // Scale factor to convert from float to the intermediate format
        static constexpr float scale = max<ZType>;

        // Receives repeated streams of not-normalized values and produces a
        // softmax normalized output stream
        template<class Mask = attention::mask::None>
            void operator()(IStream &in, OStream &out, Mask &&mask = Mask{}) {
// Use task-level pipelining in the following, allowing the loops to overlap
#pragma HLS dataflow
                // FIFO buffer holding the maximum value per row to be
                // subtracted from the softmax inputs for numerical stability,
                // i.e., to prevent overflow of std::exp.
                hls::stream<IType> max_buffer;
// This buffer needs to hold one state per repetition, i.e., per row
#pragma HLS stream variable=max_buffer depth=2
// Implement this FIFO buffer in distributed memory using shift register lookup
// tables
#pragma HLS BIND_STORAGE variable=max_buffer type=FIFO impl=SRL

                // Copy of the inputs after searching for the maximum
                IStream tmp;
// This buffer needs to hold one copy per group of elements
#pragma HLS stream variable=tmp depth=NumGroups
// Implement this FIFO buffer in distributed memory using shift register lookup
// tables
#pragma HLS BIND_STORAGE variable=tmp type=FIFO impl=SRL

                // Copy of the mask values after searching for the maximum
                hls::stream<decltype(maybe_mask(mask, 0))> _mask;
// This buffer needs to hold one copy per group of elements
#pragma HLS stream variable=_mask depth=NumGroups
// Implement this FIFO buffer in distributed memory using shift register lookup
// tables
#pragma HLS BIND_STORAGE variable=_mask type=FIFO impl=SRL

                // Scope of the zero loop to have simple short names for locals
                {
                    // Track the current maximum value
                    IType max_value = min<IType>;

                    // Operate as long as there are elements in the input stream
                    [[maybe_unused]] sm_loop0:
                    for(std::size_t i = 0; i < NumRows * NumGroups; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
                        // Read the next group of elements from the stream
                        const auto next = in.read();
                        // Send a copy of the input to the next loop
                        tmp.write(next);
                        // Slice the next group from the input stream
                        const auto buffer = Slice<IType>{}(next);
                        // Get the attention mask bits corresponding to this
                        // group. This might be "Void", depending on the mask
                        // type.
                        const auto mask_bits = maybe_mask(mask, i);
                        // Send the (optional) mask bits to the next stage
                        _mask.write(mask_bits);

                        // Process the GroupSize elements in parallel
                        for(std::size_t pe = 0; pe < GroupSize; ++pe) {
// Inner loop should be unrolled
#pragma HLS UNROLL
                            // Single element from the input stream
                            IType x = buffer(pe, 0);
                            // Optionally allows for masked attention
                            if(!is_masked(mask, mask_bits, i, pe)) {
                                // If the new value is larger than the current
                                // maximum, update the maximum
                                if(x > max_value) {
                                    // Only remember the value, we do not care
                                    // for the location here
                                    max_value = x;
                                }
                            }
                        }

                        // Forward the state at the end of each row, i.e., for
                        // each completion of a feature map
                        if(((i + 1) % NumGroups) == 0) {
                            // Send into buffer
                            max_buffer.write(max_value);
                            // Reset the maximum value
                            max_value = min<IType>;
                        }
                    }
                }

                // Structure packing all state information which needs to be
                // communicated from one loop to the other
                struct State {
                    // The total over the exponentiated feature map, i.e., the
                    // sum of exp(x) along the row in integer arithmetic. Needs
                    // to fit Len additions of ZType::width sized elements.
                    ap_uint<ZType::width + clog2(Len) + 1> total = 0;
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
#pragma HLS stream variable=state_buffer depth=2
// Implement this FIFO buffer in distributed memory using shift register lookup
// tables
#pragma HLS BIND_STORAGE variable=state_buffer type=FIFO impl=SRL

                // Value buffer between the two loops
                hls::stream<Value> value_buffer;
// This buffer needs to hold one value pack per group of elements
#pragma HLS stream variable=value_buffer depth=NumGroups
// Implement this FIFO buffer in distributed memory using shift register lookup
// tables
#pragma HLS BIND_STORAGE variable=value_buffer type=FIFO impl=SRL

                // Scope of the first loop to have simple short names for locals
                {
                    // Local variables for building up state and values in the
                    // first loop iteration
                    State state;
                    Value value;
// Completely partition the value arrays along all dimensions to avoid conflicts
// when accessing all in parallel
#pragma HLS ARRAY_PARTITION variable=value.ix complete dim=0
#pragma HLS ARRAY_PARTITION variable=value.ex complete dim=0

                    // Current maximum value per row
                    IType max_value = min<IType>;

                    // Operate as long as there are elements in the input stream
                    [[maybe_unused]] sm_loop1:
                    for(std::size_t i = 0; i < NumRows * NumGroups; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
                        // Receive new state at the start of a row, i.e., for
                        // each new feature map
                        if((i % NumGroups) == 0) {
                            // Read the new maximum for this row
                            max_value = max_buffer.read();
                        }
                        // Read and slice the next group from the input stream
                        const auto buffer = Slice<IType>{}(tmp.read());
                        // Get the attention mask bits corresponding to this
                        // group. This might be "Void", depending on the mask
                        // type.
                        const auto mask_bits = _mask.read();

                        // Process the GroupSize elements in parallel
                        for(std::size_t pe = 0; pe < GroupSize; ++pe) {
// Inner loop should be unrolled
#pragma HLS UNROLL
                            // Initialize to zero input and exponential,
                            // override if not masked.
                            value.ix[pe] = 0;
                            value.ex[pe] = 0;

                            // Optionally allows for masked attention
                            if(!is_masked(mask, mask_bits, i, pe)) {
                                // Get the input element at the pe index out of
                                // the packed group of parallel input elements
                                const auto x = buffer(pe, 0);
                                // Exact, floating-point exponential of the
                                // dequantized input Subtract the maximum for
                                // stability, i.e., prevent std::exp to overflow
                                // to infinity.
                                const auto ex = std::exp(
                                    dequant * (float(x) - max_value)
                                );
                                // Pass on the input to the next stage to be
                                // compared to maximum for overflow handling
                                value.ix[pe] = x;
                                // Pass the exact, floating-point exponential to
                                // the next stage
                                // TODO: Should probably be rounded as well for
                                //  consistency with the sum.
                                value.ex[pe] = ex;

                                // Accumulate the exponential for normalizing in
                                // the next loop. Convert from float to integer
                                // to optimize the latency.
                                state.total += ZType{std::round(ex * scale)};
                            }
                        }

                        // For each group, forward the value pack to the next
                        // loop to continue processing
                        value_buffer.write(value);
                        // Forward the state at the end of each row, i.e., for
                        // each completion of a feature map
                        if(((i + 1) % NumGroups) == 0) {
                            // Send state to be consumed by the next loop
                            state_buffer.write(state);
                            // Reset the accumulator
                            state.total = 0;
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
// Completely partition the value arrays along all dimensions to avoid conflicts
// when accessing all in parallel
#pragma HLS ARRAY_PARTITION variable=value.ix complete dim=0
#pragma HLS ARRAY_PARTITION variable=value.ex complete dim=0

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
                            // the whole feature map.
                            den = float(state.total) / scale;
                        }

                        // Receive the next group of values form the
                        // intermediate buffer
                        value = value_buffer.read();

                        // Buffer collecting GroupSize elements
                        OType buffer[GroupSize];
// Completely partition the output buffer arrays along the first dimension to
// avoid conflicts when accessing all in parallel
#pragma HLS ARRAY_PARTITION variable=buffer complete dim=0

                        // Process the GroupSize elements in parallel
                        for(std::size_t pe = 0; pe < GroupSize; ++pe) {
// Inner loop should be unrolled
#pragma HLS UNROLL
                            // Read next element into the buffer and scale to
                            // cover the right output range
                            buffer[pe] = activation.activate(
                                i % NumGroups, pe, value.ex[pe] / den
                            );
                        }

                        // Feed the output stream with flattened buffer
                        out.write(flatten(buffer));
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
