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
    class Activation = PassThroughActivation<float>
>
    struct Softmax {
        // Activation function which potentially contains parameter: e.g.
        // thresholds which need to be initialized at construction/compile time
        Activation activation;

        // Input scale parameters for converting from integer to float
        // representation: Default such that softmax covers the input range of
        // 0.0 to 1.0 mapped to 0 to 2^Width
        //  TODO: These should be properly specified from the outside
        //   cording to actual ranges and quantization parameters...
        const float iscale =
            1.0f / ((ap_uint<IType::width + 1>{1} << IType::width) - 1);
        const float ibias = 0.0;

        // Output scale parameters for converting from float to integer
        // representation: Default such that softmax covers the output range of
        // 0.0 to 1.0 mapped to 0 to 2^Width
        //  TODO: These should be properly specified from the outside
        //   cording to actual ranges and quantization parameters...
        const float oscale =
            1.0f / ((ap_uint<OType::width + 1>{1} << OType::width) - 1);
        const float obias = 0.0;

        // Short names to the input and output streams of parallel elements
        using IStream = hls::stream<ap_uint<GroupSize * IType::width>>;
        using OStream = hls::stream<ap_uint<GroupSize * OType::width>>;

        // Receives repeated streams of not-normalized values and produces a
        // softmax normalized output stream
        void operator()(IStream &in, OStream &out, const std::size_t rep = 1) {
// Use task-level pipelining in the following, allowing the loops to overlap
#pragma HLS dataflow

            // Total length oif the feature map
            static constexpr std::size_t Len = NumGroups * GroupSize;

            // Structure packing all state information which needs to be
            // communicated from one loop to the other
            struct StatePack {
                // Maximum value per row, i.e., per feature map and the count of
                // these values for overflow handling
                // @formatter:off
                IType max_value; ap_uint<clog2(Len + 1)> max_count = 0;
                // @formatter:on

                // The total over the exponentiated feature map, i.e., the sum
                // of exp(x) along the row
                float total = 0;

                // Track whether there has been an overflow accumulating the
                // total
                bool overflow = false;
            };

            // Structure packing the elementwise input and intermediate stream
            // connecting the two loops
            struct ValuePack {
                // Original input type elements as separate elements
                IType ix[GroupSize];
                // Elementwise floating point exponential of the inputs
                float ex[GroupSize];
            };

            // State buffer between the two loops
            hls::stream<StatePack> state_buffer;
// This buffer needs to hold one state per repetition, i.e., per row
#pragma HLS stream variable=state_buffer depth=rep

            // Value buffer between the two loops
            hls::stream<ValuePack> value_buffer;
// This buffer needs to hold one value pack per group of elements
#pragma HLS stream variable=value_buffer depth=rep * NumGroups

            // Scope of the first loop to have simple short names for locals
            {
                // Local variables for building up state and values in the first
                // loop iteration
                StatePack state;
                ValuePack value;

                // Operate as long as there are elements in the input stream
                sm_loop1: for(std::size_t i = 0; i < rep * NumGroups; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
                    // Read and slice the next group from the input stream
                    auto buffer = Slice<IType>{}(in.read());

                    // Process the GroupSize elements in parallel
                    for(std::size_t pe = 0; pe < GroupSize; ++pe) {
// Inner loop should be unrolled
#pragma HLS UNROLL
                        // Input element to be processed
                        const IType x = buffer(pe, 0);
                        // Keep track of the maximum value encountered
                        if(state.max_value < x || state.max_count == 0) {
                            // New maximum, occurred once
                            state.max_value = x;
                            state.max_count = 1;
                        } else if(state.max_value == x) {
                            // Got the old maximum again
                            state.max_count++;
                        }
                        // Convert to float, scale and compute exponential
                        // function
                        const float ex = std::exp(iscale * float(x) + ibias);
                        // Accumulate the exponential for normalizing in the
                        // second
                        // loop
                        state.total += ex;
                        // Insert the elements oto the value pack
                        value.ix[pe] = x;
                        value.ex[pe] = ex;
                    }

                    // For each group, forward the value pack to the next loop
                    // to continue processing
                    value_buffer.write(value);
                    // Forward the state at the end of each row, i.e., for each
                    // completion of a feature map
                    if(((i + 1) % NumGroups) == 0) {
                        // Do the overflow checking before handing over to the
                        // next loop
                        state.overflow = std::isinf(state.total);
                        // Send state to be consumed by the next loop
                        state_buffer.write(state);
                        // Reset the maximum tracking
                        state.max_count = 0;
                        // Reset the accumulator
                        state.total = 0;
                    }
                }
            }

            // Scope of the second loop to have simple short names for locals
            {
                // Local variables for holding state and values during the
                // loop iteration
                StatePack state;
                ValuePack value;

                // Denominator shared by whole feature map
                float den;

                // Operate as long as there are elements in the input stream
                sm_loop2: for(std::size_t i = 0; i < rep * NumGroups; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
                    // Receive new state at the start of a row, i.e., for each
                    // new feature map
                    if((i % NumGroups) == 0) {
                        // Receive state from connecting buffer
                        state = state_buffer.read();
                        // Update the denominator, which is shared across the
                        // whole feature map
                        // by default, normalize by the total
                        den = float(state.total);
                        // If there was an overflow, use the count of maximal
                        // values instead
                        if(state.overflow) {
                            den = float(state.max_count);
                        }
                    }

                    // Receive the next group of values form the intermediate
                    // buffer
                    value = value_buffer.read();

                    // Buffer collecting GroupSize elements
                    OType buffer[GroupSize];

                    // Process the GroupSize elements in parallel
                    for(std::size_t pe = 0; pe < GroupSize; ++pe) {
// Inner loop should be unrolled
#pragma HLS UNROLL
                        // Numerator and denominator for normalizing
                        float num;

                        // Overflow handling
                        if(state.overflow) {
                            // In case of an overflow, distribute equal weight
                            // to all occurrences of the maximum value, such
                            // that the weights still sum to one.
                            // @formatter:off
                            num = value.ix[pe] == state.max_value ? 1.0 : 0.0;
                            // @formatter:on
                        } else {
                            // In case of no overflow, normalize the exponential
                            // values by the accumulated total
                            // @formatter:off
                            num = value.ex[pe];
                            // @formatter:on
                        }

                        // Read next element into the buffer and scale to cover
                        // the right output range
                        buffer[pe] = std::round((activation.activate(
                            i % NumGroups, pe, num / den) - obias) / oscale
                        );
                    }

                    // Feed the output stream with flattened buffer
                    out.write(flatten<GroupSize>(buffer));
                }
            }
        }
    };

#endif //ATTENTION_HLSLIB_SOFTMAX_HPP
