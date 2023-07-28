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

// Flattening of buffers to bit vectors
#include "flatten.hpp"


// Streaming softmax heavily inspired by Xilinx finn-hlslib
//  Note: This is basically the same as the one in Xilinx finn-hlslib, just with
//      input scaling added before exp().
//  Note: Takes an arbitrary input Type (convertible to float) and produces
//      floating point outputs in range [0.0, 1.0].
template<std::size_t Len, class Type>
    void softmax(
        hls::stream<Type> &in, hls::stream<float> &out, float scale = 1.0) {
        // Track the maximum value and the number of maximal values for overflow
        // handling
        // @formatter:off
        Type max_value; ap_uint<clog2(Len + 1)> max_count = 0;
        // @formatter:on

        // The normalizing loop needs both, the exp-value and the original value
        // for overflow handling
        // @formatter:off
        struct Pair {
            Type xi; float ex;
        };
        // @formatter:on

        // Send data from first loop to the second loop via stream
        hls::stream<Pair> buffer;
// Buffer stream with depth to fit the entire input stream length
#pragma HLS stream variable=buffer depth=Len

        // Accumulate the sum of all exp(x) for normalizing in the second loop
        float total = 0;

        // Read the input stream of Len elements
        for(std::size_t i = 0; i < Len; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
            // Read the next input from the stream
            const Type x = in.read();

            // Keep track of the maximum value encountered
            if(max_value < x || max_count == 0) {
                // New maximum, occurred once
                max_value = x;
                max_count = 1;
            } else if(max_value == x) {
                // Got the old maximum again
                max_count++;
            }

            // Convert to float, scale and compute exponential function
            float ex = std::exp(scale * float(x));
            // Accumulate for normalizing
            total += ex;
            // Send into the stream for next loop
            buffer.write({x, ex});
        }

        // Detect overflow of the accumulator
        const bool overflow = std::isinf(total);

        // Read the buffered stream of the same length as the input stream
        for(std::size_t i = 0; i < Len; ++i) {
// Pipeline the steps of this loop
#pragma HLS pipeline II=1 style=flp
            // Read the intermediate, buffered pair
            Pair x = buffer.read();

            // Numerator and denominator of normalization quotient
            float num, den;

            // Overflow handling
            if(overflow) {
                // In case of an overflow, distribute equal weight to all
                // occurrences of the maximum value, such that the weights still
                // sum to one.
                // @formatter:off
                num = x.xi == max_value ? 1.0 : 0.0; den = max_count;
                // @formatter:on
            } else {
                // In case of no overflow, normalize the exponential values by
                // the accumulated total
                // @formatter:off
                num = x.ex; den = total;
                // @formatter:on
            }

            // Normalize and send into output stream
            out.write(num / den);
        }
    }

// Streaming softmax with PE parallelism handling (but not really in parallel,
// contains adapter splitting and merging the PEs again) and row-repetition.
//
// The outputs are scaled such that the softmax produced range of 0 to 1 is
// mapped onto the output range specified by oscale and bias.
template<std::size_t Len, std::size_t PE, class OType>
    struct Softmax {
        // Output stream of integer representation softmax weights
        hls::stream<OType> out;

        // Receives repeated streams of not-normalized values and produces a
        // softmax normalized output stream
        template<class Type>
            explicit Softmax(
                hls::stream<Type> &in,
                const float iscale = 1.0,
                const float oscale = 1.0,
                const float bias = 0.0,
                const std::size_t rep = 1) {
                // TODO: Can the following be synthesized and pipelined?
                // TODO: Parallelize properly to get rid of the splitting,
                //  merging and data-width conversions.

                // Stream of single elements
                hls::stream<ap_uint<Type::width / PE>> elems;
                // Adapt the input stream of PE parallel elements to a single
                // element stream.
                // Operate as long as there are elements in the input stream
                for(std::size_t i = 0; i < rep * Len; ++i) {
                    // Read and slice next group from the input stream
                    auto buffer = Slice<ap_uint<Type::width / PE>>{}(in.read());
                    // Collect the next N elements into the buffer
                    for(std::size_t pe = 0; pe < PE; ++pe) {
                        // Write the next element into the intermediate stream
                        elems.write(buffer(pe, 0));
                    }
                }

                // Softmax weight stream in floating-point representation
                hls::stream<float> weights;
                // Repeatedly apply softmax to the elementwise stream
                for(std::size_t i = 0; i < rep; ++i) {
                    softmax<Len*PE>(elems, weights, iscale);
                }

                // Buffer collecting PE elements
                ap_uint<OType::width / PE> buffer[PE];
                // Operate as long as there are elements in the input stream
                for(std::size_t i = 0; i < rep * Len; ++i) {
                    // Collect the next PE elements into the buffer
                    for(std::size_t pe = 0; pe < PE; ++pe) {
                        // Read next element into the buffer and scale to cover
                        // the right output range
                        buffer[pe] = std::round(
                            (weights.read() - bias) / oscale
                        );
                        // With the last pe element, the buffer is ready to be
                        // sent into the stream
                        if(pe == (PE - 1)) {
                            // Feed the output stream with flattened buffer
                            out.write(flatten<PE>(buffer));
                        }
                    }
                }
            }
    };

#endif //ATTENTION_HLSLIB_SOFTMAX_HPP