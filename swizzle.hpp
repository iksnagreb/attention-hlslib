#ifndef SWIZZLE_HPP
#define SWIZZLE_HPP

// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>
// HLS arbitrary precision types
#include <ap_int.h>

// Slicing of bit vectors
#include "interpret.hpp"

// Swizzles a SIMD x PE x Elements bitvector from SIMD first to PE first layout
//  Note: This un-swizzles as well: Flip SIMD and PE to invert the operation
template<unsigned SIMD, unsigned PE, int Width>
    ap_uint<Width> swizzle(const ap_uint<Width> &in) {
// Inline this small piece of bit moving logic
#pragma HLS INLINE
        // The width must at least be divisible by both, SIMD and PE
        //  Note: This will catch some rather obvious logic errors, but
        //  unfortunately not subtle things like confusing SIMD and PE.
        static_assert(
            Width % (SIMD * PE) == 0, "SIMD and PE must divide Width"
        );
        // Derive the datatype of a single element
        using Element = ap_uint<Width / SIMD / PE>;
        // The output vector will have the same bit width as the input, the
        // content will just be reordered
        ap_uint<Width> out;
        // Slice the input into all separate elements
        auto const elements = Slice<Element>{}(in);
        // Iterate over the SIMD and PE dimensions
        for (unsigned pe = 0; pe < PE; ++pe) {
// This should be unrolled to simple slicing and wiring logic
#pragma HLS UNROLL
            for (unsigned simd = 0; simd < SIMD; ++simd) {
// This should be unrolled to simple slicing and wiring logic
#pragma HLS UNROLL
                // Compute the source index into the input vector
                const unsigned src = pe * SIMD + simd;
                // Compute the destination index into the output vector
                const unsigned dst = (simd * PE + pe) * Element::width;
                // Stitch element from source into the output bitvector
                out(dst + Element::width - 1, dst) = elements(src, 0);
            }
        }
        // Return the swizzled bitvector
        return out;
    }

#endif // SWIZZLE_HPP
