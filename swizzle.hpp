#ifndef SWIZZLE_HPP
#define SWIZZLE_HPP

// FINN HLSLIB tensors: Typed, little-endian containers backed by ap_uint
#include "tensor.hpp"

// Swizzles a PE x SIMD tensor (SIMD dimension innermost) to SIMD x PE layout
// (PE dimension innermost), i.e., transposes the two dimensions.
//  Note: This un-swizzles as well: Flip SIMD and PE to invert the operation
template<unsigned SIMD, unsigned PE, class Elem>
    tensor<Elem, SIMD, PE> swizzle(const tensor<Elem, PE, SIMD> &in) {
// Inline this small piece of bit moving logic
#pragma HLS INLINE
        // The output tensor will have the same bit width as the input, the
        // content will just be reordered
        tensor<Elem, SIMD, PE> out;
        // Iterate over the SIMD and PE dimensions
        for (unsigned pe = 0; pe < PE; ++pe) {
// This should be unrolled to simple slicing and wiring logic
#pragma HLS UNROLL
            // Slice the SIMD elements of this PE out of the input tensor
            const tensor<Elem, SIMD> elements = in[pe];
            for (unsigned simd = 0; simd < SIMD; ++simd) {
// This should be unrolled to simple slicing and wiring logic
#pragma HLS UNROLL
                // Stitch element from source into the output tensor
                out[simd][pe] = Elem(elements[simd]);
            }
        }
        // Return the swizzled tensor
        return out;
    }

#endif // SWIZZLE_HPP
