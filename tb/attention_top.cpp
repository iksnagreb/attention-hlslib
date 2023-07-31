// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>
// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>

// Test configuration
#include "attention_config.hpp"

// Synthesizeable testbench top of the attention operator
void attention_top(QStream &q, KStream &k, VStream &v, OStream &out) {
    // Instantiate the attention operator and connect to the input streams
    ScaledDotProductAttention<EF, TF, Shapes, GroupedTypes> attention(q, k, v);
    // Transfer from input to output stream
    for(std::size_t i = 0; i < Shapes::QLen * EF; ++i) {
        out.write(attention.out.read());
    }
}