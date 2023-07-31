// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>
// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>

// Test configuration
#include "attention_config.hpp"

// Synthesizeable testbench top of the attention operator
void attention_top(QStream &q, QStream &k, VStream &v, OStream &out) {
    // Instantiate the attention operator and connect to the input streams
    ScaledDotProductAttention<EF, TF, Shapes, GroupedTypes> attention(q, k, v);
    // Transfer from input to output stream
    while(!attention.out.empty()) {
        out.write(attention.out.read());
    }
}