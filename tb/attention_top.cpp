// Test configuration
#include "attention_top.hpp"

// Synthesizeable testbench top of the attention operator
void attention_top(QStream &q, KStream &k, VStream &v, OStream &out) {
    // Instantiate the attention operator and connect to the input streams
    Attention attention; attention(q, k, v, out);
}
