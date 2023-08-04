// Test configuration
#include "attention_top.hpp"

// Synthesizeable testbench top of the attention operator
void attention_top(QStream &q, KStream &k, VStream &v, OStream &out) {
    // Instantiate the attention operator and connect to the input streams
    Attention attention(q, k, v);
    // Transfer from input to output stream
    for(std::size_t i = 0; i < Shapes::QLen * EmbFold; ++i) {
        out.write(attention.out.read());
    }
}
