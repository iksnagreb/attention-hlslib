// EXIT_SUCCESS macro
#include <cstdlib>

// HLS attention operator
#include "attention.hpp"
// Testing utility functions
#include "tests/utils.hpp"

// Test configuration
#include "attention_config.hpp"

// Computes scaled dot-product attention is software
void attention_sw(QMatrix &q, KMatrix &k, VMatrix &v, OMatrix &o) {
    // Compute the attention weights from queries and keys
    AMatrix a = matmul(q, transpose(k));
    // Normalization of attention weights
    a = softmax(a);
    // Apply the attention weights to the values
    o = matmul(a, v);
}

// Testbench main function, i.e. entrypoint generating and validating the test
// vectors
int main(int, char**) {
    // Generate random query, key and value matrices
    auto q = rand_matrix<Types::QType, Shapes::QLen, Shapes::QKDim>();
    auto k = rand_matrix<Types::KType, Shapes::KVLen, Shapes::QKDim>();
    auto v = rand_matrix<Types::VType, Shapes::KVLen, Shapes::VDim>();
    // Attention operator output matrix
    OMatrix o;
    // Compute the attention software ground-truth
    attention_sw(q, k, v, o);

    // Generate a stream of the ground-truth outputs
    RowMajorMatrixStreamer<Types::OType> o_elems(o);
    // Group the stream of ground-truth outputs according to configured folding
    GroupStreamElements<Types::OType, Shapes::VDim / EF> ground_truth(
        o_elems.out
    );

    // Generate elementwise streams of the input matrices
    RowMajorMatrixStreamer<Types::QType> q_elems(q);
    RowMajorMatrixStreamer<Types::KType> k_elems(k);
    RowMajorMatrixStreamer<Types::VType> v_elems(v);

    // Group input streams according to embedding fold (EF) configuration
    GroupStreamElements<Types::QType, Shapes::QKDim / EF> q_stream(q_elems.out);
    GroupStreamElements<Types::KType, Shapes::QKDim / EF> k_stream(k_elems.out);
    GroupStreamElements<Types::VType, Shapes::VDim / EF> v_stream(v_elems.out);

    // Target output stream
    OStream out;
    // Instantiate the attention operator top function
    attention_top(q_stream.out, k_stream.out, v_stream.out, out);

    // Validate results
    assert(all_equal(ground_truth.out, out));

    // No errors so far, terminate with status code "ok"
    return EXIT_SUCCESS;
}
