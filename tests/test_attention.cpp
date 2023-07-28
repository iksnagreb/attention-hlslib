// Setup Boost Tests
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

// Arbitrary precision integers, i.e. bit-vectors
#include <ap_int.h>

// Matrix and matrix streaming utility
#include "utils.hpp"
// ScaledDotProductAttention to be tested
#include "attention.hpp"

// Configure shapes of the attention inputs for testing
using Shapes = attention::Shapes<
    /*QKDim_=*/4, /*QLen_=*/16, /*VDim_=*/8, /*KVLen_=*/24
>;

// Configure types to be used for attention tests
using Types = attention::Types<
    /*QType_=*/ap_uint<4>,
    /*KType_=*/ap_uint<4>,
    /*VType_=*/ap_uint<4>,
    // TODO: It is kind of awkward to adapt the config here to match the tests
    //  below. Should be the other way round, such that the ground truth casts
    //  to what is specified here...
    /*AType_=*/ap_uint<11>
>;

// Embedding fold (EF) and temporal fold (TF) to be used for testing attention
static constexpr std::size_t EF = 2;
static constexpr std::size_t TF = 8;


// Tests scaled dot-product attention without mask
BOOST_AUTO_TEST_CASE(test_scaled_dot_product_attention_no_mask) {
    // Generate random query, key and value matrices
    auto q = rand_matrix<Types::QType, Shapes::QLen, Shapes::QKDim>();
    auto k = rand_matrix<Types::KType, Shapes::KVLen, Shapes::QKDim>();
    auto v = rand_matrix<Types::VType, Shapes::KVLen, Shapes::VDim>();

    // Compute the attention weights from queries and keys
    auto a = matmul(q, transpose(k));
    // Normalization of attention weights
    a = softmax(a);
    // Apply the attention weights to the values
    auto o = matmul(a, v);

    // Derive the datatype of single output elements
    using OType = std::decay_t<decltype(o[0][0])>;

    // Generate a stream of the ground-truth outputs
    RowMajorMatrixStreamer<OType> o_elems(o);
    // Group the stream of ground-truth outputs according to configured folding
    GroupStreamElements<OType, Shapes::VDim / EF> ground_truth(o_elems.out);

    // Generate elementwise streams of the input matrices
    RowMajorMatrixStreamer<Types::QType> q_elems(q);
    RowMajorMatrixStreamer<Types::KType> k_elems(k);
    RowMajorMatrixStreamer<Types::VType> v_elems(v);

    // Group input streams according to embedding fold (EF) configuration
    GroupStreamElements<Types::QType, Shapes::QKDim / EF> q_stream(q_elems.out);
    GroupStreamElements<Types::KType, Shapes::QKDim / EF> k_stream(k_elems.out);
    GroupStreamElements<Types::VType, Shapes::VDim / EF> v_stream(v_elems.out);

    // Derive the single element accumulator datatype
    using AType = MACInfo<Shapes::QKDim, Types::QType, Types::KType>::AccType;

    // Derive new type configuration of grouped streams
    using GroupedTypes = attention::Types<
        /*QType_=*/ap_uint<Types::QType::width * Shapes::QKDim / EF>,
        /*KType_=*/ap_uint<Types::KType::width * Shapes::QKDim / EF>,
        /*VType_=*/ap_uint<Types::VType::width * Shapes::VDim / EF>,
        /*AType_=*/ap_uint<Types::AType::width * Shapes::KVLen / TF>
    >;

    // Feed the attention operator with folded streams
    ScaledDotProductAttention<EF, TF, Shapes, GroupedTypes> attention(
        q_stream.out, k_stream.out, v_stream.out
    );

    // Collect and compare results
    BOOST_CHECK(all_equal(ground_truth.out, attention.out));

    // Check whether all elementwise streams have been consumed completely
    BOOST_CHECK(q_elems.out.empty());
    BOOST_CHECK(k_elems.out.empty());
    BOOST_CHECK(v_elems.out.empty());
    // Check whether all grouped streams have been consumed completely
    BOOST_CHECK(q_stream.out.empty());
    BOOST_CHECK(k_stream.out.empty());
    BOOST_CHECK(v_stream.out.empty());
}
