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
    /*AType_=*/ap_uint<4>,
    /*OType_=*/ap_uint<4>
>;

// Embedding fold and sequence fold to be used for testing attention
static constexpr std::size_t EmbFold = 2;
static constexpr std::size_t SeqFold = 8;

// Derive the input (I_ELEMS) and output (O_ELEMS) parallelism from
// the new embedding-fold concept
static constexpr std::size_t I_ELEMS = Shapes::QKDim / EmbFold;
static constexpr std::size_t O_ELEMS = Shapes::VDim / EmbFold;

// Tests scaled dot-product attention without mask
BOOST_AUTO_TEST_CASE(test_scaled_dot_product_attention_no_mask) {
    // Generate random query, key and value matrices
    auto q = rand_matrix<Types::QType, Shapes::QLen, Shapes::QKDim>();
    auto k = rand_matrix<Types::KType, Shapes::KVLen, Shapes::QKDim>();
    auto v = rand_matrix<Types::VType, Shapes::KVLen, Shapes::VDim>();

    // Compute the attention weights from queries and keys
    auto a = amatmul<Types::AType>(q, transpose(k));
    // Normalization of attention weights
    a = softmax(a);
    // Apply the attention weights to the values
    auto o = amatmul<Types::OType>(a, v);

    // Generate a stream of the ground-truth outputs
    RowMajorMatrixStreamer<Types::OType> o_elems(o);
    // Group the stream of ground-truth outputs according to configured folding
    GroupStreamElements<Types::OType, O_ELEMS> ground_truth(o_elems.out);

    // Generate elementwise streams of the input matrices
    RowMajorMatrixStreamer<Types::QType> q_elems(q);
    RowMajorMatrixStreamer<Types::KType> k_elems(k);
    RowMajorMatrixStreamer<Types::VType> v_elems(v);

    // Group input streams according to embedding fold (EF) configuration
    GroupStreamElements<Types::QType, I_ELEMS> q_stream(q_elems.out);
    GroupStreamElements<Types::KType, I_ELEMS> k_stream(k_elems.out);
    GroupStreamElements<Types::VType, O_ELEMS> v_stream(v_elems.out);

    // Configure scaled dot-product attention to do type-casting pass-through
    // activation matmul operations.
    //  Note: This is not really a practically relevant example, but it is easy
    //  to simulate without knowledge of real model parameters and quantization.
    using Attention = ScaledDotProductAttention<
        /*QKDim=*/Shapes::QKDim,
        /*QLen=*/Shapes::QLen,
        /*VDim=*/Shapes::VDim,
        /*KVLen=*/Shapes::KVLen,
        /*EmbFold=*/EmbFold,
        /*SeqFold=*/SeqFold,
        /*QType=*/Types::QType,
        /*KType=*/Types::KType,
        /*VType=*/Types::VType,
        /*MType=*/Types::AType, // Dummy
        /*AType=*/Types::AType,
        /*OType=*/Types::OType,
        /*AccQKMatMul=*/Types::AType,
        /*OutQKMatMul=*/Types::AType,
        /*ActQKMatMul=*/PassThroughActivation<Types::AType>,
        /*AccAVMatMul=*/Types::OType,
        /*OutAVMatMul=*/Types::OType,
        /*ActAVMatMul=*/PassThroughActivation<Types::OType>,
        /*ActASoftmax=*/PassThroughActivation<float>
    >;
    // Instance of the attention operator
    Attention attention;
    // Output stream to be filled by the attention operator
    Attention::OStream attention_out;
    // Apply the attention operator to the input streams
    attention(q_stream.out, k_stream.out, v_stream.out, attention_out);

    // Collect and compare results
    BOOST_CHECK(all_equal(ground_truth.out, attention_out));

    // Check whether all elementwise streams have been consumed completely
    BOOST_CHECK(q_elems.out.empty());
    BOOST_CHECK(k_elems.out.empty());
    BOOST_CHECK(v_elems.out.empty());
    // Check whether all grouped streams have been consumed completely
    BOOST_CHECK(q_stream.out.empty());
    BOOST_CHECK(k_stream.out.empty());
    BOOST_CHECK(v_stream.out.empty());
}
