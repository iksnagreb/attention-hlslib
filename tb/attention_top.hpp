#ifndef ATTENTION_HLSLIB_ATTENTION_TOP_HPP
#define ATTENTION_HLSLIB_ATTENTION_TOP_HPP

// HLS attention operator
#include "attention.hpp"
// Testing utility functions
#include "tests/utils.hpp"

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


// Configure scaled dot-product attention to do type-casting pass-through
// activation matmul operations.
//  Note: This is not really a practically relevant example, but it is easy
//  to simulate without knowledge of real model parameters and quantization.
using Attention = SDP<
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
    /*ActASoftmax=*/PassThroughActivation<Types::OType>
>;

// Short type aliases for matrix types
using QMatrix = Matrix<Types::QType, Shapes::QLen, Shapes::QKDim>;
using KMatrix = Matrix<Types::KType, Shapes::KVLen, Shapes::QKDim>;
using VMatrix = Matrix<Types::VType, Shapes::KVLen, Shapes::VDim>;
using AMatrix = Matrix<Types::AType, Shapes::QLen, Shapes::KVLen>;
using OMatrix = Matrix<Types::OType, Shapes::QLen, Shapes::VDim>;

// Short type aliases for streams connecting top in testbench
using QStream = Attention::QStream;
using KStream = Attention::KStream;
using VStream = Attention::VStream;
using OStream = Attention::OStream;

// Synthesizeable testbench top of the attention operator
void attention_top(QStream &q, QStream &k, VStream &v, OStream &out);

#endif //ATTENTION_HLSLIB_ATTENTION_TOP_HPP
