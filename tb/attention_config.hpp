#ifndef ATTENTION_HLSLIB_ATTENTION_CONFIG_HPP
#define ATTENTION_HLSLIB_ATTENTION_CONFIG_HPP

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
// TODO: It is kind of awkward to adapt the config here to match the tests
//  below. Should be the other way round, such that the ground truth casts
//  to what is specified here...
/*AType_=*/ap_uint<11>,
/*OType_=*/ap_uint<38>
>;

// Embedding fold (EF) and temporal fold (TF) to be used for testing attention
static constexpr std::size_t EF = 2;
static constexpr std::size_t TF = 8;

// Derive new type configuration of grouped streams
using GroupedTypes = attention::Types<
/*QType_=*/ap_uint<Types::QType::width * Shapes::QKDim / EF>,
/*KType_=*/ap_uint<Types::KType::width * Shapes::QKDim / EF>,
/*VType_=*/ap_uint<Types::VType::width * Shapes::VDim / EF>,
/*AType_=*/ap_uint<Types::AType::width * Shapes::KVLen / TF>,
/*OType_=*/ap_uint<Types::OType::width * Shapes::VDim / EF>
>;

// Short type aliases for matrix types
using QMatrix = Matrix<Types::QType, Shapes::QLen, Shapes::QKDim>;
using KMatrix = Matrix<Types::KType, Shapes::KVLen, Shapes::QKDim>;
using VMatrix = Matrix<Types::VType, Shapes::KVLen, Shapes::VDim>;
using AMatrix = Matrix<Types::AType, Shapes::QLen, Shapes::KVLen>;
using OMatrix = Matrix<Types::OType, Shapes::QLen, Shapes::VDim>;

// Short type aliases for streams connecting top in testbench
using QStream = hls::stream<GroupedTypes::QType>;
using KStream = hls::stream<GroupedTypes::KType>;
using VStream = hls::stream<GroupedTypes::VType>;
using OStream = hls::stream<GroupedTypes::OType>;

// Synthesizeable testbench top of the attention operator
void attention_top(QStream &q, QStream &k, VStream &v, OStream &out);

#endif //ATTENTION_HLSLIB_ATTENTION_CONFIG_HPP
