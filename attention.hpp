#ifndef ATTENTION_HPP
#define ATTENTION_HPP

// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>
// HLS arbitrary precision types: ap_uint class template
#include <ap_int.h>

// Auxiliary structures for adding bind_storage pragmas to variables via C++
// template parameters
#include "bind_storage.hpp"
// Tiling of streamed matrices
#include "stream_tiler.hpp"
// Stream matrix-matrix multiplication
#include "matmul.hpp"
// Stream softmax normalization
#include "softmax.hpp"

// Wrap config containers in a namespace to not clutter the global namespace
// with generic terms like "Shapes" and "Types"...
namespace attention {
// Container struct for attention input shape configuration
template<unsigned QKDim_, unsigned QLen_, unsigned VDim_, unsigned KVLen_>
    struct Shapes {
        // Embedding dimension of queries and keys
        static constexpr unsigned QKDim = QKDim_;
        // Length of the query sequence
        static constexpr unsigned QLen = QLen_;
        // Embedding dimension of the values
        static constexpr unsigned VDim = VDim_;
        // Length of the key and value sequence
        static constexpr unsigned KVLen = KVLen_;

        // Flag for validating template arguments of the attention mechanism
        static constexpr bool is_attention_shapes = true;

        // Tests whether the given folding is a valid configuration with respect
        // to the shape configuration in this container
        template<unsigned EF, unsigned TF>
            static constexpr bool is_valid_folding =
                // All shapes must be multiples of their corresponding fold
                !(QKDim % EF) && !(VDim % EF) && !(KVLen % TF);
    };

// Container struct for attention type configuration
template<
    class QType_, class KType_, class VType_, class AType_, class OType_ = void>
    struct Types {
        // Datatype of the query input stream elements
        using QType = QType_;
        // Datatype of the key input stream elements
        using KType = KType_;
        // Datatype of the value input stream elements
        using VType = VType_;

        // Datatype of the attention weights stream elements
        using AType = AType_;
        // Datatype of the output
        using OType = OType_;

        // Flag for validating template arguments of the attention mechanism
        static constexpr bool is_attention_types = true;
    };
}

// WIP: Refactoring of the attention operator interface
template<
    // Embedding dimension of queries and keys
    std::size_t QKDim,
    // Length of the query sequence
    std::size_t QLen,
    // Embedding dimension of the values
    std::size_t VDim,
    // Length of the key and value sequence
    std::size_t KVLen,

    // Folding along the embedding dimensions
    std::size_t EmbFold,
    // Folding along the sequence dimensions
    std::size_t SeqFold,

    // Datatype of query matrix elements
    class QType,
    // Datatype of key matrix elements
    class KType,
    // Datatype of value matrix elements
    class VType,
    // Datatype of mask matrix elements
    class MType,
    // Datatype of attention weights elements
    class AType,
    // Datatype of output elements
    class OType,

    // Datatype of accumulator elements of the Query x Key multiplication
    class AccQKMatMul = typename MACInfo<QKDim, QType, KType>::AccType,
    // Datatype of output elements of the Query x Key multiplication
    class OutQKMatMul = typename MACInfo<QKDim, QType, KType>::AccType,
    // Activation function type of the Query x Key multiplication
    class ActQKMatMul = PassThroughActivation<AccQKMatMul>,

    // Datatype of accumulator elements of the Attention x Value multiplication
    class AccAVMatMul = typename MACInfo<KVLen, AType, VType>::AccType,
    // Datatype of output elements of the Attention x Value multiplication
    class OutAVMatMul = typename MACInfo<KVLen, AType, VType>::AccType,
    // Activation function type of the Attention x Value multiplication
    class ActAVMatMul = PassThroughActivation<AccAVMatMul>,

    // Activation function type of the softmax normalization of the attention
    // weights
    class ActASoftmax = PassThroughActivation<OutQKMatMul>,

    // Resource type for the hardware implementation of the MAC blocks inside
    // the two MatMul operations
    class MacResource = ap_resource_lut,
    // Resource type for the hardware implementation of the memories inside the
    // two StreamTiler instances
    class MemResource = Resource::AUTO
>
    struct ScaledDotProductAttention {
        // Tests whether the given folding is a valid configuration with respect
        // to the shape configuration
        static constexpr bool is_valid_folding =
            // All shapes must be multiples of their corresponding fold
            !(QKDim % EmbFold) && !(VDim % EmbFold) && !(KVLen % SeqFold);
        // Stop compiling if the folding is invalid
        static_assert(is_valid_folding, "Invalid Folding");

        // Derive the input (I_ELEMS) and output (O_ELEMS) parallelism from
        // the new embedding-fold concept
        static constexpr std::size_t I_ELEMS = QKDim / EmbFold;
        static constexpr std::size_t O_ELEMS = VDim / EmbFold;

        // Parallel elements along sequence dimensions according to the new
        // sequence-fold concept
        static constexpr std::size_t S_ELEMS = KVLen / SeqFold;

        // Key matrix stream tiling: Keys arrive in column-major order and tiles
        // are required in column-major order for multiplication as well.
        using KTiler = Col2ColStreamTiler<
            // Note: Embeddings along columns, Sequence along rows
            KType, EmbFold, SeqFold, I_ELEMS, S_ELEMS, MemResource
        >;

        // Value matrix stream tiling: Values arrive in row-major order but
        // tiles are required in column-major order for multiplication.
        using VTiler = Row2ColStreamTiler<
            // Note: Sequence along columns, Embeddings along rows
            VType, SeqFold, EmbFold, O_ELEMS, S_ELEMS, MemResource
        >;

        // MatMul instance configured to do the Query x Key multiplication
        using QKMatMul = MatMul<
            // Size configuration of the matmul operator, i.e. expected number
            // of inputs (tiles) and number of elements to process in parallel.
            //  Note: Embeddings along columns, Sequence along rows, mathing the
            //  KTiler output
            EmbFold, SeqFold, I_ELEMS, S_ELEMS,
            // Input and output types configuration just as above, no matmul
            // types are inferred here, everything can be specified above.
            //  Note: These are elementwise types. Accumulators, outputs and
            //  activations can (and should be) different from the AVMatMul.
            QType, KType, AccQKMatMul, OutQKMatMul, ActQKMatMul,
            // Forward the resource type to select the hardware implementation
            // of MAC operations
            MacResource
        >;

        // MatMul instance configured to do the Attention x Value multiplication
        using AVMatMul = MatMul<
            // Size configuration of the matmul operator, i.e. expected number
            // of inputs (tiles) and number of elements to process in parallel.
            //  Note: Sequence along columns, Embeddings along rows, mathing the
            //  VTiler output
            SeqFold, EmbFold, S_ELEMS, O_ELEMS,
            // Input and output types configuration just as above, no matmul
            // types are inferred here, everything can be specified above.
            //  Note: These are elementwise types. Accumulators, outputs and
            //  activations can (and should be) different from the QKMatMul.
            AType, VType, AccAVMatMul, OutAVMatMul, ActAVMatMul,
            // Forward the resource type to select the hardware implementation
            // of MAC operations
            MacResource
        >;

        // Short names for Input/Output/Internal streams of parallel elements:
        //  Inputs are I_ELEMS groups along the embedding dimensions, outputs
        //  are O_ELEMS groups along embedding dimensions as well.
        //
        //  The mask (might be an input) and the attention weights (produced
        //  internally) are processed in S_ELEMS groups along one of the
        //  sequence dimensions (i.e. th KVLen dimension).
        using QStream = hls::stream<ap_uint<QType::width * I_ELEMS>>;
        using KStream = hls::stream<ap_uint<KType::width * I_ELEMS>>;
        using VStream = hls::stream<ap_uint<VType::width * O_ELEMS>>;
        using MStream = hls::stream<ap_uint<MType::width * S_ELEMS>>;
        using AStream = hls::stream<ap_uint<AType::width * S_ELEMS>>;
        using OStream = hls::stream<ap_uint<OType::width * O_ELEMS>>;

        // Instance objects of the Query-Key and Attention-Value matmul as
        // configured above: These might have activation functions requiring
        // parameters to be initialized once at construction/compile time and
        // thus cannot be instantiated within the operator function call.
        QKMatMul qk_matmul;
        AVMatMul av_matmul;

        // Instance object of the softmax normalization function: This might
        // have scales and an activation function requiring parameters to be
        // initialized once at construction/compile time and thus cannot be
        // instantiated within the operator function call.
        Softmax<
            SeqFold, S_ELEMS, OutQKMatMul, AType, ActASoftmax, QLen
        > softmax;

        // Sets up the attention operator by initializing the activations
        //  Note: Just passes the activation initializer to the matmul and
        //  softmax constructors.
        //  Note: For default-constructible activations this can resemble a
        //  default constructor, i.e. no argument as well.
        explicit ScaledDotProductAttention(
            const ActQKMatMul &act_qk_matmul = {},
            const ActAVMatMul &act_av_matmul = {},
            const ActASoftmax &act_a_softmax = {},
            const float dequant_softmax = 1.0
        ) : qk_matmul{act_qk_matmul},
            av_matmul{act_av_matmul},
            softmax{act_a_softmax, dequant_softmax} {
            // Nothing else to do here...
        }

        // Scaled Dot-Product Attention operator interface as function-call
        // style. Receives three matrices - query, key, value - as input streams
        // and forwards the optional attention mask to the softmax operator.
        template<class Mask>
            void operator()(
                QStream &q, KStream &k, VStream &v, OStream &out, Mask &&mask) {
// Allow functions and loops to overlap in the following
#pragma HLS dataflow

                // Tiling of the streamed key matrix
                KTiler k_tiles(k, QLen);
                // Tiling of the streamed value matrix
                VTiler v_tiles(v, QLen);

                // Multiply the query to the tiled key stream feeding some
                // internal stream connecting to the attention-weights
                // normalization softmax.
                typename QKMatMul::OutStream qk_out;
                qk_matmul(q, k_tiles.out, qk_out, QLen);
// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=qk_out depth=SeqFold
// Implement this FIFO buffer in distributed memory using shift register lookup
// tables
#pragma HLS BIND_STORAGE variable=qk_out type=FIFO impl=SRL

                // Normalize the attention weights via softmax feeding some
                // internal stream connecting to the attention-values matmul.
                AStream softmax_out;
                softmax(qk_out, softmax_out, mask);
// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=softmax_out depth=SeqFold
// Implement this FIFO buffer in distributed memory using shift register lookup
// tables
#pragma HLS BIND_STORAGE variable=softmax_out type=FIFO impl=SRL

                // Multiply the normalized attention weights to the tiled value
                // stream directly feeding the output stream.
                av_matmul(softmax_out, v_tiles.out, out, QLen);
            }

        // No masking version of the attention operator
        void operator()(QStream &q, KStream &k, VStream &v, OStream &out) {
            (*this)(q, k, v, out, attention::mask::NONE);
        }
    };

#endif // ATTENTION_HPP
