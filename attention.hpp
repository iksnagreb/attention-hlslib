#ifndef ATTENTION_HPP
#define ATTENTION_HPP

// HLS Stream library: hls::stream<> class template
#include <hls_stream.h>
// HLS arbitrary precision types: ap_uint class template
#include <ap_int.h>

// Tiling of streamed matrices
#include "stream_tiler.hpp"
// Stream matrix-matrix multiplication
#include "matmul.hpp"

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
                !(QKDim % EF) && !(QLen % TF) && !(VDim % EF) && !(KVLen % TF);
    };

// Container struct for attention type configuration
template<class QType_, class KType_, class VType_, class AType_>
    struct Types {
        // Datatype of the query input stream elements
        using QType = QType_;
        // Datatype of the key input stream elements
        using KType = KType_;
        // Datatype of the value input stream elements
        using VType = VType_;

        // TODO: Add attention weights and output types?

        // Datatype of the attention weights stream elements
        using AType = AType_;

        // Flag for validating template arguments of the attention mechanism
        static constexpr bool is_attention_types = true;
    };
}

// Streamed Scaled Dot-Product Attention Mechanism
//
// New Concept: Embedding-Fold (EF) and Temporal-Fold (TF)
// Rules:
//  1. Embedding dimensions are always divided by EF, never by TF
//  2. Sequence dimensions are always divided by TF, never by EF
template<unsigned EF, unsigned TF, class Shapes, class Types>
    struct ScaledDotProductAttention {
        // Validate that the Shapes container is actually a proper attention
        // shape configuration, i.e. instance of the Shapes template
        static_assert(Shapes::is_attention_shapes, "Invalid Shapes");
        // Validate that the Types container is actually a proper attention
        // type configuration, i.e. instance of the Types template
        static_assert(Types::is_attention_types, "Invalid Types");
        // Validate the folding configuration against the shape configuration
        static_assert(
            Shapes::template is_valid_folding<EF, TF>, "Invalid Folding"
        );

        // Derive type-aliases of the input streams for readability
        using QStream = hls::stream<typename Types::QType>;
        using KStream = hls::stream<typename Types::KType>;
        using VStream = hls::stream<typename Types::VType>;

        // Key matrix stream tiling: Keys arrive in column-major order and are
        // required in column-major order for multiplication as well.
        using KTiler = Col2ColStreamTiler<
            // Note: Embeddings along columns, Sequence along rows
            EF, TF, Shapes::KVLen / TF, typename Types::KType
        >;
        // Value matrix stream tiling: Values arrive in row-major order but are
        // required in column-major order for multiplication.
        using VTiler = Row2ColStreamTiler<
            // Note: Sequence along columns, Embeddings along rows
            TF, EF, Shapes::KVLen / TF, typename Types::VType
        >;

        // Datatype of key tiles, i.e. the datatype produced by KTiler stream
        using KTile = typename KTiler::Tile;
        // Datatype of value tiles, i.e. the datatype produced by VTiler stream
        using VTile = typename VTiler::Tile;

        // Derive the input (I_ELEMS) and output (O_ELEMS) parallelism from
        // the new embedding-fold (EF) and temporal-fold (TF) concept
        static constexpr unsigned I_ELEMS = Shapes::QKDim / EF;
        static constexpr unsigned O_ELEMS = Shapes::VDim / EF;

        // MatMul instance configured to do the Query x Key multiplication
        using QKMatMul = TiledStreamMatMul<
            // Note: Order/Tiling matches that of the KTiler instance
            EF, TF, I_ELEMS, Shapes::KVLen / TF, typename Types::QType, KTile
        >;
        // MatMul instance configured to do the Attention x Value multiplication
        using AVMatMul = TiledStreamMatMul<
            // Note: Order/Tiling matches that of the VTiler instance
            TF, EF, Shapes::KVLen / TF, O_ELEMS, typename Types::AType, VTile
        >;

        // As long as there is no softmax and/or datatype conversion
        // implemented, the output type of the QK matmul must be the attention
        // weights type.
        //  TODO: Remove/Refine this assertion as soon as softmax is implemented
        static_assert(
            std::is_same<
                typename Types::AType, typename QKMatMul::OutType>::value,
            "Mismatch of specified and derived type of attention weights"
        );

        // Output stream of the operator
        //  TODO: Automatically derives the output type. Should this be
        //   specified via Types as well?
        hls::stream<typename AVMatMul::OutType> out;

        // Computes streamed scaled dot-product attention without masking
        ScaledDotProductAttention(QStream &q, KStream &k, VStream &v) {
            // Tiling of the streamed key matrix
            KTiler k_tiles(k, Shapes::QLen);
            // Tiling of the streamed value matrix
            //  Note: Tiles need to be transposed. Specifying this here feels
            //  somewhat awkward...
            VTiler v_tiles(v, Transpose<Shapes::KVLen / TF>{}, Shapes::QLen);

            // Query x Keys multiplication producing raw, not-yet-normalized,
            // not-yet-masked attention weights
            QKMatMul matmul1(q, k_tiles.out, Shapes::QLen);

            // Dummy attention weights stream. TODO: Replace by softmax
            hls::stream<typename Types::AType> &a = matmul1.out;

            // Attention Weights x Values multiplication producing the output
            AVMatMul matmul2(a, v_tiles.out, Shapes::QLen);

            // Moves data from second matmul output to operator output
            //  Note: This probably adds one cycle delay?
            for(unsigned i = 0; i < Shapes::QLen * EF; ++i) {
                out.write(matmul2.out.read());
            }
        }

        // TODO: Add constructor overloads handling masked attention
    };

#endif // ATTENTION_HPP
