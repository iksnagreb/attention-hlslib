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

        // TODO: Add attention weights and output types?

        // Datatype of the attention weights stream elements
        using AType = AType_;

        // Datatype of the output
        using OType = OType_;

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

        // Output stream of the operator
        //  TODO: Automatically derives the output type. Should this be
        //   specified via Types as well?
        hls::stream<typename AVMatMul::OutType> out;

        // Computes streamed scaled dot-product attention without masking
        ScaledDotProductAttention(QStream &q, KStream &k, VStream &v) {
// Allow functions and loops to overlap in the following
#pragma HLS dataflow
            // Tiling of the streamed key matrix
            KTiler k_tiles(k, Shapes::QLen);
// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=k_tiles.out depth=Shapes::QLen * TF * EF
            // Tiling of the streamed value matrix
            //  Note: Tiles need to be transposed. Specifying this here feels
            //  somewhat awkward...
            VTiler v_tiles(v, Transpose<O_ELEMS>{}, Shapes::QLen);
// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=v_tiles.out depth=Shapes::QLen * TF * EF

            // Query x Keys multiplication producing raw, not-yet-normalized,
            // not-yet-masked attention weights
            QKMatMul matmul1(q, k_tiles.out, Shapes::QLen);
// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=matmul1.out depth=Shapes::QLen * TF

            // Parallel elements of the attention weights stream
            constexpr auto PE = (Shapes::KVLen / TF);
            // Width of the attention weights (input and output) required to
            // compute dummy scale-factors
            constexpr auto IWidth = Types::AType::width / PE;
            constexpr auto OWidth = QKMatMul::OutType::width / PE;

            // Compute input and output scale-factors such that softmax covers
            // the input and output range of 0.0 to 1.0 mapped to 0 to 2^Width.
            //  TODO: These should be properly specified from the outside
            //   according to actual ranges and quantization parameters...
            auto oscale = 1.0f / ((ap_uint<IWidth + 1>{1} << IWidth) - 1);
            auto iscale = 1.0f / ((ap_uint<OWidth + 1>{1} << OWidth) - 1);

            // Normalize the attention weights via softmax
            Softmax<TF, Shapes::KVLen / TF, typename Types::AType> a(
                matmul1.out, iscale, oscale, /*bias=*/0.0f, Shapes::QLen
            );
// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=a.out depth=Shapes::QLen * TF

            // Attention Weights x Values multiplication producing the output
            AVMatMul matmul2(a.out, v_tiles.out, Shapes::QLen);
// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=matmul2.out depth=Shapes::QLen * EF

// Set depth of the output stream to fit the entire output length
#pragma HLS stream variable=out depth=Shapes::QLen * EF
            // Moves data from second matmul output to operator output
            //  Note: This probably adds one cycle delay?
            for(unsigned i = 0; i < Shapes::QLen * EF; ++i) {
                out.write(matmul2.out.read());
            }
        }

        // TODO: Add constructor overloads handling masked attention
    };

#endif // ATTENTION_HPP
