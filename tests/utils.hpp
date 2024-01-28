#ifndef ATTENTION_HLSLIB_UTILS_HPP
#define ATTENTION_HLSLIB_UTILS_HPP

// Use nested std::array to represent a matrix
#include <array>
// std::unique_ptr
#include <memory>
// std::rand
#include <cstdlib>
// std::printf
#include <cstdio>

// Matrix of shape M x N of Type: This sis essentially a wrapper around a heap
// allocated, nested std::array. Heap allocation is necessary for simulating
// large, i.e. real-world sample inputs to the attention operator.
template<class Type, std::size_t M, std::size_t N>
    class Matrix {
        // Put the nested array into a unique pointer which can be heap
        // allocated and is managed automatically
        std::unique_ptr<std::array<std::array<Type, N>, M>> matrix;

    public:
        // Allocates an empty matrix on the heap
        Matrix()
            : matrix{std::make_unique<std::array<std::array<Type, N>, M>>()} {}

        // Copy construction of a matrix on the heap
        Matrix(const Matrix &m)
            : matrix{
            std::make_unique<std::array<std::array<Type, N>, M>>(*(m.matrix))
        } {}

        // First level indexing operator of a const matrix. Returns the
        // second-level matrix by value, i.e. does not allow for modification.
        std::array<Type, N> operator[](std::size_t index) const {
            return (*matrix)[index];
        }

        // First level indexing operator of a non-const matrix. Returns
        // reference to the second-level matrix allowing modification.
        std::array<Type, N> &operator[](std::size_t index) {
            return (*matrix)[index];
        }

        // Copy-assignment operator (must be non-const as it modifies). Does not
        // reallocate memory, just copies all elements
        Matrix &operator=(const Matrix &m) {
            // Copy all elements of the nested array (relies on copy assignment
            // of std::array)
            *matrix = *(m.matrix);
            // Return a reference to the (modified) object itself for operator
            // chaining
            return *this;
        }

        // Elementwise addition operator
        friend Matrix operator+ (const Matrix &lhs, const Matrix &rhs) {
            // Allocate a new result matrix
            Matrix result;
            // Iterate the indices in row-major order
            for(unsigned i = 0; i < M; ++i) {
                for(unsigned j = 0; j < N; ++j) {
                    // Add elements from left and right hand side into result
                    result[i][j] = lhs[i][j] + rhs[i][j];
                }
            }
            // Return the result matrix by copy
            return result;
        }
    };


// Tiled matrix of R x C tiles each of M x N elements of type
template<class Type, std::size_t R, std::size_t C, std::size_t M, std::size_t N>
    using TiledMatrix = Matrix<Matrix<Type, M, N>, R, C>;

// Creates a randomly filled matrix of shape M x N of Type
template<class Type, std::size_t M, std::size_t N>
    Matrix<Type, M, N> rand_matrix() {
        // Allocate the matrix on the stack
        Matrix<Type, M, N> matrix;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < N; ++j) {
                // Generate random number in range 0 to 2 ^ Type::width
                matrix[i][j] = std::rand() % (1 << Type::width);  // NOLINT
            }
        }
        // Return the matrix from the stack by copy
        return matrix;
    }

// Creates a randomly filled matrix of shape M x N
template<std::size_t M, std::size_t N>
    Matrix<float, M, N> randf_matrix() {
        // Allocate the matrix on the stack
        Matrix<float, M, N> matrix;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < N; ++j) {
                // Generate random number in range 0.0 to 1.0
                matrix[i][j] = float(std::rand()) / float(RAND_MAX);   // NOLINT
            }
        }
        // Return the matrix from the stack by copy
        return matrix;
    }

// Creates a causal attention mask matrix of shape M x N
template<std::size_t M, std::size_t N>
    Matrix<float, M, N> causal_mask() {
        // Allocate the matrix on the stack
        Matrix<float, M, N> matrix;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < N; ++j) {
                // Mask elements above the main diagonal by -inf
                matrix[i][j] = j > i ? -INFINITY : 0.0;  // NOLINT
            }
        }
        // Return the matrix from the stack by copy
        return matrix;
    }

// Creates a random attention mask matrix of shape M x N
template<std::size_t M, std::size_t N>
    Matrix<float, M, N> randf_mask() {
        // Allocate the matrix on the stack
        Matrix<float, M, N> matrix;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < N; ++j) {
                // Mask elements above the main diagonal by -inf
                matrix[i][j] = std::rand() % 2 ? -INFINITY : 0.0;  // NOLINT
            }
        }
        // Return the matrix from the stack by copy
        return matrix;
    }

// Tiles a matrix into R rows and C columns
template<std::size_t R, std::size_t C, class Type, std::size_t M, std::size_t N>
    auto tile_matrix(const Matrix<Type, M, N> &matrix) {
        // Allocate the tiled matrix on the stack
        TiledMatrix<Type, R, C, M / R, N / C> tiled;
        // Iterate the tile indices in row-major order
        for(unsigned i = 0; i < R; ++i) {
            for(unsigned j = 0; j < C; ++j) {
                // Iterate each tile in row-major order
                for(unsigned k = 0; k < M / R; ++k) {
                    for(unsigned l = 0; l < N / C; ++l) {
                        // Fill element of the tiled matrix
                        tiled[i][j][k][l] =
                            matrix[i * (M / R) + k][j * (N / C) + l];
                    }
                }
            }
        }
        // Return the tiled matrix from the stack by copy
        return tiled;
    }

// Tag indicating transposed tiling
struct TransposeTile {
};

// Tiles a matrix into R rows and C columns; Tiles in col-major order
template<std::size_t R, std::size_t C, class Type, std::size_t M, std::size_t N>
    auto tile_matrix(const Matrix<Type, M, N> &matrix, TransposeTile) {
        // Allocate the tiled matrix on the stack
        //  Note: Transposed tile order (last two dimensions)
        TiledMatrix<Type, R, C, N / C, M / R> tiled;
        // Iterate the tile indices in row-major order
        for(unsigned i = 0; i < R; ++i) {
            for(unsigned j = 0; j < C; ++j) {
                // Iterate each tile in col-major order
                for(unsigned l = 0; l < N / C; ++l) {
                    for(unsigned k = 0; k < M / R; ++k) {
                        // Fill element of the tiled matrix
                        tiled[i][j][l][k] =
                            matrix[i * (M / R) + k][j * (N / C) + l];
                    }
                }
            }
        }
        // Return the tiled matrix from the stack by copy
        return tiled;
    }

// Transposes a matrix
template<class Type, std::size_t M, std::size_t N>
    auto transpose(const Matrix<Type, M, N> &matrix) {
        // Allocate the matrix on the stack
        Matrix<Type, N, M> transposed;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < N; ++i) {
            for(unsigned j = 0; j < M; ++j) {
                // Copy transposed element
                transposed[i][j] = matrix[j][i];
            }
        }
        // Return the matrix from the stack by copy
        return transposed;
    }

// Flattening of buffers to bit vectors
#include "flatten.hpp"

// Flattens a tiled matrix by converting each tile to a flat bitvector
template<std::size_t R, std::size_t C, class Type, std::size_t M, std::size_t N>
    auto flatten_tiles(const TiledMatrix<Type, R, C, M, N> &tiles) {
        // Allocate the tiled matrix on the stack
        Matrix<ap_uint<Type::width * M * N>, R, C> flattened;
        // Iterate the tile indices in row-major order
        for(unsigned i = 0; i < R; ++i) {
            for(unsigned j = 0; j < C; ++j) {
                // Create a flat buffer of elements from each tile
                Type buffer[M * N];
                // Iterate each tile in row-major order
                for(unsigned k = 0; k < M; ++k) {
                    for(unsigned l = 0; l < N; ++l) {
                        // Collect tile elements
                        buffer[k * N + l] = tiles[i][j][k][l];
                    }
                }
                // Flatten the buffer into a bitvector
                flattened[i][j] = flatten<N * M>(buffer);
            }
        }
        // Return the flattened tiled matrix from the stack by copy
        return flattened;
    }

// Format and print a matrix to stdout
template<class Type, std::size_t M, std::size_t N>
    void print_matrix(
        const Matrix<Type, M, N> &matrix, const char *f = "%8x ") {
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < N; ++j) {
                // Formatted print of the matrix element
                std::printf(f, matrix[i][j]);
            }
            // Insert a line break
            std::printf("\n");
        }
    }

// Utility to get the accumulator type necessary to fit a MAC operation
#include "mac_info.hpp"

// Multiplies two matrices
template<class Lhs, class Rhs, std::size_t M, std::size_t N, std::size_t L>
    auto matmul(const Matrix<Lhs, M, N> &lhs, const Matrix<Rhs, N, L> &rhs) {
        // Allocate the result matrix on the stack
        Matrix<typename MACInfo<N, Lhs, Rhs>::AccType, M, L> result;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < L; ++j) {
                // Clear the accumulator
                result[i][j] = 0;
                // Iterate the common dimension, i.e. accumulate the dot-product
                for(unsigned k = 0; k < N; ++k) {
                    // Accumulate the dot-product
                    result[i][j] += lhs[i][k] * rhs[k][j];
                }
            }
        }
        // Return the matrix from the stack by copy
        return result;
    }

// Multiplies two matrices with explicitly specified accumulator type
template<
    class Acc, class Lhs, class Rhs, std::size_t M, std::size_t N, std::size_t L
>
    auto amatmul(const Matrix<Lhs, M, N> &lhs, const Matrix<Rhs, N, L> &rhs) {
        // Allocate the result matrix on the stack
        Matrix<Acc, M, L> result;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < L; ++j) {
                // Clear the accumulator
                result[i][j] = 0;
                // Iterate the common dimension, i.e. accumulate the dot-product
                for(unsigned k = 0; k < N; ++k) {
                    // Accumulate the dot-product
                    result[i][j] += lhs[i][k] * rhs[k][j];
                }
            }
        }
        // Return the matrix from the stack by copy
        return result;
    }

// Multiplies two matrices
template<std::size_t M, std::size_t N, std::size_t L>
    auto
    matmul(const Matrix<float, M, N> &lhs, const Matrix<float, N, L> &rhs) {
        // Allocate the result matrix on the stack
        Matrix<float, M, L> result;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < L; ++j) {
                // Clear the accumulator
                result[i][j] = 0;
                // Iterate the common dimension, i.e. accumulate the dot-product
                for(unsigned k = 0; k < N; ++k) {
                    // Accumulate the dot-product
                    result[i][j] += lhs[i][k] * rhs[k][j];
                }
            }
        }
        // Return the matrix from the stack by copy
        return result;
    }

// Compares two matrices element-by-element
template<class Type, std::size_t M, std::size_t N>
    bool all_equal(
        const Matrix<Type, M, N> &lhs, const Matrix<Type, M, N> &rhs) {
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < N; ++j) {
                // If any pair of elements is not equal, the matrices differ
                if(lhs[i][j] != rhs[i][j]) {
                    return false;
                }
            }
        }
        // All elements are equal
        return true;
    }

// Compares two matrices element-by-element
template<class Type, std::size_t M, std::size_t N>
    bool all_close(const Matrix<Type, M, N> &lhs, const Matrix<Type, M, N> &rhs,
                   const Type epsilon) {
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            for(unsigned j = 0; j < N; ++j) {
                // If any pair of elements is not equal, the matrices differ
                if(std::abs(lhs[i][j] - rhs[i][j]) > epsilon) {
                    return false;
                }
            }
        }
        // All elements are equal
        return true;
    }

// HLS streaming interface
#include <hls_stream.h>

// Prints a stream element-by-element to stdout
template<class Type>
    void print_stream(hls::stream<Type> &in, const char *f = "%8x ") {
        // Print as long as there are elements int the stream
        while(!in.empty()) {
            // Print formatted element to stdout
            std::printf(f, in.read());
        }
    }

// Compares two streams element-by-element
template<class Type>
    bool all_equal(hls::stream<Type> &lhs, hls::stream<Type> &rhs) {
        // Iterate while both streams have elements
        while(!lhs.empty() && !rhs.empty()) {
            // If any pair of elements is not equal, the streams differ
            if(lhs.read() != rhs.read()) {
                return false;
            }
        }
        // If both streams are empty, all elements are equal
        return lhs.empty() && rhs.empty();
    }

// Streams a matrix element-by-element in row-major order
template<class Type>
    struct RowMajorMatrixStreamer {
        // The stream fed by the streamer
        hls::stream<Type> out;

        // Takes a matrix (block of memory) and inserts it into the stream
        template<std::size_t M, std::size_t N>
            explicit RowMajorMatrixStreamer(
                const Matrix<Type, M, N> &in, const unsigned rep = 1) {
                // Repeat the matrix stream
                for(unsigned n = 0; n < rep; ++n) {
                    // Iterate the indices in row-major order
                    for(unsigned i = 0; i < M; ++i) {
                        for(unsigned j = 0; j < N; ++j) {
                            // Insert element into the stream
                            out.write(in[i][j]);
                        }
                    }
                }
            }

        // Extracts a matrix from a row-major oder stream into the buffer
        template<std::size_t M, std::size_t N>
            static void read(hls::stream<Type> &in, Matrix<Type, M, N> &out) {
                // Iterate the indices in row-major order
                for(unsigned i = 0; i < M; ++i) {
                    for(unsigned j = 0; j < N; ++j) {
                        // Extract element from the stream
                        out[i][j] = in.read();
                    }
                }
            }
    };

// Streams a matrix element-by-element in column-major order
template<class Type>
    struct ColMajorMatrixStreamer {
        // The stream fed by the streamer
        hls::stream<Type> out;

        // Takes a matrix (block of memory) and inserts it into the stream
        template<std::size_t M, std::size_t N>
            explicit ColMajorMatrixStreamer(
                const Matrix<Type, M, N> &in, const unsigned rep = 1) {
                // Repeat the matrix stream
                for(unsigned n = 0; n < rep; ++n) {
                    // Iterate the indices in column-major order
                    for(unsigned j = 0; j < N; ++j) {
                        for(unsigned i = 0; i < M; ++i) {
                            // Insert element into the stream
                            out.write(in[i][j]);
                        }
                    }
                }
            }

        // Extracts a matrix from a column-major oder stream into the buffer
        template<std::size_t M, std::size_t N>
            static void read(hls::stream<Type> &in, Matrix<Type, M, N> &out) {
                // Iterate the indices in column-major order
                for(unsigned j = 0; j < N; ++j) {
                    for(unsigned i = 0; i < M; ++i) {
                        // Extract element from the stream
                        out[i][j] = in.read();
                    }
                }
            }
    };

// Arbitrary precision numbers, i.e. bit-vectors
#include <ap_int.h>
// Slicing of bit-vectors
#include <interpret.hpp>

// Adapts a stream by grouping N sequential elements
template<class Type, std::size_t N>
    struct GroupStreamElements {
        // The stream fed by the streamer
        hls::stream<ap_uint<Type::width * N>> out;

        // Takes the input stream of Type and feeds the grouped stream
        explicit GroupStreamElements(hls::stream<Type> &in) {
            // Buffer collecting N elements
            Type buffer[N];
            // Operate as long as there are elements in the input stream
            while(!in.empty()) {
                // Collect the next N elements into the buffer
                for(std::size_t i = 0; i < N; ++i) {
                    // The stream must not be empty before completing a group
                    //  I.e., the length of the stream must be a multiple of N
                    assert(!in.empty());
                    // Read next element into the buffer
                    buffer[i] = in.read();
                }
                // Feed the output stream with flattened buffer
                out.write(flatten<N>(buffer));
            }
        }
    };

// Adapts a stream by un-grouping N sequential elements
template<class Type, std::size_t N>
    struct SplitStreamElements {
        // The input type must be divisible by N
        static_assert(
            Type::width % N == 0, "Type::width must be divisible by N"
        );
        // The stream fed by the streamer
        hls::stream<ap_uint<Type::width / N>> out;

        // Takes the input stream of Type and feeds the grouped stream
        explicit SplitStreamElements(hls::stream<Type> &in) {
            // Operate as long as there are elements in the input stream
            while(!in.empty()) {
                // Read and slice next group from the input stream
                auto buffer = Slice < ap_uint<Type::width / N>>
                {}
                (in.read());
                // Collect the next N elements into the buffer
                for(std::size_t i = 0; i < N; ++i) {
                    // Write the next element into the output stream
                    out.write(buffer(i, 0));
                }
            }
        }
    };

// Softmax normalizes a vector of N floats
template<std::size_t N>
    std::array<float, N> softmax(const std::array<float, N> &x) {
        // Allocate a result vector of the same size on the stack
        std::array<float, N> y{};
        // Track the total, maximum value and the number of occurrences of the
        // maximum value for overflow handling
        // @formatter:off
        float total = 0.0, max_value = -INFINITY; std::size_t max_count = 0;
        // @formatter:on

        // First pass over the input values to compute exp(x) and track maximum
        for(std::size_t i = 0; i < N; ++i) {
            // Keep track of the maximum value encountered
            if(max_value < x[i] || max_count == 0) {
                // New maximum, occurred once
                max_value = x[i];
                max_count = 1;
            } else if(max_value == x[i]) {
                // Got the old maximum again
                max_count++;
            }
            // Compute exp(x) to output buffer and accumulate total
            total += y[i] = std::exp(x[i]);
        }

        // Second pass over values to normalize and handle overflow
        for(std::size_t i = 0; i < N; ++i) {
            // Overflow handling
            if(std::isinf(total)) {
                // In case of an overflow, distribute equal weight to all
                // occurrences of the maximum value, such that the weights still
                // sum to one.
                y[i] = x[i] == max_value ? 1.0f / float(max_count) : 0.0f;
            } else {
                // In case of no overflow, normalize the exponential values by
                // the accumulated total
                y[i] = y[i] / total;
            }
        }

        // Return the stack-allocated results by copy
        return y;
    }

// Softmax normalizes a vector of N floats
template<int Width, std::size_t N>
    std::array<ap_uint<Width>, N> softmax(
        const std::array<ap_uint<Width>, N> &x,
        // Note: Assumes inputs and outputs to cover the whole range of 0.0 to
        // 1.0 mapped to the 0 to 2^Width - 1 integer range
        const float iscale = 1.0f / ((ap_uint<Width + 1>{1} << Width) - 1),
        const float oscale = 1.0f / ((ap_uint<Width + 1>{1} << Width) - 1),
        const float bias = 0.0) {
        // Allocate a result vector of the same size on the stack
        // @formatter:off
        std::array<float, N> tmp{}; std::array<ap_uint<Width>, N> y{};
        // @formatter:on
        // Track the total, maximum value and the number of occurrences of the
        // maximum value for overflow handling
        // @formatter:off
        float total = 0.0, max_value = -INFINITY; std::size_t max_count = 0;
        // @formatter:on

        // First pass over the input values to compute exp(x) and track maximum
        for(std::size_t i = 0; i < N; ++i) {
            // Keep track of the maximum value encountered
            if(max_value < x[i] || max_count == 0) {
                // New maximum, occurred once
                max_value = x[i];
                max_count = 1;
            } else if(max_value == x[i]) {
                // Got the old maximum again
                max_count++;
            }
            // Compute exp(x) to output buffer and accumulate total
            total += tmp[i] = std::exp(iscale * float(x[i]));
        }

        // Second pass over values to normalize and handle overflow
        for(std::size_t i = 0; i < N; ++i) {
            // Overflow handling
            if(std::isinf(total)) {
                // In case of an overflow, distribute equal weight to all
                // occurrences of the maximum value, such that the weights still
                // sum to one.
                tmp[i] = x[i] == max_value ? 1.0f / float(max_count) : 0.0f;
            } else {
                // In case of no overflow, normalize the exponential values by
                // the accumulated total
                tmp[i] = tmp[i] / total;
            }
            // Shift and scale the output back to integer representation
            y[i] = std::round((tmp[i] - bias) / oscale);
        }

        // Return the stack-allocated results by copy
        return y;
    }

// Softmax normalizes each row of the matrix
template<std::size_t M, std::size_t N>
    Matrix<float, M, N> softmax(const Matrix<float, M, N> &xs) {
        // Allocate the matrix on the stack
        Matrix<float, M, N> ys;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            // Softmax normalize the row
            ys[i] = softmax(xs[i]);
        }
        // Return the matrix from the stack by copy
        return ys;
    }

// Softmax normalizes each row of the matrix
template<int Width, std::size_t M, std::size_t N>
    Matrix<ap_uint<Width>, M, N> softmax(
        const Matrix<ap_uint<Width>, M, N> &xs,
        // Note: Assumes inputs and outputs to cover the whole range of 0.0 to
        // 1.0 mapped to the 0 to 2^Width - 1 integer range
        const float iscale = 1.0f / ((ap_uint<Width + 1>{1} << Width) - 1),
        const float oscale = 1.0f / ((ap_uint<Width + 1>{1} << Width) - 1),
        const float bias = 0.0) {
        // Allocate the matrix on the stack
        Matrix<ap_uint<Width>, M, N> ys;
        // Iterate the indices in row-major order
        for(unsigned i = 0; i < M; ++i) {
            // Softmax normalize the row
            ys[i] = softmax(xs[i], iscale, oscale, bias);
        }
        // Return the matrix from the stack by copy
        return ys;
    }

// Quantizes a matrix to integer representation using Width bits
template<std::size_t Width, std::size_t M, std::size_t N>
    struct Quantized {
        // Integer representation of the inputs using Width bits
        Matrix<ap_uint<Width>, M, N> z{};
        // Scale factor and bias for mapping the range of min to max to the
        // range of 0 to (2^Width - 1).
        float scale, bias;

        // Quantizes a float matrix
        explicit Quantized(
            const Matrix<float, M, N> &x, const float scale, const float bias)
            : scale{scale}, bias{bias} {
            // Iterate the indices in row-major order
            for(unsigned i = 0; i < M; ++i) {
                for(unsigned j = 0; j < N; ++j) {
                    // Quantize the element
                    z[i][j] = std::round((x[i][j] - bias) / scale);
                }
            }
        }

        // Quantizes a float matrix
        explicit Quantized(const Matrix<float, M, N> &x) {
            // Find minimum and maximum of the input matrix
            float min = +INFINITY, max = -INFINITY;
            // Iterate the indices in row-major order
            for(unsigned i = 0; i < M; ++i) {
                for(unsigned j = 0; j < N; ++j) {
                    // Update minimum if new value is smaller
                    min = x[i][j] < min ? x[i][j] : min;
                    // Update maximum if new value is larger
                    max = x[i][j] > max ? x[i][j] : max;
                }
            }

            // 2^Width
            const auto n = (ap_uint<Width + 1>{1} << Width);
            // Scale factor and bias for mapping the range of min to max to the
            // range of 0 to (2^Width - 1).
            scale = (max - min) / float((n - 1)), bias = min;

            // Iterate the indices in row-major order
            for(unsigned i = 0; i < M; ++i) {
                for(unsigned j = 0; j < N; ++j) {
                    // Quantize the element
                    z[i][j] = std::round((x[i][j] - bias) / scale);
                }
            }
        }

        // De-quantizes to float representation
        auto dequantize() const {
            // Float representation
            Matrix<float, M, N> x{};
            // Iterate the indices in row-major order
            for(unsigned i = 0; i < M; ++i) {
                for(unsigned j = 0; j < N; ++j) {
                    // Quantize the element
                    x[i][j] = scale * z[i][j] + bias;
                }
            }
            // Return the de-quantized representation
            return x;
        }
    };

// Quantizes matrix with Type/Size deduction
template<class Type, std::size_t M, std::size_t N>
    auto quantize(const Matrix<float, M, N> &matrix) {
        return Quantized<Type::width, M, N>{matrix};
    }

// Gets the number of possible values of a datatype
template<class Type>
    constexpr std::size_t get_num_possible_values = 0;

// Specialize for ap_uint types
template<int Width>
    constexpr std::size_t get_num_possible_values<ap_uint<Width>> =
        ap_uint<Width + 1>{1} << Width;

// Specialize for ap_int types
template<int Width>
    constexpr std::size_t get_num_possible_values<ap_int<Width>> =
        ap_uint<Width + 1>{1} << Width;

// FINN HLSLIB activation functions: e.g. pass-through and thresholds
#include <activations.hpp>
// Numeric limits of arbitrary precision datatypes
#include "limits.hpp"

// Generates quantized identity function thresholds activations of IType to
// OType
template<std::size_t NF, std::size_t PE, class IType, class OType>
    auto make_identity_thresholds() {
        // The number of thresholds is determined by the range of the output
        // datatype
        constexpr std::size_t steps = get_num_possible_values<OType> - 1;

        // Range of input and output type values
        auto irange = float((max<IType> - min<IType>));
        auto orange = float((max<OType> - min<OType>));
        // The scale, or step size, is determined by the ratio between input
        // and output range
        auto scale = irange / orange;

        // Create a thresholds activation function instance
        ThresholdsActivation<NF, PE, steps, IType, OType> thresholds;

        // Iterate all three dimensions of the thresholds: Thresholds are folded
        for(std::size_t nf = 0; nf < NF; ++nf) {
            for(std::size_t pe = 0; pe < PE; ++pe) {
                for(std::size_t i = 0; i < steps; ++i) {
                    thresholds.m_thresholds[pe][nf][i] =
                        scale * float(i) + min<IType>;
                }
            }
        }

        // Return the prepared thresholds activation function
        return thresholds;
    }

// Activation function quantizing floating-point values via scale and bias
// parameters and rounding
template<class Type>
    class QuantActivation : public Activation<float, Type> {
        // Scale factor to be used for quantizing float values
        float scale;
        // Bias to be added to each value
        float bias;

    public:
        // Initializes the quantizer from scale and bias parameters
        explicit QuantActivation(const float scale, const float bias)
            : scale{scale}, bias{bias} {}

        // Applies activation function to a single element
        Type activate(unsigned const, unsigned const, float const &accu) const {
#pragma HLS inline
            // Quantize, round and implicitly type-cast from float to Type
            return std::round((accu - bias) / scale);
        }
    };

#endif //ATTENTION_HLSLIB_UTILS_HPP
