#ifndef ATTENTION_HLSLIB_MATRIX_HPP
#define ATTENTION_HLSLIB_MATRIX_HPP

// Use nested std::array to represent a matrix
#include <array>
// std::rand
#include <cstdlib>
// std::printf
#include <cstdio>

// Matrix of shape M x N of Type
template<class Type, std::size_t M, std::size_t N>
    using Matrix = std::array<std::array<Type, N>, M>;

// Creates a randomly filled matrix of shape M x N of Type
template<class Type, std::size_t M, std::size_t N>
    Matrix<Type, M, N> rand_matrix() {
        // Allocate the matrix on the stack
        Matrix<Type, M, N> matrix;
        // Iterate the indices in row-major order
        for (unsigned i = 0; i < M; ++i) {
            for (unsigned j = 0; j < N; ++j) {
                // Generate random number in range 0 to 2 ^ Type::width
                matrix[i][j] = std::rand() % (2 << Type::width);
            }
        }
        // Return the matrix from the stack by copy
        return matrix;
    }

// Format and print a matrix to stdout
template<class Type, std::size_t M, std::size_t N>
    void print_matrix(
        const Matrix<Type, M, N> &matrix, const char *f = "%8x ") {
        // Iterate the indices in row-major order
        for (unsigned i = 0; i < M; ++i) {
            for (unsigned j = 0; j < N; ++j) {
                // Formatted print of the matrix element
                std::printf(f, matrix[i][j]);
            }
            // Insert a line break
            std::printf("\n");
        }
    }

// Multiplies two matrices
template<class Lhs, class Rhs, std::size_t M, std::size_t N, std::size_t L>
    auto matmul(const Matrix<Lhs, M, N> &lhs, const Matrix<Rhs, N, L> &rhs) {
        // Allocate the result matrix on the stack
        Matrix<decltype(Lhs{} * Rhs{}), M, L> result;
        // Iterate the indices in row-major order
        for (unsigned i = 0; i < M; ++i) {
            for (unsigned j = 0; j < L; ++j) {
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
        for (unsigned i = 0; i < M; ++i) {
            for (unsigned j = 0; j < N; ++j) {
                // If any pair of elements is not equal, the matrices differ
                if(lhs[i][j] != rhs[i][j]) {
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
            explicit RowMajorMatrixStreamer(const Matrix<Type, M, N> &in) {
                // Iterate the indices in row-major order
                for (unsigned i = 0; i < M; ++i) {
                    for (unsigned j = 0; j < N; ++j) {
                        // Insert element into the stream
                        out.write(in[i][j]);
                    }
                }
            }

        // Extracts a matrix from a row-major oder stream into the buffer
        template<std::size_t M, std::size_t N>
            static void read(hls::stream<Type> &in, Matrix<Type, M, N> &out) {
                // Iterate the indices in row-major order
                for (unsigned i = 0; i < M; ++i) {
                    for (unsigned j = 0; j < N; ++j) {
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
            explicit ColMajorMatrixStreamer(const Matrix<Type, M, N> &in) {
                // Iterate the indices in column-major order
                for (unsigned j = 0; j < N; ++j) {
                    for (unsigned i = 0; i < M; ++i) {
                        // Insert element into the stream
                        out.write(in[i][j]);
                    }
                }
            }

        // Extracts a matrix from a column-major oder stream into the buffer
        template<std::size_t M, std::size_t N>
            static void read(hls::stream<Type> &in, Matrix<Type, M, N> &out) {
                // Iterate the indices in column-major order
                for (unsigned j = 0; j < N; ++j) {
                    for (unsigned i = 0; i < M; ++i) {
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
// Flattening of buffers to bit vectors
#include "flatten.hpp"

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
                for (std::size_t i = 0; i < N; ++i) {
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
                auto buffer = Slice<ap_uint<Type::width / N>>{}(in.read());
                // Collect the next N elements into the buffer
                for(std::size_t i = 0; i < N; ++i) {
                    // Write the next element into the output stream
                    out.write(buffer(i, 0));
                }
            }
        }
    };

#endif //ATTENTION_HLSLIB_MATRIX_HPP
