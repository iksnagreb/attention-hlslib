# Attention Operator Library based on Vitis HLS and FINN

## Dependencies

The library builds on the [FINN HLS library](https://github.com/Xilinx/finn-hlslib) and requires a
version providing the tensor interface (`tensor.hpp`): All streams of the operators carry
`tensor<Elem, N>` (or `tensor<Elem, N, M>` for tiles) instead of flat `ap_uint<>` bit vectors. The
tensors are backed by a single `ap_uint<>` with the first element at the least significant bits,
i.e., the bit layout of the streams is the same as that of the flat bit vectors used before. At the
top-level function, declare the stream arguments like the FINN HLS library does for its own
operators, with `#pragma HLS aggregate variable=<port> compact=bit` next to the `axis` interface
pragma.
