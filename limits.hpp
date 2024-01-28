#ifndef ATTENTION_HLSLIB_LIMITS_HPP
#define ATTENTION_HLSLIB_LIMITS_HPP

// Numeric limits of standard types
#include <limits>
// HLS arbitrary precision types: ap_uint class template
#include <ap_int.h>

// Minimum value of a generic datatype
template<class Type>
    constexpr Type min = std::numeric_limits<Type>::min();

// Maximum value of a generic datatype
template<class Type>
    constexpr Type max = std::numeric_limits<Type>::max();

// Minimum value of HLS unsigned arbitrary precision types
template<int Width>
    constexpr auto min<ap_uint<Width>> = 0;

// Maximum value of HLS unsigned arbitrary precision types
template<int Width>
    constexpr auto max<ap_uint<Width>> = (ap_uint<Width + 1>{1} << Width) - 1;

// Minimum value of HLS signed arbitrary precision types
template<int Width>
    constexpr auto min<ap_int<Width>> = -(max<ap_uint<Width - 1>> + 1);

// Maximum value of HLS signed arbitrary precision types
template<int Width>
    constexpr auto max<ap_int<Width>> = max<ap_uint<Width - 1>>;

#endif //ATTENTION_HLSLIB_LIMITS_HPP
