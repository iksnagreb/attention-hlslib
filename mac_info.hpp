#ifndef MAC_INFO_HPP
#define MAC_INFO_HPP

// Recursive template wrapper to get the accumulator width and datatype of a
// MAC unit to fit the biggest possible result of accumulating Num Lhs x Rhs
// multiplications
template<unsigned Num, class Lhs, class Rhs>
    struct MACInfo;

// Template wrapper to get the accumulator width and datatype of a MAC unit to
// fit the biggest possible result of accumulating Num Lhs x Rhs multiplications
template<class Lhs, class Rhs>
    struct MACInfo<1, Lhs, Rhs> {
        // Base case to stop the recursion when only one multiplication remains
        using AccType = decltype(Lhs{} * Rhs{});

        // Width of the accumulator type
        static constexpr unsigned acc_width = AccType::width;
    };

// Recursive template wrapper to get the accumulator width and datatype of a
// MAC unit to fit the biggest possible result of accumulating Num Lhs x Rhs
// multiplications
template<unsigned Num, class Lhs, class Rhs>
    struct MACInfo {
        // Accumulating a zero or negative number of products does not make any
        // sense
        static_assert(
            Num > 0, "Accumulation of zero or negative number of elements."
        );

        // Infer the datatype by recursively expanding along the accumulation
        using AccType = decltype(
            Lhs{} * Rhs{} + typename MACInfo<Num-1, Lhs, Rhs>::AccType{}
        );

        // Width of the accumulator type
        static constexpr unsigned acc_width = AccType::width;
    };

#endif // MAC_INFO_HPP
