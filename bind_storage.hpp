#ifndef BIND_STORAGE_HPP
#define BIND_STORAGE_HPP

// Storage resource implementation options according to UG1399
enum Resource {
    LUTRAM, BRAM, URAM, SRL, AUTO, MEMORY
};

// Storage type options according to UG1399
enum Storage {
    FIFO, RAM_1P, RAM_1WNR, RAM_2P, RAM_S2P, RAM_T2P, ROM_1P, ROM_2P, ROM_NP
};

// Auxiliary type for adding bind_storage directives to a variable of generic
// type. This auxiliary helps with partial specialization/instantiation of the
// template via the following macro. Use the bind_storage interface function
// template below.
template<Storage Storage, Resource Resource>
    struct BindStorage;

// Utility for converting macro arguments to macro strings used to assemble the
// HLS pragma from its constituents
#define STRINGIFY(a) #a

// Macro for generating template specializations of BindStorage, creating a
// static interface with generic Type which can be applied to a variable.
#define BIND_STORAGE(STORAGE, RESOURCE) \
    template<>                                                                 \
        struct BindStorage<Storage::STORAGE, Resource::RESOURCE> {             \
            template<class Type>                                               \
                constexpr static void bind_storage(Type &&var) {               \
_Pragma(STRINGIFY(HLS bind_storage variable=var type=STORAGE impl=RESOURCE))   \
                }                                                              \
        };

// The above macro effectively expands to the following class template
// specialization, for example to specify an AUTO FIFO storage:

// Specialization binding the storage of Type to be implemented as a FIFO where
// Vitis HLS decides whether to implement as LUTRAM, BRAM, URAM or SRL
template<>
    struct BindStorage<Storage::FIFO, Resource::AUTO> {
        // Static interface is used to apply the pragma, as HLS pragmas may only
        // be at function scope. The interface is generic with respect to the
        // type and value category of the variable which the directive should be
        // applied to.
        template<class Type>
            constexpr static void bind_storage(Type &&var) {
            // Directive for the storage to be implemented as a FIFO where Vitis
            // HLS decides whether to implement as LUTRAM, BRAM, URAM or SRL
#pragma HLS bind_storage variable=var type=FIFO impl=AUTO
        }
    };

// Interface to be used for selecting and inserting bind_storage directives via
// template metaprogramming. Type can be inferred from the input.
template<Storage Storage, Resource Resource, class Type>
    constexpr void bind_storage(Type &&var) {
        BindStorage<Storage, Resource>::bind_storage(var);
    }


// Generate all BindStorage specializations for FIFOs with all allowed
// implementations according to UG1399
// BIND_STORAGE(FIFO, AUTO); // Manually specified as example above
BIND_STORAGE(FIFO, BRAM);
BIND_STORAGE(FIFO, LUTRAM);
BIND_STORAGE(FIFO, URAM);
BIND_STORAGE(FIFO, MEMORY);
BIND_STORAGE(FIFO, SRL);

// Generate all BindStorage specializations for RAM_1P with all allowed
// implementations according to UG1399
BIND_STORAGE(RAM_1P, AUTO);
BIND_STORAGE(RAM_1P, BRAM);
BIND_STORAGE(RAM_1P, URAM);

// Generate all BindStorage specializations for RAM_1WNR with all allowed
// implementations according to UG1399
BIND_STORAGE(RAM_1WNR, AUTO);
BIND_STORAGE(RAM_1WNR, BRAM);
BIND_STORAGE(RAM_1WNR, LUTRAM);
BIND_STORAGE(RAM_1WNR, URAM);

// Generate all BindStorage specializations for RAM_2P with all allowed
// implementations according to UG1399
BIND_STORAGE(RAM_2P, AUTO);
BIND_STORAGE(RAM_2P, BRAM);
BIND_STORAGE(RAM_2P, LUTRAM);
BIND_STORAGE(RAM_2P, URAM);

// Generate all BindStorage specializations for RAM_S2P with all allowed
// implementations according to UG1399
BIND_STORAGE(RAM_S2P, AUTO);
BIND_STORAGE(RAM_S2P, BRAM);
BIND_STORAGE(RAM_S2P, LUTRAM);
BIND_STORAGE(RAM_S2P, URAM);

// Generate all BindStorage specializations for RAM_T2P with all allowed
// implementations according to UG1399
BIND_STORAGE(RAM_T2P, AUTO);
BIND_STORAGE(RAM_T2P, BRAM);
// BIND_STORAGE(RAM_T2P, LUTRAM); // TODO: Not available for some devices?
BIND_STORAGE(RAM_T2P, URAM);

// Generate all BindStorage specializations for ROM_1P with all allowed
// implementations according to UG1399
BIND_STORAGE(ROM_1P, AUTO);
BIND_STORAGE(ROM_1P, BRAM);
BIND_STORAGE(ROM_1P, LUTRAM);
// BIND_STORAGE(ROM_1P, URAM);    // TODO: Not available for some devices?

// Generate all BindStorage specializations for ROM_2P with all allowed
// implementations according to UG1399
BIND_STORAGE(ROM_2P, AUTO);
BIND_STORAGE(ROM_2P, BRAM);
BIND_STORAGE(ROM_2P, LUTRAM);
// BIND_STORAGE(ROM_2P, URAM);    // TODO: Not available for some devices?

// Generate all BindStorage specializations for ROM_NP with all allowed
// implementations according to UG1399
BIND_STORAGE(ROM_NP, AUTO);
BIND_STORAGE(ROM_NP, BRAM);
BIND_STORAGE(ROM_NP, LUTRAM);
// BIND_STORAGE(ROM_NP, URAM);    // TODO: Not available for some devices?

#endif // BIND_STORAGE_HPP
