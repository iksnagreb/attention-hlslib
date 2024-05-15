#ifndef ATTENTION_HLSLIB_BIND_STORAGE_HPP
#define ATTENTION_HLSLIB_BIND_STORAGE_HPP

// Namespace to organize resource implementation options
//  See UG1399 for details
namespace Resource {
// Tag type for selecting distributed RAM based implementation of storage
//  Corresponds to impl=LUTRAM on bind_storage pragma, see UG1399
struct LUTRAM {
};

// Tag type for selecting block RAM based implementation of storage
//  Corresponds to impl=BRAM on bind_storage pragma, see UG1399
struct BRAM {
};

// Tag type for selecting block RAM based implementation of storage
//  Corresponds to impl=URAM on bind_storage pragma, see UG1399
struct URAM {
};

// Tag type for selecting shift register logic based implementation of storage
//  Corresponds to impl=SRL on bind_storage pragma, see UG1399
struct SRL {
};

// Tag type for selecting implementation of storage automatically by Vitis
//  Corresponds to impl=AUTO on bind_storage pragma, see UG1399
struct AUTO {
};

// Tage type selecting implementation to be selected by Vivado
//  Corresponds to impl=MEMORY on bind_storage pragma, see UG1399
struct MEMORY {
};
}

// Namespace to organize storage type options
//  See UG1399 for details
namespace Storage {
// Tag type selecting a FIFO storage
struct FIFO {
};

// Tag type selecting a single-port RAM storage
struct RAM_1P {
};

// Tag type selecting a RAM with 1 write port and N read ports, using N banks
// internally
struct RAM_1WNR {
};

// Tag type selecting a dual-port RAM that allows read operations on one port
// and both read and write operations on the other port.
struct RAM_2P {
};

// Tag type selecting a dual-port RAM that allows read operations on one port
// and write operations on the other port.
struct RAM_S2P {
};

// Tag type selecting a true dual-port RAM with support for both read and write
// on both ports.
struct RAM_T2P {
};

// Tag type selecting a single-port ROM
struct ROM_1P {
};

// Tag type selecting a dual-port ROM
struct ROM_2P {
};

// Tag type selecting a multi-port ROM
struct ROM_NP {
};
}

// Adds bind_storage pragmas to a variable declaration of Type, depending on
// template argument for selecting the resource type the storage should be bound
// to.
template<class Type, class Storage, class Resource>
    struct BindStorage;

// Utility for converting macro arguments to macro strings
#define STRINGIFY(a) #a

// Macro for generating template specializations of BindStorage
#define BIND_STORAGE(STORAGE, RESOURCE) \
    template<class Type>                                                       \
        struct BindStorage<Type, Storage::STORAGE, Resource::RESOURCE> {       \
            Type var;                                                          \
            BindStorage() {                                                    \
_Pragma(STRINGIFY(HLS bind_storage variable=var type=STORAGE impl=RESOURCE))   \
            }                                                                  \
            operator Type &() {                                                \
                return var;                                                    \
            }                                                                  \
            operator const Type &() const {                                    \
                return var;                                                    \
            }                                                                  \
        }

// The above macro effectively expands to the following class template
// specialization, for example to specify an AUTO FIFO storage:

// Specialization binding the storage of Type to be implemented as a FIFO where
// Vitis HLS decides whether to implement as LUTRAM, BRAM, URAM or SRL
template<class Type>
    struct BindStorage<Type, Storage::FIFO, Resource::AUTO> {
        // Wrap an object of the specified Type as a publicly exposed member of
        // the struct
        Type var;

        // Constructor is used to apply the pragma, as HLS pragmas may only be
        // at function scope. The constructor is closest we can get to "the body
        // of the function where the variable is defined" according to UG1399.
        BindStorage() {
// Set the pragma for the storage to be implemented as a FIFO where  Vitis HLS
// decides whether to implement as LUTRAM, BRAM, URAM or SRL
#pragma HLS bind_storage variable=var type=FIFO impl=AUTO
        }

        // Implicit conversion to a reference of the underlying storage object
        // of Type: This should make BinsStorage behave *almost* as Type
        operator Type &() {  // NOLINT: Intentionally implicit
            return var;
        }

        // Implicit conversion to a reference of the underlying storage object
        // of Type: This should make BinsStorage behave *almost* as Type
        operator const Type &() const {  // NOLINT: Intentionally implicit
            return var;
        }
    };

// Generate all BindStorage specializations for FIFOs with all allowed
// implementations according to UG1399
//BIND_STORAGE(FIFO, AUTO); // Manually specified as example above
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
//BIND_STORAGE(RAM_T2P, LUTRAM);
BIND_STORAGE(RAM_T2P, URAM);

// Generate all BindStorage specializations for ROM_1P with all allowed
// implementations according to UG1399
BIND_STORAGE(ROM_1P, AUTO);
BIND_STORAGE(ROM_1P, BRAM);
BIND_STORAGE(ROM_1P, LUTRAM);
//BIND_STORAGE(ROM_1P, URAM);

// Generate all BindStorage specializations for ROM_2P with all allowed
// implementations according to UG1399
BIND_STORAGE(ROM_2P, AUTO);
BIND_STORAGE(ROM_2P, BRAM);
BIND_STORAGE(ROM_2P, LUTRAM);
//BIND_STORAGE(ROM_2P, URAM);

// Generate all BindStorage specializations for ROM_NP with all allowed
// implementations according to UG1399
BIND_STORAGE(ROM_NP, AUTO);
BIND_STORAGE(ROM_NP, BRAM);
BIND_STORAGE(ROM_NP, LUTRAM);
//BIND_STORAGE(ROM_NP, URAM);

#endif //ATTENTION_HLSLIB_BIND_STORAGE_HPP
