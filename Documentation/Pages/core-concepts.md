# Core concepts

TNL is based on the following core concepts:

1. _Core concepts_.
  The main concepts used in TNL are the _memory space_, which represents the
  part of memory where given data is allocated, and the _execution model_,
  which represents the way how given (typically parallel) algorithm is executed.
  For example, data can be allocated in the main system memory, in the GPU
  memory, or using the CUDA Unified Memory which can be accessed from the host
  as well as from the GPU. On the other hand, algorithms can be executed using
  either the host CPU or an accelerator (GPU), and for each there are many ways
  to manage parallel execution. The usage of memory spaces is abstracted with
  \ref TNL::Allocators "allocators" and the execution model is represented by
  \ref TNL::Devices "devices".
   1. \ref TNL::Allocators "Allocators"
      - Allocator handles memory allocation and deallocation.
      - TNL allocators are fully compatible with the
        [standard C++ concept](https://en.cppreference.com/w/cpp/named_req/Allocator)
      - Multiple allocators can correspond to the same "memory space".
   2. \ref TNL::Devices "Devices"
      (TODO: rename to `Executor` or something like that)
      - Device is responsible for the execution of algorithms in a specific way.
      - Algorithms can be specialized by the `Device` template parameter.
2. \ref TNL::Algorithms "Algorithms"
   - Basic (container-free) algorithms specialized by `Device`/`Executor`.
   - `parallelFor`, `reduce`, `MultiReduction`, `sort`, ...
3. \ref TNL::Containers "Containers"
    TNL provides generic containers such as array, multidimensional array or array
    views, which abstract data management and execution of common operations on
    different hardware architectures.
   - Classes for general data structures.
     (TODO: alternatively use "Dense" and "Sparse", because a dense matrix can
     be an extended alias for 2D array)
   - `Array`, `Vector`, `NDArray`, ...
4. Views
   - Views wrap only a raw pointer to data and some metadata (such as the array
     size), they do not do allocation and deallocation of the data. Hence, views
     have a fixed size which cannot be changed.
   - Views have a copy-constructor which does a shallow copy. As a result, views
     can be passed-by-value to CUDA kernels or captured-by-value by device
     lambda functions.
   - Views have a copy-assignment operator which does a deep copy.
   - Views have all other methods present in the relevant container (data
     structure).


TODO: formalize the concepts involving lambda functions (e.g. in `reduce`)


## Programming principles

TNL follows common programming principles and design patterns to maintain a
comprehensible and efficient code base. We highlight some principles with
respect to the support for different compute architectures:

- CUDA kernels should not operate with needlessly extensive objects, e.g.
  objects which include smart pointers, because this wastes the device
  registers.
- CUDA kernels should not operate with "distributed" objects – they should
  operate only with the "local parts" of the distributed objects. MPI support is
  a higher layer than CUDA support and distributed objects generally contain
  attributes which should not be needed by CUDA kernels.
- Smart pointers should be cached if appropriate in order to avoid repeated
  memory allocations and copies.
