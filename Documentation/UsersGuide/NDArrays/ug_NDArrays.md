# Multidimensional arrays   {#ug_NDArrays}

[TOC]

## Introduction

Many algorithms in scientific computing work with coefficients indexed by three, four or even more
indices and multidimensional arrays are a natural data structure for representing such values in
the computer memory.  Since the C++ language supports only one-dimensional arrays natively,
multidimensional arrays have to be implemented explicitly (e.g., in a library) based on a mapping
of multidimensional data to an internal one-dimensional array.

An interesting problem is how to choose the mapping from the multidimensional index space into the
one-dimensional index space.  Even for two-dimensional arrays (i.e., matrices) it can be argued
whether they should be stored in the _row-major_ format, where rows are stored as 1D arrays, or in
the _column-major_ format, where columns are stored as 1D arrays.  The optimal choice depends on
the operations that we want to do with the data, as well as on the hardware architecture that will
process the data.  For example, the row-major format is suitable for algorithms processing a matrix
row by row on the CPU, while for GPU algorithms also processing a matrix row by row the
column-major format is more appropriate.  For three- and more-dimensional arrays there are even
more combinations of possible array orderings.

For these reasons, we developed several data structures which allow to configure the indexing of
multidimensional data and thus optimize the data structure for given algorithm and hardware
architecture.  This chapter walks through the options available in TNL.

## Dynamic N-dimensional array

Dynamic N-dimensional arrays are objects of the \ref TNL::Containers::NDArray class. It has several
template parameters, four of them are the most important:

- `Value` specifies the type of values stored in the array
- `SizesHolder` specifies the dimension and static sizes of the array
- `Permutation` specifies the layout of the array in memory
- `Device` specifies the \ref TNL::Devices "device" that will be used for running operations on the
  array

The sizes of the array are specified using \ref TNL::Containers::SizesHolder as follows. The first
template parameter specifies the `IndexType` used for indexing (which will be used even by
`NDArray`) and the following sequence of integers specifies the static sizes of the array along the
individual axes. For a dynamic array, there must be at least one axis with zero static size.

For example, here we specify all axes as dynamic:

```{.cpp}
using My2DSizes = TNL::Containers::SizesHolder< int, 0, 0 >,
using My3DSizes = TNL::Containers::SizesHolder< int, 0, 0, 0 >,
```

The layout of the array in memory can be configured using a permutation specified using \ref
std::index_sequence. By default, the `NDArray` class uses an identity permutation. Users can use
this, for example, to switch between _row-major_ (`std::index_sequence<0, 1>`) and _column-major_
(`std::index_sequence<1, 0>`) layout of a 2D array, or to specify an arbitrary layout for a
higher-dimensional array.

When all template parameters are known, the `NDArray` can be instantiated:

\snippet NDArrayExample_RowMajorArray.cpp instantiation

Then, the dynamic sizes can be set by calling \ref TNL::Containers::NDArray::setSizes "setSizes".
Here we set the sizes to create a \f$ 3 \times 4 \f$ array, where the first size corresponds to the
0-axis and the second size to the 1-axis:

\snippet NDArrayExample_RowMajorArray.cpp allocation

Elements of the array can be accessed using the \ref TNL::Containers::NDArray::operator()()
"operator()":

\snippet NDArrayExample_RowMajorArray.cpp initialization

Note that regardless of the permutation, the order of indices in the `operator()` logically
corresponds to the order of sizes in the `setSizes` call (and in the `SizesHolder<...>`
specification). This allows the programmer to change the layout of the array simply by setting a
different permutation in the template parameters without having to rewrite the order of indices
everywhere the array is used.

To examine the layout, we can print the underlying 1D array:

\snippet NDArrayExample_RowMajorArray.cpp output

Output:

\include NDArrayExample_RowMajorArray.out

Notice that the identity permutation works such that indices along the last axis change the fastest
and indices along the first axis change the slowest.

In the following example, we show the effect of three different permutations on a \f$ 3 \times 3
\times 3 \f$ array.

\includelineno NDArrayExample_3Dpermutations.cpp

Output:

\include NDArrayExample_3Dpermutations.out

The \ref TNL::Containers::NDArray class also has a corresponding view, \ref
TNL::Containers::NDArrayView, which has the same semantics as the \ref TNL::Containers::ArrayView
for the basic one-dimensional array.

See the reference documentation for the overview of all operations available for N-dimensional
arrays.

### Different permutations depending on the device

In practice, the optimal permutation for given application often depends on the hardware
architecture used for the computations. For example, one permutation may be optimal for GPU
computations and a different permutation may be optimal for CPU computations. Hence, it may be
desirable to create a generic data structure which uses different permutations for different
devices. This may be achieved, for example, by defining the following template alias:

```{.cpp}
template< typename Value,
          typename SizesHolder,
          typename HostPermutation,
          typename CudaPermutation,
          typename Device >
using NDArray = TNL::Containers::NDArray< Value,
                                          SizesHolder,
                                          std::conditional_t< std::is_same< Device, TNL::Devices::Cuda >::value,
                                                              CudaPermutation,
                                                              HostPermutation >,
                                          Device >;
```

## Static N-dimensional array

Static N-dimensional arrays can be created using the \ref TNL::Containers::StaticNDArray class. It
has the same base class as \ref TNL::Containers::NDArray, but it exposes different template
parameters and uses \ref TNL::Containers::StaticArray instead of \ref TNL::Containers::Array for
the underlying storage. Hence, there is no allocator (the array lives on the stack) and the device
is always \ref TNL::Devices::Sequential.

Static sizes are specified as positive integers in the sequence passed to \ref
TNL::Containers::SizesHolder. For example, to create a static row-major \f$ 3 \times 4 \f$ array:

```{.cpp}
using StaticRowMajorArray = StaticNDArray< int,  // Value
                                           SizesHolder< int, 3, 4 >,     // SizesHolder
                                           std::index_sequence< 0, 1 >,  // Permutation
StaticRowMajorArray a;
```

As it is a static array, the \ref TNL::Containers::StaticNDArray::setSizes "setSizes" method is not
called and the elements can be accessed using \ref TNL::Containers::StaticNDArray::operator()()
"operator()".

Note that static and dynamic sizes can be combined. For example, we may want to create a 2D array
which has a constant size of 3 rows, but the number of columns is a priori unknown and has to be
set at run-time. This can be achieved using \ref TNL::Containers::NDArray as follows:

\includelineno NDArrayExample_HalfStaticArray.cpp

Notice that the `SizesHolder<...>` specification contains positive values for static sizes and
zeros for dynamic sizes, whereas the `setSizes(...)` call has zeros for static sizes and positive
values for dynamic sizes.

Output:

\include NDArrayExample_HalfStaticArray.out

## Sliced dynamic N-dimensional array

The \ref TNL::Containers::SlicedNDArray allows to create _sliced_ N-dimensional arrays. The idea of
slicing is that one particular axis is not indexed contiguously, but after a given number of
elements the indexing continues with the other dimensions. The slicing is specified using the \ref
TNL::Containers::SliceInfo class which is specified in the template parameters of `SlicedNDArray`
after `Permutation`.

In the following example, we create a two-dimensional array with a static number of rows (3),
dynamic number of columns (10), and slicing size of 4 along the 1-axis. Hence, the array is divided
into 3 slices, each containing 4 columns (the allocated size of the array is aligned to 12
columns).

\includelineno NDArrayExample_SlicedArray.cpp

Output:

\include NDArrayExample_SlicedArray.out

Notice how the numbering goes to the second row (starting with the value 10) after numbering 4
elements of the first row, then to the third row (starting with the value 20) after numbering 4
elements of the second row, and back to the first row (starting with the value 4) after numbering 4
elements of the last row. Compare the output with the previous example using a non-sliced array
with the same sizes.

## Distributed N-dimensional array

\ref TNL::Containers::DistributedNDArray "DistributedNDArray" extends \ref TNL::Containers::NDArray "NDArray" in a similar way that \ref TNL::Containers::DistributedArray "DistributedArray" extends \ref TNL::Containers::Array "Array".
Because it is N-dimensional, there is more freedom in the way it can be decomposed.
The following example creates a 2D array oriented in a column-major layout and decomposed along the vertical axis (y-axis) with one layer of overlaps (ghost regions).
The elements in ghost regions get initialized to `-1` and later they are synchronized using \ref TNL::Containers::DistributedNDArraySynchronizer.
Note that the synchronization is periodic by default (i.e., the first rank synchronizes values with the last rank).

\include Containers/DistributedNDArrayExample.cpp

Possible output:

\include DistributedNDArrayExample.out
