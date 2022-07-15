# Multidimensional arrays tutorial   {#tutorial_NDArrays}

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
architecture.  This tutorial walks through the options available in TNL.

## Dynamic N-dimensional array

Dynamic N-dimensional arrays are objects of the \ref TNL::Containers::NDArray class. It has several
template parameters, four of them are the most important:

- `Value` specifies the type of values stored in the array
- `SizesHolder` specifies the dimension and static sizes of the array
- `Permutation` specifies the layout of the array in memory
- `Device` specifies the \ref TNL::Devices "device" that will be used for running operations on the
  array

TODO

## Sliced dynamic N-dimensional array

TODO

## Static N-dimensional array

TODO

## Distributed N-dimensional array

TODO
