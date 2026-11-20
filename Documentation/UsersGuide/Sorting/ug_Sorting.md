# Sorting  {#ug_Sorting}

[TOC]

## Introduction

TNL offers several different parallel algorithms for sorting of arrays (or
vectors) and also sorting based on user defined swapping. The latter is more
general but also less efficient.

## Sorting of arrays and vectors

The sorting of arrays and vectors is accessible via the following functions:

* \ref TNL::Algorithms::ascendingSort for sorting elements of array in
  ascending order,
* \ref TNL::Algorithms::descendingSort for sorting elements of array in
  descending order,
* \ref TNL::Algorithms::sort for sorting with user defined ordering.

The following example demonstrates the use of ascending and descending sort:

\includelineno SortingExample.cpp

Here we create an array with random sequence of integers using the
\ref TNL::Algorithms::parallelFor "parallelFor" function and then we sort the
array in ascending order using \ref TNL::Algorithms::ascendingSort "ascendingSort"
and descending order using the \ref TNL::Algorithms::descendingSort "descendingSort".

The result looks as follows:

\include SortingExample.out

How to achieve the same result with user defined ordering is demonstrated by the
following example:

\includelineno SortingExample2.cpp

The result looks as follows:

\include SortingExample2.out

The same way, one can sort also \ref TNL::Containers::ArrayView,
\ref TNL::Containers::Vector and \ref TNL::Containers::VectorView.

## Sorting with user-defined swapping

\includelineno SortingExample3.cpp

In this example, we fill array `array` with random numbers and array `index`
with numbers equal to position of an element in the array. We want to sort the
array `array` and permute the `index` array correspondingly. This is achieved
by calling a variant of the `sort` function, which does not accept an array-like
data structure, but only range of indexes and two lambda functions. The first
lambda function defines the ordering of the elements by comparing elements of
array `array`. The second lambda function is responsible for swapping elements.
Note that we do not swap only elements of array `array`, but also `index` array.

The result looks as follows:

\include SortingExample3.out

## Three-way partitioning

Sometimes sorting the whole array is not necessary. When only the elements
ordered before, equivalent to, and ordered after a given *pivot* value need to
be separated, TNL provides the \ref TNL::Algorithms::pivotPartition
"pivotPartition" function with two overloads:

* an in-place overload `pivotPartition( data, pivot, compare )`, which
  rearranges the elements of `data`,
* an out-of-place overload `pivotPartition( input, output, pivot, compare )`,
  which writes the partitioned elements to `output` and leaves `input`
  unchanged. The input and output ranges must not overlap.

Both return a \ref TNL::Algorithms::PivotPartitionResult "PivotPartitionResult"
structure which supports structured bindings. Its `lessEnd` and
`greaterBegin` fields are boundaries relative to the beginning of the output
range, so the three buckets are `[0, lessEnd)`, `[lessEnd, greaterBegin)` and
`[greaterBegin, size)`.

Equivalence with respect to the pivot is defined by two comparator calls: an
element is equivalent to the pivot if neither `compare( value, pivot )` nor
`compare( pivot, value )` is true (`operator==` is not used). The comparator
must impose a strict weak ordering.

Every input element is preserved exactly once - the output is a permutation
of the input. In particular, pivot-equivalent elements are moved or copied
rather than reconstructed from the pivot, so comparison-equivalent elements
may carry distinct payloads. This is demonstrated by the following example,
where several elements share the comparison key but each has a unique id:

\includelineno pivotPartitionExample.cpp

The result looks as follows:

\include pivotPartitionExample.out

Note that the operation is not stable - the order of the elements within each
bucket is unspecified (the example shows the order produced by the host
implementation).

The complexity is `O( N )` for all overloads. The auxiliary storage depends
on the device and the overload: the host in-place overload needs only `O( 1 )`
auxiliary storage, the GPU out-of-place overload needs `O( B )` auxiliary
storage for the per-block counts, where `B = ceil( N / 1024 )`, and the GPU
in-place overload partitions into a scratch array of `N` elements and copies
the result back, i.e. it needs `O( N + B )` auxiliary storage in total.
