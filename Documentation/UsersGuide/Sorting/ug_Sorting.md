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
