# Segments  {#ug_Segments}

[TOC]

## Introduction

*Segments* represent a data structure designed for managing multiple *local arrays*
(also referred to as *segments*), which generally have different sizes. All
the local arrays are stored in a single contiguous *global array*. The segments
structure provides a mapping between indices of individual local
arrays and indices within the global array.

Importantly, segments do not store any data themselves. Instead, they serve as
a lightweight abstraction layer that facilitates efficient access to and
operations on groups of linear containers (i.e., local arrays) of variable size.

Using segments, one can perform various parallel operations, such as for-style
traversals, flexible reduction and prefix-sum, sorting or searching across the
individual segments.

A typical example of the use of segments is in implementing different sparse
matrix formats. For instance, a sparse matrix like the following:

\f[
\left(
\begin{array}{ccccc}
    1  &  0  &  2  &  0  &  0 \\
    0  &  0  &  5  &  0  &  0 \\
    3  &  4  &  7  &  9  &  0 \\
    0  &  0  &  0  &  0  & 12 \\
    0  &  0  & 15  & 17  & 20
\end{array}
\right)
\f]

This matrix is usually compressed, meaning that the zero elements are omitted.
The resulting compressed structure looks like:

\f[
\begin{array}{ccccc}
    1  &   2  \\
    5   \\
    3  &   4  &  7 &  9   \\
    12 \\
    15 & 17  & 20
\end{array}
\f]

To retain information about the positions of the non-zero elements, we
also store their column indices, forming a second structure:

\f[
\begin{array}{ccccc}
    0  &   2  \\
    2   \\
    0  &   1  &  2 &  3   \\
    4 \\
    2 & 3  & 4
\end{array}
\f]

Both of these “matrices” (non-zero values and their corresponding column indices)
are typically stored in memory row-wise in contiguous arrays for performance reasons.
For example, the array of values becomes:

\f[
\begin{array}{|cc|c|cccc|c|cc|}
1 & 2 &  5 & 3 & 4 & 7 & 9 & 12 & 15 & 17 & 20
\end{array}
\f]

And the corresponding array of column indices is:

\f[
\begin{array}{|cc|c|cccc|c|cc|}
0 & 2 & 2 & 0 & 1 & 2 & 3 & 4 & 2 & 3 & 4
\end{array}
\f]

What we see above is the so-called [CSR sparse matrix format](
https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)).
This is the most widely used format for storing sparse matrices
and was designed for high performance on CPUs.

However, CSR is not always the most efficient format for sparse
matrix storage and access on GPUs. For this reason, many alternative
formats have been developed to achieve better performance on
parallel architectures. These formats often use a different
layout of matrix elements in memory, and must address the
following challenges:

1. Efficient memory storage of matrix elements to enable:
   - Coalesced memory accesses on GPUs
   - Good spatial locality for effective use of CPU caches
2. Efficient mapping of GPU threads to matrix rows or elements
   for parallel processing.

The TNL library provides support for several sparse matrix
formats implemented as segments. Some of these formats - such as
the Ellpack based formats - use *padding elements* (e.g., padding zeros)
to equalize segment sizes (e.g. compressed matrix row lengths) and
optimize memory access patterns. The following is the list of
currently implemented sparse matrix formats in TNL:

1. [CSR format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format))
   (\ref TNL::Algorithms::Segments::CSR) This is the most widely used
   sparse matrix format. It is simple and efficient, especially on CPUs. Today,
   there also exist optimized GPU kernels for CSR. The following GPU
   kernels are implemented in TNL:
   1. [Scalar](http://mgarland.org/files/papers/nvr-2008-004.pdf) Maps
      one GPU thread for each segment (matrix row).
   2. [Vector](http://mgarland.org/files/papers/nvr-2008-004.pdf) Maps
      one warp of GPU threads for each segment (matrix row).
   3. [Adaptive](https://ieeexplore.ieee.org/document/7397620) Dynamically
      selects the most suitable kernel based on the segment properties.
2. [Ellpack format](http://mgarland.org/files/papers/nvr-2008-004.pdf)
   (\ref TNL::Algorithms::Segments::Ellpack) This format pads all segments
   to the same length, which can be inefficient when there are a few very
   long segments. It offers good memory access especially on GPUs.
3. [SlicedEllpack format](https://link.springer.com/chapter/10.1007/978-3-642-11515-8_10)
   (\ref TNL::Algorithms::Segments::SlicedEllpack) Also known as
   [Row-grouped CSR format](https://arxiv.org/abs/1012.2270) this format groups
   segments into blocks (e.g., of 32 segments). Padding is applied only within
   each group, reducing the performance loss caused by outlier segment lengths.
4. [ChunkedEllpack format](http://geraldine.fjfi.cvut.cz/~oberhuber/data/vyzkum/publikace/12-heller-oberhuber-improved-rgcsr-format.pdf)
   (\ref TNL::Algorithms::Segments::ChunkedEllpack) Similar to SlicedEllpack,
   but allows segments to be split into smaller chunks, enabling more GPU threads
   to work on the same segment concurrently. This improves performance for very
   long segments.
5. [BiEllpack format](https://www.sciencedirect.com/science/article/pii/S0743731514000458?casa_token=2phrEj0Ef1gAAAAA:Lgf6rMBUN6T7TJne6mAgI_CSUJ-jR8jz7Eghdv6L0SJeGm4jfso-x6Wh8zgERk3Si7nFtTAJngg)
   (\ref TNL::Algorithms::Segments::BiEllpack) Is similar to ChunkedEllpack.
   In addition, it sorts segments within each slice according to their length.
   This further improves memory access patterns and computational efficiency
   on parallel architectures.

TODO: Finish sorted segments and add example
Inspired by ()[], TNL also offers *sorted segments*. Here, the segments are first
sorted by their size in descending order. After that they are stored using the underlying segments.


Especially in the case of GPUs, the performance of each format strongly depends
on the distribution of segment sizes. Therefore, we cannot claim that any one of
the above formats consistently outperforms the others. To achieve optimal
performance, it is often necessary to experiment with multiple formats and
select the best one for a given problem. This is why TNL offers a variety
of formats, and additional ones may be added in the future.

The need for such data structures is not limited to sparse matrices.
Segments can be applied in a wide range of problems, including but
 not limited to:

1. [Graphs](https://en.wikipedia.org/wiki/Graph_(discrete_mathematics)) - Each
   segment corresponds to a graph node; elements in a segment represent the node's neighbors.
2. [Unstructured numerical meshes](https://en.wikipedia.org/wiki/Types_of_mesh)
   \- These meshes can be viewed as graphs and handled similarly.
3. [Particle in cell method](https://en.wikipedia.org/wiki/Particle-in-cell)
   \- Each segment represents a cell; the elements are indices of the particles
   in that cell.
4. [K-means clustering](https://en.wikipedia.org/wiki/K-means_clustering)
   \- Each segment corresponds to a cluster; elements in the segment are
   vectors assigned to that cluster.
5. [Hashing](https://arxiv.org/abs/1907.02900) - Segments represent rows in
   a hash table; elements in a segment are entries that collide in the same row.

In general, segments are useful for any problem involving
**data structures consisting of contiguous blocks with irregular sizes**,
where various operations are performed per block.
The term segments is derived from segmented parallel reduction and
[segmented scan (prefix-sum)](https://en.wikipedia.org/wiki/Segmented_scan).

## Segments setup

Segments are defined solely by the sizes of the individual segments.
The following example shows how to create them:

\snippet Algorithms/Segments/printSegmentsExample-1.cpp segments setup

We use the constructor with an initializer list, where each element of
the list defines the size of one segment. We then print the sizes of the
individual segments. This function is called for different segment types
(except \ref TNL::Algorithms::Segments::SlicedEllpack,
as it behaves the same as \ref TNL::Algorithms::Segments::Ellpack for
such a small example).

The full example reads as:

\includelineno Algorithms/Segments/printSegmentsExample-1.cpp

The result looks as follows:

\include printSegmentsExample-1.out

We can see that the actual sizes of the segments differ for all
Ellpack-based formats. As mentioned earlier, these formats often use
**padding elements** to achieve more **efficient memory access**.
For example, the \ref TNL::Algorithms::Segments::ChunkedEllpack
format includes many such elements. However, this overhead is mainly
a result of the very small example presented here;
on large data, the overhead is usualy not so significant.

Let us remind the reader that segments represent a sparse format
rather than a data structure, as they do not store any data themselves.
The following example shows how to connect segments to an array:

\snippet Algorithms/Segments/printSegmentsExample-2.cpp segments setup

We first show how to create segments using a  vector (\ref TNL::Containers::Vector)
`sizes` that stores the segment sizes. The same constructor also works with arrays
(\ref TNL::Containers::Array) arrays views (\ref TNL::Containers::ArrayView)
and vector views (\ref TNL::Containers::VectorView).

Next, we print the actual segment sizes depending on the underlying format,
as in the previous example.

\snippet Algorithms/Segments/printSegmentsExample-2.cpp data setup

Then we allocate the array `data` of the size required by segments using
the method `getStorageSize` (e.g., \ref TNL::Algorithms::Segments::CSR::getStorageSize).
This method tells us how many elements are needed for the segments to
address all elements by their global index.

We use the `forAllElements` method to label each element of the array `data`
with its rank.

\snippet Algorithms/Segments/printSegmentsExample-2.cpp data print

Finally, we use the function \ref TNL::Algorithms::Segments::print,
which takes a lambda function `fetch` as a parameter. This lambda reads data from
the array `data` (using the array view `data_view`) based on the given `globalIdx`.

The full example reads as:

\includelineno Algorithms/Segments/printSegmentsExample-2.cpp

The result looks as follows:

\include printSegmentsExample-2.out

What we observe demonstrates that different segment formats can use very
different mappings from an element identified by its *segment index* and
*local index* (i.e., the rank of the element within the segment) to a *global index*,
which serves as an address in the associated container.

TODO: Example for construction of sorted segments

## Iteration over elements of segments

In this section, we show how to iterate over the elements of segments and how
to manipulate with them. There are several ways to specify which segments
should be traversed:

1. **All Segments:**
Functions `forElements` (\ref TNL::Algorithms::Segments::forElements) and
`forAllElements` (\ref TNL::Algorithms::Segments::forAllElements) iterate
in parallel over all elements of all segments (or specified range of segmets)
and apply the given lambda function to each of them.
2. **Selected Segments by Index:**
The same function, `forElements` (\ref TNL::Algorithms::Segments::forElements),
can additionally be provided with an array of segment indices. In that case,
they iterate in parallel only over the elements of the specified segments
and apply the given lambda function to each of them.
3. **Conditional Segment Selection:**
Functions `forElementsIf` (\ref TNL::Algorithms::Segments::forElementsIf) and
`forAllElementsIf` (\ref TNL::Algorithms::Segments::forAllElementsIf) iterate
in parallel over all elements of the segments that fulfill a given condition
(specified by a lambda function) and apply a provided lambda function to each of them.

All functions mentioned above traverse the elements of segments in parallel.
In some cases, however, it is better to iterate over the elements of a particular
segment sequentially. This can be done in the same way as for parallel traversal:

1. **All Segments:**
Functions `forSegments` (\ref TNL::Algorithms::Segments::forSegments) and
`forAllSegments` (\ref TNL::Algorithms::Segments::forAllSegments) iterate
in parallel over all segments (or specified range of segments) and apply the given
lambda function to each of them.
1. **Selected Segments by Index:**
The same functions `forSegments` (\ref TNL::Algorithms::Segments::forSegments) and
`forAllSegments` (\ref TNL::Algorithms::Segments::forAllSegments) can also additionally
be provided with an array of segment indices. In that case, they iterate in
parallel only over the specified segments and apply the given lambda function
to each of them.
1. **Conditional Segment Selection:**
Functions `forSegmentsIf` (\ref TNL::Algorithms::Segments::forSegmentsIf) and
`forAllSegmentsIf` (\ref TNL::Algorithms::Segments::forAllSegmentsIf) iterate
in parallel over all segments that fulfill a given condition
(specified by a lambda function) and apply a provided lambda function to each of them.

Functions iterating over particular segments use a segment view
(\ref TNL::Algorithms::Segments::SegmentView) to access the elements of given
segment. The segment view offers iterator for better convenience.

### Function forElements

The following example shows use of the function `forElements`.

\snippet Algorithms/Segments/SegmentsExample_forElements.cpp setup

We first create segments with linearly increasing sizes
(i.e., a structure resembling a lower triangular matrix).
Next, we allocate an array data with the same size as the total
number of elements managed by the segments. This size can be
obtained using the getStorageSize method
(see, e.g., \ref TNL::Algorithms::Segments::CSR::getStorageSize).

\snippet Algorithms/Segments/SegmentsExample_forElements.cpp traversing-1

We then create an array view `data_view` for convenient access in lambda
functions. After that, we call the function `forAllElements`, which iterates
in parallel over all elements in the segments. For each element, it
invokes a user-provided lambda function. The lambda receives three arguments:

- `segmentIdx`: index of the segment the element belongs to,
- `localIdx`: index of the element within its segment,
- `globalIdx`: global index of the element within the array data.

We use the globalIdx to assign the corresponding entry in data to the segment index.

\snippet Algorithms/Segments/SegmentsExample_forElements.cpp printing-1

Next, we print the array `data`, which shows the segment membership of each element
by its value. The layout of the elements depends on the segment type
(i.e., the sparse format used). We also print the contents of `data`
grouped by segments using the function `printSegments`, which iterates
over all elements and accesses the values using a `fetch` lambda function.

Traversing segments this way includes even padding elements used by
certain segment formats to achieve more efficient memory accesses.
This is, however, not what we usually want to do. To skip the padding
elements, we need to check the size of each segment explicitly, as
demonstrated in the following code snippet from the same example.

\snippet Algorithms/Segments/SegmentsExample_forElements.cpp traversing-2

Here we first erase the array `data` by setting all its elements to zero. Next,
we traverse the segment elements in the same way, but inside the lambda function,
we check the size of each segment, which in this case equals the segment index.
Finally, we print the data managed by the segments as before.

\snippet Algorithms/Segments/SegmentsExample_forElements.cpp printing-2

The full example looks as follows:

\includelineno Algorithms/Segments/SegmentsExample_forElements.cpp

Note, that for the Ellpack format, the output of the traversing through **all** elements looks as follows:

```text
Seg. 0: [ 0, 0, 0, 0, 0 ]
Seg. 1: [ 1, 1, 1, 1, 1 ]
Seg. 2: [ 2, 2, 2, 2, 2 ]
Seg. 3: [ 3, 3, 3, 3, 3 ]
Seg. 4: [ 4, 4, 4, 4, 4 ]
```

But if we check the segments sizes and **skip the padding elements**, we get the following output:

```text
Seg. 0: [ 0, 0, 0, 0, 0 ]
Seg. 1: [ 1, 1, 0, 0, 0 ]
Seg. 2: [ 2, 2, 2, 0, 0 ]
Seg. 3: [ 3, 3, 3, 3, 0 ]
Seg. 4: [ 4, 4, 4, 4, 4 ]
```

The result of the entire example looks as follows:

\include SegmentsExample_forElements.out

Note, that the function \ref TNL::Algorithms::Segments:forElements allows to specify range of segments
over which we aim to traverse.

### Function forElements with segment indexes

The function `forElements` also allows traversing only selected segments, based on their indexes.
We begin with the same setup as in the previous example:

\snippet Algorithms/Segments/SegmentsExample_forElementsWithIndexes.cpp setup

Next, we iterate only over **even-indexed segments**:

\snippet Algorithms/Segments/SegmentsExample_forElementsWithIndexes.cpp traversing

We first create the array `segmentIndexes` that stores the indexes of even segments.
This array is passed as the second argument to the `forElements` function,
which then processes only those specified segments.

Finally, we print the contents of the data array managed by the segments in the
same way as in the previous example.

The full example is shown below:

\includelineno Algorithms/Segments/SegmentsExample_forElementsWithIndexes.cpp

The result of the entire example looks as follows:

\include SegmentsExample_forElementsWithIndexes.out

### Function forElements with condition on segment indexes

Traversing only the elements of even-indexed segments can also be achieved using
the function `forElementsIf`. This function allows you to specify which segments
to traverse using **a condition expressed as a lambda function**.

The usage is demonstrated in the following example. The setup is the same as in
the previous examples. The traversing part looks as follows:

\snippet Algorithms/Segments/SegmentsExample_forElementsIf.cpp traversing

Instead of array with segments indexes, the function `forAllElementsIf` takes a lambda function.

\snippet Algorithms/Segments/SegmentsExample_forElementsIf.cpp condition

This lambda expresses the condition that a segment must fulfill to be included in the traversal.
It accepts a single argument, `segmentIdx`, which represents the segment index used in the condition.

The remainder of the example follows the same structure as before:

\includelineno Algorithms/Segments/SegmentsExample_forElementsIf.cpp

The result of the entire example is:

\include SegmentsExample_forElementsIf.out

Note that there is also the function \ref TNL::Algorithms::Segments::forElementsIf
which allows for conditional traversing of segments **within a given range**.

### Function forSegments

The function `forSegments` iterates in parallel over selected segments.
However, the iteration over elements **within each segment is sequential**.

There are two main reasons for this behavior:

1. **Sequential dependency**: The computation for an element may depend on the result
  of the computation for the previous element in the same segment. In such cases,
  sequential processing within a segment is necessary to preserve correctness.
2. **Common computations within a segment**: If part of the computation is the
   same for all elements in a segment, it is more efficient to compute it once
   per segment, then process the elements. The function `forElements` does not
   allow sharing data between different elements in a segment, and would
   therefore repeat the common computation for every element, which is inefficient.

#### Sequential dependency

The first situation — where sequential processing is required — is demonstrated in
the following example:

\snippet Algorithms/Segments/SegmentsExample_forSegments-1.cpp traversing

In this example, we compute the cumulative sum of all elements within each segment.
This is a typical case where the result for one element depends on the result of
the previous element.

The lambda function passed to `forAllSegments` receives a segment view, and then
uses a for loop to iterate over all elements of the segment. Each element is represented
by a variable `element` of type \ref TNL::Algorithms::Segments::SegmentElement,
which provides access to methods:

-  `localIndex` – index of the element within the segment
-  `segmentIndex` – index of the segment
-  `globalIndex` – global index in the data array

To perform the cumulative sum, we use an auxiliary variable `sum`, which accumulates
the values of the elements as we iterate over them. This demonstrates the sequential
dependency that prevents parallel execution at the element level within a segment.

The full example looks as follows:

\includelineno SegmentsExample_forSegments-1.cpp

The result looks as follows:

\include SegmentsExample_forSegments-1.out

**Note:** The cumulative sum of elements within segments — also referred to as a
*scan* or *prefix sum* — can also be computed using the function
\ref TNL::Algorithms::Segments::scan.

#### Common computations

Now let’s take a look at the second situation — when there are common computations
shared by all elements of a segment.

In the following example, we use the function `forAllSegments` to normalize each element
by dividing it by the sum of all values in its segment.

\snippet Algorithms/Segments/SegmentsExample_forSegments-2.cpp traversing

This process includes two steps inside the segment traversal:

- First, we compute the sum of all elements in the segment. This is the common
  computation for the entire segment.
- Then, we iterate over the elements again and divide each by the segment sum,
  stored in the variable sum.

The full example looks as follows:

\includelineno Algorithms/Segments/SegmentsExample_forSegments-2.cpp

The result looks as:

\include SegmentsExample_forSegments-2.out

### Function forSegments with segment indexes

Similar to the function \ref TNL::Algorithms::Segments::forElements, the function
\ref TNL::Algorithms::Segments::forSegments also allows you to specify a range of
segments to traverse, or explicitly provide the indexes of segments to be traversed.
The later is demonstrated in the following code snippet:

\snippet Algorithms/Segments/SegmentsExample_forSegmentsWithIndexes.cpp traversing

The example computes cumulative sums in the same way as the previous example,
but only for the selected segments.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_forSegmentsWithIndexes.cpp

And the result looks as:

\include SegmentsExample_forSegmentsWithIndexes.out

### Function forSegments with condition on segment indexes

Another way to specify which segments to traverse is by using a condition.
For this purpose, you can use the function \ref TNL::Algorithms::Segments::forSegmentsIf.

This is demonstrated in the following code snippet:

\snippet Algorithms/Segments/SegmentsExample_forSegmentsIf.cpp traversing

The example computes cumulative sums, just like in the previous examples,
but this time only for the even-indexed segments, based on the condition provided
in a lambda function.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_forSegmentsIf.cpp

And the result looks as:

\include SegmentsExample_forSegmentsIf.out


## Flexible reduction within segments

In this section, we explain an extension of [flexible reduction](#ug_ReductionAndScan)
to segments. It enables you to reduce all elements within the same segment and
store the result in an output array.

There are several ways to specify which segments should be included in the reduction:

1. **All Segments**
Use the functions `reduceSegments`(\ref TNL::Algorithms::Segments::reduceSegments) and
`reduceAllSegments`(\ref TNL::Algorithms::Segments::reduceAllSegments) to perform
reduction over all segments (or a specified range of segments).
2. **Selected Segments by Index**
The same function, `reduceSegments`(\ref TNL::Algorithms::Segments::reduceSegments),
can also be provided with an array of segment indices. In this case, reduction is
performed only within the specified segments.
3. **Conditional Segment Selection**
Use the functions `reduceSegmentsIf`(\ref TNL::Algorithms::Segments::reduceSegmentsIf) and
`reduceAllSegmentsIf`(\ref TNL::Algorithms::Segments::reduceAllSegmentsIf)
to perform reduction only in segments that fulfill a given condition,
which is specified using a lambda function.


### Function reduceSegments

Redcution within segments is demosntrated in the following code snippet:

\snippet Algorithms/Segments/SegmentsExample_reduceSegments.cpp reduction

The function `reduceAllSegments`, which we call at the end, requires three lambda functions:

1. `fetch`, which reads data associated with individual elements of the segments.
   The fetch function can be written in two forms - *brief* and *full*:

   - *Brief form:* The lambda function receives only the global index and the compute flag:

   ```cpp
   auto fetch = [=] __cuda_callable__ ( int globalIdx ) -> double { ... };
   ```

   - *Full form:* The lambda function receives the segment index, local index, and global index:

   ```cpp
   auto fetch = [=] __cuda_callable__ ( int segmentIdx, int localIdx, int globalIdx ) -> double { ... };
   ```

   Here:
   - `segmentIdx` is the index of the segment,
   - `localIdx` is the index of the element within the segment,
   - `globalIdx` is the index of the element in the global array.

Many segment formats are optimized for significantly better performance
when the brief variant is used. The form of the fetch lambda function is
automatically detected using [SFINAE](https://en.cppreference.com/w/cpp/language/sfinae) ,
which makes both variants easy to use.
2. `reduce` is a lambda function representing the reduction operation.
   In our case, it is defined as:

   ```cpp
   auto reduce = [=] __cuda_callable__ ( const double& a, const double& b ) -> double { return a + b; };
   ```

   Alternatively, you can simply use the predefined functors like
   \ref TNL::Plus, \ref TNL::Times etc.
2. `keep` is a lambda function responsible for storing the results of
   the reduction. It should be defined as:

   ```cpp
   auto keep = [=] __cuda_callable__ ( int segmentIdx, const double& value ) mutable { ... };
   ```

   Here, `segmentIdx` is the index of the segment whose reduction
   result we are storing, and `value` is the result of the reduction.

To use reduction within segments, we first create a vector `sums`
to store the results and prepare a view to this vector for use inside
the lambda functions.

We demonstrate the use of both fetch function variants - the full form
via `fetch_full` and the brief form via `fetch_brief`.

Next, we define the lambda function `keep`, which stores the sum from
each segment into the vector sums.

Finally, we call the function `reduceAllSegments`
(\ref TNL::Algorithms::Segments::reduceAllSegments)
to compute the reductions in the segments, first, using `fetch_full`
and then, using `fetch_brief`. In both cases, we use tghe functor `TNL::Plus`
for the reduction operation. We then print the results, which
is supposed be the same for both variants.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_reduceSegments.cpp

The result looks as follows:

\include SegmentsExample_reduceSegments.out

### Function reduceSegments with segment indexes

Reduction can also be performed only in segments with specified indexes.
This is demonstrated in the following code snippet:

\snippet Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexes.cpp reduction

Here, we call the function \ref TNL::Algorithms::Segments::reduceSegments which
takes, as its second parameter, an array `segmentIndexes` containing the indices
of segments in which we want to perform the reduction. This function also accepts
a lambda function for fetching data in either the *full* or *brief* form,
such as `fetch_full` and `fetch_brief`. The `keep` lambda function is now
slightly different and takes the following parameters:

1. `indexOfSegmentIdx` - the position of the segment index in the array `segmentIndexes`.
   This information is useful when we want to store the results of the reductions
   at the positions corresponding to segment indexes in the `segmentIndexes` array -
   that is, in a compressed format.
2. `segmentIdx` - the actual index of the segment being reduced.
3. `value` - the result of the reduction for the given segment.

The difference between the use of indexOfSegmentIdx and segmentIdx is
demonstrated using two vectors:
- `sum` - a vector whose size equals the total number of all segments.
- `compressedSum` - a vector whose size equals the number of segments
  that actively participate in the reduction.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexes.cpp

And the output looks as:

\include SegmentsExample_reduceSegmentsWithSegmentIndexes.out

### Function reduceSegmentsIf with condition on segment indexes

Reduction within segments with a condition on segment indexes
is demonstrated by the following code snippet:

\snippet Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp reduction

It works the same way as reduction within segments specified by segment
indexes. However, instead of providing an array of segment indexes, we
use a lambda function with a condition. Only those segments whose
indices satisfy the condition will participate in the reduction.

Note that even when using the function `reduceSegmentsIf`, the
lambda function `keep` receives the parameter `indexOfSegmentIdx`, which has
the same meaning as before — it represents the rank (position)
of the segment among those actively participating in the reduction.

The fulle example reads as:

\includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp

And the output looks as:

\include SegmentsExample_reduceSegmentsIf.out

### Reduction with argument

The function `reduceSegmentsWithArgument` also works with the positions
of elements within segments. This is useful in situations where we need
to determine, for example, the position of the smallest or largest
element in each segment. This functionality is demonstrated in the
following code snippet:

\snippet Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp reduction

Here we search for the maximum element in each segment. The function works
similarly to `reduceSegments`, but it requires reduction functors with argument,
such as `TNL::MinWithArg` or `TNL::MaxWithArg`. The lambda function `keep`
receives an additional parameter `localIdx`, which represents the position
of the maximum element within the segment.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp

The result looks as:

\include SegmentsExample_reduceSegmentsWithArgument.out

### Function reduceAll

The function `reduceAll` performs **local** reduction within
specified segments, followed by a **global** reduction of the
intermediate results obtained from the individual segment
reductions.

It is demonstrated by the following code snippet, where we compute
the sum of the maximum values across all segments.

\snippet Algorithms/Segments/SegmentsExample_reduceAll.cpp reduction

The function accepts the following lambda functions and functors:

1. **Local fetch:** A lambda function for fetching data from segments.
   It can be provided in either *full* or *brief* form, as in other
   segment-reduction functions.
2. **Local reduction:** A functor representing the reduction operation
    to be performed within individual segments.
3. **Global fetch:** A lambda function that receives the result of
   the local reduction for each segment, optionally performs a
   transformation, and returns the value to be used as input for the
   global reduction.
4. **Global reduction:** A functor representing the global reduction,
   i.e., the reduction performed across all segment-level results.

Since the result of this function is a single value, a `keep` lambda
function is not required.

In our example, we first need to find the maximum value in each segment.
Therefore, the local reduction is performed using `TNL::Max`.
Next, we want to compute the sum of these maxima, so the global
reduction is performed using `TNL::Plus`.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_reduceAll.cpp

And the output looks as:

\include SegmentsExample_reduceAll.cpp

**Note:** To perform complete reduction on only specific segments,
you can use the functions \ref TNL::Algorithms::Segments::reduceAll
with segment indexes or \ref TNL::Algorithms::Segments::reduceAllIf
with a condition on segment indexes. These functions behave the same
way as \ref TNL::Algorithms::Segments::reduceSegments with segment
indexes and \ref TNL::Algorithms::Segments::reduceSegmentsIf, and
therefore we do not cover them in more detail in this user guide.

## Scan (prefix-sum)

With the function \ref TNL::Algorithms::Segments::scan, you can perform
scan (or prefix-sum) within segments. Both *inclusive* and *exclusive*
scans are supported, as demonstrated by the following code snippet:

\snippet Algorithms/Segments/SegmentsExample_scan.cpp scan

The function accepts the following lambda functions:

- `fetch` is responsible for reading data managed by the segments. This
function receives the parameters `segmentIdx`, `localIdx`, and `globalIdx`,
which have the same meaning as in other reduction operations.
- `write` writes the results to the output array. It receives parameters
`globalIdx` (the position in the output array) and `value`, which is the
result of the scan operation at the given position.

The function also accepts functors expressing the reduction operation,
such as `TNL::Plus` and similar. Depending on what kind of scan you
want to perform, you can call either
\ref TNL::Algorithms::Segments::inclusiveScanAllSegments or
\ref TNL::Algorithms::Segments::exclusiveScanAllSegments.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_scan.cpp

The output looks as:

\include SegmentsExample_scan.out

**Note**: Onecan also use
\ref TNL::Algorithms::Segments::inclusiveScanSegments and
\ref TNL::Algorithms::Segments::exclusiveScanSegments
with an array of segment indexes to specify the segments in which scan
operations should be performed. Alternatively, one may use
\ref TNL::Algorithms::Segments::inclusiveScanSegmentsIf and
\ref TNL::Algorithms::Segments::exclusiveScanSegmentsIf,
where segments are selected based on a condition on their index.

## Find

The function \ref TNL::Algorithms::Segments::findInSegments is used
for parallel searching within segments. The following code snippet
shows how to find a specific number in each segment:

\snippet Algorithms/Segments/SegmentsExample_find.cpp find

The function takes two lambda functions as arguments:

1. `condition` - A lambda function that receives parameters `segmentIdx`,
   `localIdx`, and `globalIdx` (with their usual meanings), and returns
   a boolean value. A return value of `true` indicates that the given element
   satisfies the search condition, while `false` means it does not.
2. `keep` - A lambda function that receives parameters `segmentIdx`, `localIdx`,
   and `found`. If `found` is `true`, then the searched element was found in
   the segment with index `segmentIdx` at the position `localIdx`.

In our example, a boolean value indicating whether the searched element was found
in each segment is stored in the array `found`, and the positions of the found
elements are stored in the array `positions`.

The full example reads as:

\includelineno Algorithms/Segments/SegmentsExample_find.cpp

The output reads as:

\include SegmentsExample_find.out

## Sort

The function \ref TNL::Algorithms::Segments::sortSegments sorts data managed by the segments:

\snippet Algorithms/Segments/SegmentsExample_sort.cpp ascending sort

As demonstrated in the snippet above, three lambda functions must be provided:

1. `fetch` - This lambda receives parameters `segmentIdx`, `localIdx`, and `globalIdx`,
    which have their usual meanings. It is responsible for reading the data to be sorted.
    The function must also account for padding elements. A common strategy is
    to return either the maximum or lowest value of the given data type
    (depending on whether sorting is ascending or descending), using
    `std::numeric_limits< int >::max()` or `std::numeric_limits< int >::lowest()`.
2. `compare` - A lambda function that compares two elements `a` and `b`.
   It should return `true` if `a` should appear before `b` and `false` otherwise.
3. `swap` - This lambda performs the actual swap of two elements at positions `globalIdx1` and `globalIdx2`.

The following code snippet demonstrates sorting with the function
\ref TNL::Algorithms::Segments::sortSegments using specified segment
indexes and in descending order:

\snippet Algorithms/Segments/SegmentsExample_sort.cpp descending sort

In addition to the lambda functions `fetch` and `swap`, this version also
accepts an array of segment indexes `segmentIndexes` specifying which segments
should be sorted. The `compare` lambda function (which evaluates the condition `a <= b`
for ascending order) is replaced with the `compareDesc` function, which evaluates
the condition `a >= b` for descending order.

The full example showcasing also use of function \ref TNL::Algorithms::Segments::sortAllSegmentsIf
reads as:

\includelineno Algorithms/Segments/SegmentsExample_sort.cpp

And the output looks as:

\include SegmentsExample_sort.out

## Print

The function \ref TNL::Algorithms::Segments::print is used to print
data managed by segments. It requires a lambda function `fetch` which
reads data from individual elements.

The following code snippet demonstrates its use:

\snippet Algorithms/Segments/SegmentsExample_forSegments-1.cpp printing

The complete example looks as follows:

\includelineno Algorithms/Segments/SegmentsExample_forSegments-1.cpp

The output of the example is:

\include SegmentsExample_forSegments-1.out

**Note:** The function `print` does not mask padding elements. If your
segment format includes padding, those elements will be printed as well.

