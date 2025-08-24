# General concepts  {#ug_GeneralConcepts}

[TOC]

## Introduction {#ug_general_concepts_introduction}

In this part we describe some general and core concepts of programming with TNL.
Understanding these ideas may significantly help to use TNL more efficiently by
understanding the design of TNL algorithms and data structures.

The main goal of TNL is to allow developing high performance algorithms that
can run on multicore CPUs and GPUs. TNL offers a unified interface, so the
developer can write common code for all supported architectures.

## Devices and allocators {#ug_general_concepts_devices_and_allocators}

TNL offers unified interface for both CPUs (also referred as a host system) and
GPUs (referred as device). The connection between CPU and GPU is usually
realized by [PCI-Express bus](https://en.wikipedia.org/wiki/PCI_Express) which
is orders of magnitude slower compared to the speed of the GPU global memory.
Therefore, the communication between CPU and GPU must be reduced as much as
possible. As a result, the programmer needs to consider two different address
spaces, one for CPU and one for GPU. To distinguish between the address spaces,
each data structure requiring dynamic memory allocation needs to know on what
device it resides. This is done by a template parameter `Device`. For example
the following code creates two arrays, one on CPU and the other on GPU:

\includelineno snippet_devices_and_allocators_arrays_example.cpp

Since now, [C++ template specialization](https://en.wikipedia.org/wiki/Partial_template_specialization)
takes care of using the right methods for given device (in meaning hardware
architecture and so the  device can be even CPU). For example, calling a method
`setSize`

\includelineno snippet_devices_and_allocators_arrays_setsize_example.cpp

results in different memory allocation on CPU (for `host_array`) and on GPU
(for `cuda_array`). The same holds for assignment

\includelineno snippet_devices_and_allocators_arrays_assignment_example.cpp

in which case appropriate data transfer from CPU to GPU is performed. Each such
data structure contains inner type named `DeviceType` which tells where it
resides as we can see here:

\includelineno snippet_devices_and_allocators_arrays_device_deduction.cpp

If we need to specialize some parts of algorithm with respect to its device we
can do it by means of  \ref std::is_same :

\includelineno snippet_devices_and_allocators_arrays_device_test.cpp

## Algorithms and lambda functions {#ug_general_concepts_algorithms_and_lambda_functions}

Developing a code for GPUs (e.g. in [CUDA](https://developer.nvidia.com/CUDA-zone))
consists mainly of writing [kernels](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#kernels),
which are special functions running on the GPU in parallel. This can be very
hard and tedious work especially when it comes to debugging.
[Parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
is a perfect example of an algorithm which is relatively hard to understand and
implement on one hand, but it is necessary to use frequently. Writing tens of
lines of code every time we need to sum up some data is exactly what we mean by
tedious programming. TNL offers skeletons or patterns of such algorithms and
combines them with user defined [lambda functions](https://en.cppreference.com/w/cpp/language/lambda).
This approach is not absolutely general, which means that you can use it only
in situations when there is a skeleton/pattern (see \ref TNL::Algorithms)
suitable for your problem. But when there is, it offers several advantages:

1. Implementing lambda functions is much easier compared to implementing GPU
   kernels.
2. Code implemented this way works even on CPU, so the developer writes only
   one code for both hardware architectures.
3. The developer may debug the code on CPU first and then just run it on GPU.
   Quite likely it will work with only a little or no changes.

The following code snippet demonstrates it using \ref "TNL::Algorithms::parallelFor":

\includelineno snippet_algorithms_and_lambda_functions_parallel_for.cpp

In this example, we assume that all arrays `v1`, `v2` and `sum` were properly
allocated in the appropriate address space. If `Device` is \ref TNL::Devices::Host,
the lambda function is processed sequentially or in parallel by several OpenMP
threads on CPU. If `Device` is \ref TNL::Devices::Cuda, the lambda function is
called from a CUDA kernel (this is why it is annotated with `__cuda_callable__`,
which expands to `__host__ __device__`) by appropriate number of CUDA threads.

One more example demonstrates use of \ref "TNL::Algorithms::reduce":

\includelineno snippet_algorithms_and_lambda_functions_reduction.cpp

We will not explain the parallel reduction in TNL at this moment (see the
section about [flexible parallel reduction](ug_ReductionAndScan)), we hope that
the idea is intuitively understandable from the code snippet. If `Device` is
\ref TNL::Devices::Host, the scalar product is evaluated sequentially or in
parallel by several OpenMP threads on CPU, if `Device` is \ref TNL::Devices::Cuda,
the [parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf)
fine tuned with the lambda functions is performed. Fortunately, there is no
performance drop. On the contrary, since it is easy to generate CUDA kernels
for particular situations, we may get more efficient code. Consider computing
a scalar product of sum of vectors like this:

\f[
s = (u_1 + u_2, v_1 + v_2).
\f]

This can be solved by the following code:

\includelineno snippet_algorithms_and_lambda_functions_reduction_2.cpp

Notice that compared to the previous code example, we have changed only the
`fetch` lambda function to perform the sums of `u1[ i ] + u2[ i ]` and
`v1[ i ] + v2[ i ]`. Now we get a completely new CUDA kernel tailored exactly
for this specific problem.

Doing the same with [Cublas](https://developer.nvidia.com/cublas), for example,
would require splitting the computation into three separate kernels:

1. Kernel to compute \f$u_1 = u_1 + u_2\f$.
2. Kernel to compute \f$v_1 = v_1 + v_2\f$.
3. Kernel to compute \f$product = ( u_1, v_1 )\f$.

This could be achieved with the following code:

\includelineno snippet_algorithms_and_lambda_functions_reduction_cublas.cpp

We believe that C++ lambda functions with properly designed patterns of
parallel algorithms could make programming of GPUs significantly easier. We see
a parallel with [MPI standard](https://en.wikipedia.org/wiki/Message_Passing_Interface)
which in the nineties defined frequent communication operations in distributed
parallel computing. It made programming of distributed systems much easier and
at the same time MPI helps to write efficient programs. We aim to add
additional skeletons or patterns to \ref TNL::Algorithms.

## Views and shared pointers {#ug_general_concepts_views_and_shared_pointers}

You might notice that in the previous section we used only C style arrays
represented by pointers in the lambda functions. There is a general difficulty
when the programmer needs to access dynamic data structures in lambda functions
that should be callable on GPU. The outside variables may be captured either
_by value_ or _by reference_. The first case would be as follows:

\includelineno snippet_shared_pointers_and_views_capture_value.cpp

However, in this case a __deep copy__ of the array `a` will be made and so any
modifications to `a` inside the lambda will have no effect.
Capturing _by reference_ may look as follows:

\includelineno snippet_shared_pointers_and_views_capture_reference.cpp

This code is correct on CPU (e.g. when `Device` is \ref TNL::Devices::Host).
However, we are not allowed to pass references to CUDA kernels and so this
source code would not even compile with the CUDA compiler. To overcome this
issue, TNL offers two solutions:

1. Data structures views
2. Shared pointers

### Data structures views {#ug_general_concepts_data_structures_view}

A _view_ is a kind of lightweight reference object related to a dynamic data
structure. Unlike full data structures, views are not resizable, but they may
provide read-only as well as read-write access to the data. Another important
distinction is the copy-constructor: while the copy-constructor of a data
structure typically makes a _deep copy_, copy-constructing a view results in a
_shallow copy_. Intuitively, views represent references to a data structure
and thus copies of a view provide references to the same data. Therefore, views
can be captured _by value_ in lambda functions, which provides a way to
transfer a _reference_ to a data structure into a computational kernel.

The example with the array would look as follows:

\includelineno snippet_shared_pointers_and_views_capture_view.cpp

Compared to the previous code example, we first obtain a view for the array
using its `getView` method. Then we capture it _by value_, i.e. using `[ view ]`
rather than `[ &a ]`, and finally, we use `view` rather than the array `a`
inside the lambda function.

The view has very similar interface (see \ref TNL::Containers::ArrayView) as
the array (\ref TNL::Containers::Array) and so there is mostly no difference in
using array and its view. In TNL, each data structure designed for use in GPU
kernels (it means that it has methods annotated as `__cuda_callable__`)
provides also a `getView` method for getting the appropriate view of the object.

Note that the relation between a data structure and its view is one-way only:
a view has a reference to the data structure, but the data structure does not
keep references to any of its views that are currently in use. Consequently,
there are operations on the data structure after which the __views may become
invalid__, because the data structure could not update all the independent
objects referencing it. The most common example of such operation is
reallocation, which is shown in the following example:

\includelineno snippet_shared_pointers_and_views_capture_view_change.cpp

This code would not work correctly, because the array is first allocated with
`size` elements, then a view is obtained, and then the array size is changed to
`2 * size`, which will cause data reallocation and the view will now be invalid
(it will contain a pointer to an inaccessible memory segment with no means to
check and correct this state). This example can be fixed simply by using the
operations in a correct sequence: the view used in the lambda function must be
obtained _after_ resizing the data structure.

Note that changing the data managed by the array after fetching the view is not
an issue. See the following example:

\includelineno snippet_shared_pointers_and_views_capture_view_change_2.cpp

The method `setElement` changes the value of an element, but it does not cause
data reallocation or change of the array size, so the view is still valid and
usable in the lambda function.

### Shared pointers {#ug_general_concepts_shared_pointers}

TNL offers smart pointers working across different devices (meaning CPU or GPU).
See \ref ug_Pointers for more details on this topic.
