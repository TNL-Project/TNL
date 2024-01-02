# System Benchmarks

## CPU Memory Access Benchmark

This benchmark measures the efficiency of CPU access to the system memory. The methodology is inspired by the article [What every programmer should know about memory](https://lwn.net/Articles/250967/).

The benchmark evaluates the efficiency of CPU memory access based on:

1. The number of threads.
2. The memory access pattern, whether it is random or sequential.
3. The memory access type, involving either only reading or both reading and writing.
4. For sequential access patterns with multiple threads, the benchmark tests efficiency in two scenarios:
  - Each thread accesses its own contiguous memory block.
  - The threads access the test array elements sequentially, i.e., a thread with index `TID` accesses elements at indexes `TID+i*NUM_THREADS`, where `NUM_THREADS` is the total number of threads.

The benchmark involves testing an array of a given size in bytes. The array consists of testing elements structured as follows:

```cpp
template<int Size>
class ArrayElement
{
   ArrayElement* next;
   long int data[Size];
};
```

The size of each element is `(Size+1)` times the size of a pointer on the given system, because the size of `long int` is equal to the size of a pointer. The element contains a pointer to the next element for array traversal, and the rest is data. The elements are linked either sequentially or randomly. The array is repeatedly traversed to measure the effective bandwidth and the number of CPU cycles necessary for moving from one element to the next. After each test, the array size is increased, and the test is repeated.

The benchmark can be executed using the following command:

```bash
tnl-benchmark-memory-access
```

For available settings, use the `--help` command line argument. Alternatively, a bash script can be used to run a complete set of tests:

```bash
run-tnl-benchmark-memory-access
```

Upon completion, several log files are generated. These can be processed using a Python script:

```bash
process-tnl-benchmark-memory-access -i tnl-benchmark-memory-access.log
```

This script generates `.html` and `.pdf` files visualizing the benchmark results:

1. `tnl-benchmark-memory-access.html` contains a table with all the results.
2. Files named `{sequential|random}-{threads-count}-threads-{read|write}-{blocks|interleaving}-element-size-{element_size}-{bw|cycles}.pdf` show graphs visualizing results of specific tests.
3. Files named `sequential-random-comparison-{threads_count}-threads-{read|write}-element-size-{element_size}-{bw|cycles}.pdf` compare sequential and random access.
4. Files named `threads-comparison-{sequential|random}-{read|write}-{blocks|interleaving}-element-size-{element_size}-{bw|cycles}.pdf` compare memory accesses with different numbers of threads.
5. Files named `read-write-comparison-{sequential|random}-{threads_count}-threads-{blocks|interleaving}-element-size-{element_size}-{bw|cycles}.pdf` compare read and write tests.
6. Files named `blocked-interleaved-comparison-{threads_count}-threads-{test_type}-element-size-{element_size}-{bw|cycles}.pdf` compare blocked and interleaved array ordering for sequential access.
7. Files named `element-size-comparison-{threads_count}-threads-{access}-{test_type}-{ordering}-{cycles|bw}.pdf` compare various element sizes.

Files with `bw` indicate memory bandwidth usage, while those with `cycles` indicate the number of CPU cycles. The `interleaving` ordering of the array is tested only for sequential access with more than one thread.
