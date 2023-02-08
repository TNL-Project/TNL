// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#include <TNL/Config/ConfigDescription.h>

#pragma once

/**
 * \brief Benchmark for measuring efficiency of access of CPU to the system memory.
 *
 * The benchmark methodology is inspired by text [What every programmer should know about memory](https://lwn.net/Articles/250967/).
 *
 * The benchmark measures efficiency of access of CPU to the system memory depending on:
 *
 * 1. Number of threads.
 * 2. The memory access pattern is random or sequential.
 * 3. The memory access consists of only reading or even writing.
 * 4. In case of sequential access pattern with multiple threads we test efficiency when each thread has its own contiguous block in
 *    the memory or the threads are accessing the test array elements one sequentially, i.e. thread with index TID is accessing elements
 *    with indexes `TID+i*NUM_THREADS` where `NUM_THREADS` is number of all threads.
 *
 * In the benchmark, testing array of given size in bytes is first created. The array is made of testing elements whose size is given by
 * a template parameter `Size`. The real size of the element equals `Size` times size of pointer. The element contains pointer to the next
 * element for traversing and the rest is data. The elements are connected either sequentially or randomly. The array is traversed repeatedly
 * and we measure effective bandwidth and the number of CPY cycles necessary for traversing from one element to the next one. After the test
 * finishes we increase the array size and perform the same test again.
 */
struct MemoryAccessBenchmark
{
   static void configSetup( TNL::Config::ConfigDescription& config );

   template< int ElementSize >
   static bool performBenchmark( const TNL::Config::ParameterContainer& parameters );
};

#include "MemoryAccessBenchmark.hpp"
