// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Execution of the C++ standard library and Thrust host algorithms in the
 * benchmarks.
 *
 * STDEXEC is the execution policy argument of the standard algorithms (empty
 * when the parallel algorithms are not available) and StdAlgorithmsDevice is
 * the device on which the algorithms called with it run. ThrustHostDevice is
 * the device on which the Thrust algorithms called with thrust::host run.
 */

#include <type_traits>

#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

#if __CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4 && __CUDACC_VER_MINOR__ <= 6
   // nvcc 12.4 fails when used with GCC 13 and the `execution` header is included
   // https://gitlab.com/-/snippets/3684272
   #define STDEXEC
#elif defined( HAVE_TBB ) && defined( __cpp_lib_parallel_algorithm )
   #include <execution>  // std::execution policies

   #if defined( __NVCOMPILER )
      // nvc++ does not support par_unseq for std::max_element: https://gitlab.com/-/snippets/2576260
      #define STDEXEC std::execution::par,
   #else
      #define STDEXEC std::execution::par_unseq,
   #endif
   #define STDEXEC_PARALLEL
#else
   #define STDEXEC
#endif

#ifdef HAVE_THRUST
   #include <thrust/execution_policy.h>
#endif

namespace TNL::Benchmarks {

// Device on which the C++ standard library algorithms called with STDEXEC run.
#ifdef STDEXEC_PARALLEL
using StdAlgorithmsDevice = Devices::Host;
#else
using StdAlgorithmsDevice = Devices::Sequential;
#endif

#ifdef HAVE_THRUST
// Device on which the Thrust algorithms called with thrust::host run. The Thrust host backend
// is selected by CMake (TBB, OpenMP, or the sequential CPP backend).
using ThrustHostDevice = std::conditional_t< THRUST_HOST_SYSTEM == THRUST_HOST_SYSTEM_CPP, Devices::Sequential, Devices::Host >;
#endif

}  // namespace TNL::Benchmarks
