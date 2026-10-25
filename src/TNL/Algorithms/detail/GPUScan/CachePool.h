// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <mutex>

#include <TNL/Backend.h>
#include <TNL/Containers/Array.h>
#include <TNL/Algorithms/detail/GPUScan/GPUScanKernelLookback.h>

namespace TNL::Algorithms::detail {

// Cache of the lookback states array, reused across calls to `performLookback`
// to avoid `cudaMalloc`/`cudaFree` churn.
template< typename ValueType >
struct LookbackCache
{
   std::mutex mutex;
   Containers::Array< LookbackState< ValueType >, Devices::Cuda > states;
   // epoch of the last scan which used the states, see LookbackStatus
   unsigned int epoch = 0;
   // counter used by the kernel to assign the tiles to the blocks; it is
   // reset only together with the states, each scan uses the values
   // starting at tileCounterBase
   Containers::Array< unsigned long long, Devices::Cuda > tileCounter;
   unsigned long long tileCounterBase = 0;
};

// Pool of the lookback caches, one per device. Following CudaReductionBuffer
// and Backend::StreamPool, the pool is a singleton constructed on first use,
// so it is destroyed before the CUDA context and the cached arrays are freed
// in its destructor. The pool is shared by all launcher instantiations with
// the same value type.
template< typename ValueType >
class LookbackCachePool
{
public:
   static LookbackCachePool&
   getInstance()
   {
      // note that this ensures construction on first use, and thus also correct
      // destruction before the CUDA context is destroyed
      // https://stackoverflow.com/questions/335369/finding-c-static-initialization-order-problems#335746
      static LookbackCachePool instance;
      return instance;
   }

   LookbackCache< ValueType >&
   getCache()
   {
      std::scoped_lock lock( mutex );
      // std::map does not invalidate references to its elements on insertion
      return pool[ Backend::getDevice() ];
   }

   // copy-constructor and copy-assignment are meaningless for a singleton class
   LookbackCachePool( const LookbackCachePool& ) = delete;
   LookbackCachePool&
   operator=( const LookbackCachePool& ) = delete;

private:
   // private constructor of the singleton
   LookbackCachePool() = default;

   std::mutex mutex;
   std::map< int, LookbackCache< ValueType > > pool;
};

}  // namespace TNL::Algorithms::detail
