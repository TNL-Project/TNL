// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <mutex>
#include <thread>
#include <tuple>

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

// Key of the caches: the active device, the stream the kernels are launched
// on, and the id of the host thread. The thread is part of the key, because
// the default stream is per-thread, so the same stream handle used by
// different host threads may denote different streams. Keying the caches by
// the stream (and the thread) makes them safe for asynchronous scans: the
// kernels sharing one cache are ordered on the same stream by construction.
using CacheKey = std::tuple< int, Backend::stream_t, std::thread::id >;

// Pool of the lookback caches, one per device, stream and host thread (see
// the comment above CacheKey for the keying rationale). Following
// CudaReductionBuffer and Backend::StreamPool, the pool is a singleton
// constructed on first use, so it is destroyed before the CUDA context and
// the cached arrays are freed in its destructor. The pool is shared by all
// launcher instantiations with the same value type.
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
   getCache( Backend::stream_t stream )
   {
      const CacheKey key( Backend::getDevice(), stream, std::this_thread::get_id() );
      std::scoped_lock lock( mutex );
      // std::map does not invalidate references to its elements on insertion
      return pool[ key ];
   }

   // copy-constructor and copy-assignment are meaningless for a singleton class
   LookbackCachePool( const LookbackCachePool& ) = delete;
   LookbackCachePool&
   operator=( const LookbackCachePool& ) = delete;

private:
   // private constructor of the singleton
   LookbackCachePool() = default;

   std::mutex mutex;
   std::map< CacheKey, LookbackCache< ValueType > > pool;
};

// Pool of the block shifts arrays of the two-phase scan, one per device,
// stream and host thread, reused across calls to avoid `cudaMalloc`/
// `cudaFree` churn: freeing the shifts as a local array would serialize the
// host with the pending kernels, which an asynchronous scan avoids by
// definition. Each cache grows as needed; when the requested size shrinks
// we keep the existing (larger) allocation. Reusing a cache across calls
// is safe only for kernels ordered on the same stream, which the cache
// keying (see the comment above CacheKey) guarantees. The recursive scan of
// the block shifts reuses the cache in place, which assumes that the
// recursion does not enter the two-phase path again - this holds as long as
// the first phase spans at most
// `blockSize * valuesPerThread * maxGridSize()` blocks, which no allocatable
// array can exceed. Like the lookback cache pool, it is a singleton
// constructed on first use and shared by all launcher instantiations with
// the same value type.
template< typename ValueType >
class BlockShiftsCachePool
{
public:
   static BlockShiftsCachePool&
   getInstance()
   {
      // note that this ensures construction on first use, and thus also correct
      // destruction before the CUDA context is destroyed
      // https://stackoverflow.com/questions/335369/finding-c-static-initialization-order-problems#335746
      static BlockShiftsCachePool instance;
      return instance;
   }

   Containers::Array< ValueType, Devices::Cuda >&
   getCache( Backend::stream_t stream )
   {
      const CacheKey key( Backend::getDevice(), stream, std::this_thread::get_id() );
      std::scoped_lock lock( mutex );
      // std::map does not invalidate references to its elements on insertion
      return pool[ key ];
   }

   // copy-constructor and copy-assignment are meaningless for a singleton class
   BlockShiftsCachePool( const BlockShiftsCachePool& ) = delete;
   BlockShiftsCachePool&
   operator=( const BlockShiftsCachePool& ) = delete;

private:
   // private constructor of the singleton
   BlockShiftsCachePool() = default;

   std::mutex mutex;
   std::map< CacheKey, Containers::Array< ValueType, Devices::Cuda > > pool;
};

}  // namespace TNL::Algorithms::detail
