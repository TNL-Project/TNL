// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

/* Cooperative scan across the threads of one CUDA block.
 *
 * GPUBlockScan is the generic implementation using shared memory; for the
 * fundamental arithmetic types it is specialized to GPUBlockScanShfl, which
 * scans within warps using the warp shuffle intrinsics and needs shared memory
 * only for the per-warp results.
 *
 * These are building blocks of the scan kernels (see GPUTileScan.h and
 * GPUScanKernelLookback.h), but they are also used directly by other
 * algorithms (e.g. the segments traversing kernels and the GPU sorting).
 */

#include <TNL/Assert.h>
#include <TNL/Backend.h>
#include <TNL/Algorithms/detail/ScanType.h>

namespace TNL::Algorithms::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

// Keeping the following storage structures outside of GPUScan objects make
// them independent of the scan operation type. This allows to reuse the same
// storage for both inclusive and exclusive scan.
template< typename ValueType, int BlockSize >
struct GPUScanStorage
{
   // accessed via Backend::getInterleaving()
   ValueType chunkResults[ BlockSize + BlockSize / Backend::getNumberOfSharedMemoryBanks() ];
   ValueType warpResults[ Backend::getWarpSize() ];
};

template< typename ValueType >
struct GPUScanShflStorage
{
   ValueType warpResults[ Backend::getWarpSize() ];
};

/* Template for cooperative scan across the CUDA block of threads.
 * It is a *cooperative* operation - all threads must call the operation,
 * otherwise it will deadlock!
 *
 * The default implementation is generic and the reduction is done using
 * shared memory. Specializations can be made based on `Reduction` and
 * `ValueType`, e.g. using the `__shfl_sync` intrinsics for supported
 * value types.
 */
template< ScanType scanType, int blockSize, typename Reduction, typename ValueType >
struct GPUBlockScan
{
   // storage to be allocated in shared memory
   using Storage = GPUScanStorage< ValueType, blockSize >;

   /* Cooperative scan across the CUDA block - each thread will get the
    * result of the scan according to its ID.
    *
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue  Value of the calling thread to be reduced.
    * \param tid          Index of the calling thread (usually `threadIdx.x`,
    *                     unless you know what you are doing).
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   __device__
   static ValueType
   scan( const Reduction& reduction, ValueType identity, ValueType threadValue, int tid, Storage& storage )
   {
      // verify the configuration
      TNL_ASSERT_EQ( blockDim.x, blockSize, "unexpected block size in GPUBlockScan::scan" );
      static_assert(
         blockSize / Backend::getWarpSize() <= Backend::getWarpSize(),
         "blockSize is too large, it would not be possible to scan warpResults using one warp" );

      // store the threadValue in the shared memory
      const int chunkResultIdx = Backend::getInterleaving( tid );
      storage.chunkResults[ chunkResultIdx ] = threadValue;
      __syncthreads();

      // perform the parallel scan on chunkResults inside warps
      const int lane_id = tid % Backend::getWarpSize();
      const int warp_id = tid / Backend::getWarpSize();
      auto warp = cg::tiled_partition< Backend::getWarpSize() >( cg::this_thread_block() );
   #pragma unroll
      for( int stride = 1; stride < Backend::getWarpSize(); stride *= 2 ) {
         ValueType result;
         if( lane_id >= stride )
            result = reduction(
               storage.chunkResults[ chunkResultIdx ], storage.chunkResults[ Backend::getInterleaving( tid - stride ) ] );
         // We must sync all threads in a warp after read and before write to avoid race condition
         warp.sync();
         if( lane_id >= stride )
            storage.chunkResults[ chunkResultIdx ] = result;
         warp.sync();
      }
      threadValue = storage.chunkResults[ chunkResultIdx ];

      // the last thread in warp stores the intermediate result in warpResults
      if( lane_id == Backend::getWarpSize() - 1 )
         storage.warpResults[ warp_id ] = threadValue;
      __syncthreads();

      // perform the scan of warpResults using one warp
      if( warp_id == 0 ) {
   #pragma unroll
         for( int stride = 1; stride < blockSize / Backend::getWarpSize(); stride *= 2 ) {
            ValueType result;
            if( lane_id >= stride )
               result = reduction( storage.warpResults[ tid ], storage.warpResults[ tid - stride ] );
            // We must sync all threads in a warp after read and before write to avoid race condition
            warp.sync();
            if( lane_id >= stride )
               storage.warpResults[ tid ] = result;
            warp.sync();
         }
      }
      __syncthreads();

      // shift threadValue by the warpResults
      if( warp_id > 0 )
         threadValue = reduction( threadValue, storage.warpResults[ warp_id - 1 ] );

      // shift the result for exclusive scan
      if( scanType == ScanType::Exclusive ) {
         storage.chunkResults[ chunkResultIdx ] = threadValue;
         __syncthreads();
         threadValue = ( tid == 0 ) ? identity : storage.chunkResults[ Backend::getInterleaving( tid - 1 ) ];
      }

      __syncthreads();
      return threadValue;
   }
};

template<
   ScanType scanType,
   int __unused,  // the __shfl implementation does not depend on the blockSize
   typename Reduction,
   typename ValueType >
struct GPUBlockScanShfl
{
   // storage to be allocated in shared memory
   using Storage = GPUScanShflStorage< ValueType >;

   /* Cooperative scan across the CUDA block - each thread will get the
    * result of the scan according to its ID.
    *
    * \param reduction    The binary reduction functor.
    * \param identity     Neutral element for given reduction operation, i.e.
    *                     value such that `reduction(identity, x) == x` for any `x`.
    * \param threadValue  Value of the calling thread to be reduced.
    * \param tid          Index of the calling thread (usually `threadIdx.x`,
    *                     unless you know what you are doing).
    * \param storage      Auxiliary storage (must be allocated as a __shared__
    *                     variable).
    */
   __device__
   static ValueType
   scan( const Reduction& reduction, ValueType identity, ValueType threadValue, int tid, Storage& storage )
   {
      const int lane_id = tid % Backend::getWarpSize();
      const int warp_id = tid / Backend::getWarpSize();

      // perform the parallel scan across warps
      ValueType total;
      threadValue = warpScan< scanType >( reduction, identity, threadValue, lane_id, total );

      // the last thread in warp stores the result of inclusive scan in warpResults
      if( lane_id == Backend::getWarpSize() - 1 )
         storage.warpResults[ warp_id ] = total;
      __syncthreads();

      // the first warp performs the scan of warpResults
      if( warp_id == 0 ) {
         // read from shared memory only if that warp existed
         if( tid < blockDim.x / Backend::getWarpSize() )
            total = storage.warpResults[ lane_id ];
         else
            total = identity;
         storage.warpResults[ lane_id ] = warpScan< ScanType::Inclusive >( reduction, identity, total, lane_id, total );
      }
      __syncthreads();

      // shift threadValue by the warpResults
      if( warp_id > 0 )
         threadValue = reduction( threadValue, storage.warpResults[ warp_id - 1 ] );

      __syncthreads();
      return threadValue;
   }

   /* Helper function.
    * Cooperative scan across the warp - each thread will get the result of the
    * scan according to its ID.
    * return value = thread's result of the *warpScanType* scan
    * total = thread's result of the *inclusive* scan
    */
   template< ScanType warpScanType >
   __device__
   static ValueType
   warpScan( const Reduction& reduction, ValueType identity, ValueType threadValue, int lane_id, ValueType& total )
   {
   // perform an inclusive scan
   #pragma unroll
      for( int stride = 1; stride < Backend::getWarpSize(); stride *= 2 ) {
         const ValueType otherValue = Backend::warp_shuffle_up( threadValue, stride );
         if( lane_id >= stride )
            threadValue = reduction( threadValue, otherValue );
      }

      // set the result of the inclusive scan
      total = threadValue;

      // shift the result for exclusive scan
      if( warpScanType == ScanType::Exclusive ) {
         threadValue = Backend::warp_shuffle_up( threadValue, 1 );
         if( lane_id == 0 )
            threadValue = identity;
      }

      return threadValue;
   }
};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, int > : public GPUBlockScanShfl< scanType, blockSize, Reduction, int >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, unsigned int >
: public GPUBlockScanShfl< scanType, blockSize, Reduction, unsigned int >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, long > : public GPUBlockScanShfl< scanType, blockSize, Reduction, long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, unsigned long >
: public GPUBlockScanShfl< scanType, blockSize, Reduction, unsigned long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, long long >
: public GPUBlockScanShfl< scanType, blockSize, Reduction, long long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, unsigned long long >
: public GPUBlockScanShfl< scanType, blockSize, Reduction, unsigned long long >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, float > : public GPUBlockScanShfl< scanType, blockSize, Reduction, float >
{};

template< ScanType scanType, int blockSize, typename Reduction >
struct GPUBlockScan< scanType, blockSize, Reduction, double >
: public GPUBlockScanShfl< scanType, blockSize, Reduction, double >
{};

#endif

}  // namespace TNL::Algorithms::detail
