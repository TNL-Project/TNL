// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>

#include "../QuicksortPivot.h"
#include "QuicksortTask.h"

namespace TNL::Algorithms::Sorting::detail {

#if defined( __CUDACC__ ) || defined( __HIP__ )

/**
 * \brief Returns the offset of the sample with the given index.
 *
 * The samples are evenly spaced over the range. The quotient/remainder
 * decomposition keeps the intermediate products below the range size, so it does
 * not overflow for any index type.
 */
template< typename Index >
__device__
Index
sampleOffset( Index taskSize, int nSamples, int sample )
{
   const Index last = taskSize - 1;
   const Index divisor = static_cast< Index >( nSamples - 1 );
   return sample * ( last / divisor ) + sample * ( last % divisor ) / divisor;
}

/**
 * \brief Returns the index of the sample whose rank is closest to the middle.
 *
 * \param partitionBegin first index of the partitioned range
 * \param taskSize number of elements in the partitioned range
 * \param samples one sample per thread, stored in shared memory
 *
 * Must be called by the whole block. Ranks are computed by counting, so no
 * sorting network and only two barriers are needed. The returned index refers to
 * a real element of the range, which keeps the sort a permutation.
 */
template< typename Value, typename Index, typename Compare >
__device__
Index
pickSampledPivotIndex( Index partitionBegin, Index taskSize, int nSamples, const Value* samples, const Compare& compare )
{
   const int half = nSamples / 2;
   int rank = 0;
   if( threadIdx.x < nSamples ) {
      const Value mySample = samples[ threadIdx.x ];
      for( int j = 0; j < nSamples; ++j )
         rank += compare( samples[ j ], mySample ) ? 1 : 0;
   }
   else {
      rank = 2 * half;  // largest possible deviation, the thread can never win
   }

   const int deviation = rank > half ? rank - half : half - rank;
   // the deviation is packed with the thread index so that ties are broken
   // deterministically; a reduction is used instead of an atomic because
   // hundreds of threads serialized on one shared memory location cost more than
   // the rest of the kernel
   int packed = ( deviation << 16 ) | static_cast< int >( threadIdx.x );
   for( int offset = warpSize / 2; offset > 0; offset >>= 1 ) {
      const int other = Backend::warp_shuffle_down( packed, offset );
      packed = other < packed ? other : packed;
   }

   const int lane = static_cast< int >( threadIdx.x ) % warpSize;
   const int warp = static_cast< int >( threadIdx.x ) / warpSize;
   const int warps = static_cast< int >( blockDim.x ) / warpSize;
   __shared__ int warpBest[ 32 ];
   if( lane == 0 )
      warpBest[ warp ] = packed;
   __syncthreads();

   __shared__ Index pivotIdx;
   if( threadIdx.x == 0 ) {
      int best = warpBest[ 0 ];
      for( int w = 1; w < warps; ++w )
         best = warpBest[ w ] < best ? warpBest[ w ] : best;
      const int winner = best & 0xffff;
      pivotIdx = partitionBegin + sampleOffset( taskSize, nSamples, winner );
   }
   __syncthreads();
   return pivotIdx;
}

#endif

template< typename Value, typename Index, typename Compare >
__global__
void
GPUQuicksortKernelPrepareTasks(
   Containers::ArrayView< QuicksortTask< Index >, Devices::GPU > tasks,
   Containers::ArrayView< int, Devices::GPU > blockToTask,
   Containers::ArrayView< int, Devices::GPU > reductionTaskInitMem,
   Containers::ArrayView< Value, Devices::GPU, Index > src,
   Compare compare,
   int nSamples,
   bool canBufferSamples )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( blockIdx.x >= tasks.getSize() )
      return;

   int start = blockIdx.x == 0 ? 0 : reductionTaskInitMem[ blockIdx.x - 1 ];
   int end = reductionTaskInitMem[ blockIdx.x ];
   for( int i = start + threadIdx.x; i < end; i += blockDim.x )
      blockToTask[ i ] = blockIdx.x;

   QuicksortTask< Index >& task = tasks[ blockIdx.x ];
   const Index taskSize = task.partitionEnd - task.partitionBegin;

   /*
    * The median of three elements gives splits around 70/30 on average, which
    * roughly doubles the number of partitioning rounds compared to the median of
    * the data. Sampling the range is much cheaper than a full median estimate and
    * the quality saturates quickly: the gain over the median of three is already
    * there for a few dozen samples, while every sample is a separate memory
    * transaction that is paid again for each task in every round.
    */
   // named differently than the buffers in the other kernels, because all extern
   // __shared__ declarations of the same name in a translation unit must be
   // compatible, but this one is aligned for arbitrary Value types
   extern __shared__ __align__( 16 ) unsigned char sampleMem[];
   Value* samples = reinterpret_cast< Value* >( sampleMem );
   const bool useSampledPivot = canBufferSamples && taskSize > static_cast< Index >( nSamples ) && nSamples > 1;
   Index pivotIdx;
   if( useSampledPivot ) {
      if( threadIdx.x < nSamples ) {
         const Index offset = sampleOffset( taskSize, nSamples, static_cast< int >( threadIdx.x ) );
         samples[ threadIdx.x ] = src[ task.partitionBegin + offset ];
      }
      __syncthreads();
      pivotIdx = pickSampledPivotIndex( task.partitionBegin, taskSize, nSamples, samples, compare );
   }
   else {
      pivotIdx =
         task.partitionBegin + pickMedianOfThree( src.getView( task.partitionBegin, task.partitionEnd ), compare ).second;
   }

   if( threadIdx.x == 0 )
      task.init( start, end - start, pivotIdx );
#endif
}
}  // namespace TNL::Algorithms::Sorting::detail
