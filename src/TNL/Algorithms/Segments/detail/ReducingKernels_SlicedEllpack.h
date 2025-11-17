// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< int BlockSize,
          int ThreadsPerSegment,
          typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__device__
void
reduceSegmentsRowMajorSlicedEllpackKernel( const int gridIdx,
                                           const Segments& segments,
                                           IndexBegin begin,
                                           IndexEnd end,
                                           Fetch& fetch,
                                           Reduction& reduce,
                                           ResultKeeper& keep,
                                           const Value& identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr Index SliceSize = Segments::getSliceSize();

   const Index segmentIdx = ( begin / SliceSize ) * SliceSize + Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment;
   if( segmentIdx < begin || segmentIdx >= end )
      return;

   const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

   const Index sliceIdx = segmentIdx / SliceSize;
   const Index segmentInSliceIdx = segmentIdx % SliceSize;
   TNL_ASSERT_LT( sliceIdx, segments.getSliceSegmentSizesView().getSize(), "" );

   const Index segmentSize = segments.getSliceSegmentSizesView()[ sliceIdx ];

   const Index beginIdx = segments.getSliceOffsetsView()[ sliceIdx ] + segmentInSliceIdx * segmentSize;
   const Index endIdx = beginIdx + segmentSize;
   TNL_ASSERT_EQ( beginIdx, segments.getGlobalIndex( segmentIdx, 0 ), "" );
   TNL_ASSERT_LE( endIdx, segments.getStorageSize(), "" );

   ReturnType result = identity;
   if constexpr( argumentCount< Fetch >() == 3 ) {
      Index localIdx = laneIdx;
      for( Index globalIdx = beginIdx + laneIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment ) {
         TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
         TNL_ASSERT_LT( globalIdx, endIdx, "" );
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
         localIdx += ThreadsPerSegment;
      }
   }
   else {
      for( Index i = beginIdx + laneIdx; i < endIdx; i += ThreadsPerSegment ) {
         result = reduce( result, fetch( i ) );
      }
   }

   // Parallel reduction
   #if defined( __HIP__ )
   if( ThreadsPerSegment > 16 ) {
      result = reduce( result, __shfl_down( result, 16 ) );
      result = reduce( result, __shfl_down( result, 8 ) );
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      result = reduce( result, __shfl_down( result, 8 ) );
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      result = reduce( result, __shfl_down( result, 4 ) );
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      result = reduce( result, __shfl_down( result, 2 ) );
      result = reduce( result, __shfl_down( result, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      result = reduce( result, __shfl_down( result, 1 ) );
   #else
   if( ThreadsPerSegment > 16 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 8 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 4 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 2 ) {
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   }
   else if( ThreadsPerSegment > 1 )
      result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
   #endif
   // Write the result
   if( ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 ) {
      keep( segmentIdx, result );
   }
#endif
}

// For this kernel we assume that Segments::getSliceSize() * ThreadsPerSegment >= WarpSize.
template< int BlockSize,
          int ThreadsPerSegment,
          typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__device__
void
reduceSegmentsColumnMajorSlicedEllpackKernel( const int gridIdx,
                                              const Segments& segments,
                                              IndexBegin begin,
                                              IndexEnd end,
                                              Fetch& fetch,
                                              Reduction& reduce,
                                              ResultKeeper& keep,
                                              const Value& identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;
   constexpr Index SliceSize = Segments::getSliceSize();

   static_assert( ThreadsPerSegment * Segments::getSliceSize() <= BlockSize,
                  "There are not enough threads in the block for the given configuration (ThreadsPerSegment and SliceSize)." );
   static_assert( ThreadsPerSegment * Segments::getSliceSize() >= Backend::getWarpSize(),
                  "The SliceSize is too small for given configuration (ThreadsPerSegment and warp size)." );
   /////
   // To describe this kernel we assume that the SlizeSize = 4. Then the mapping of segment elements is as follows:
   //          +----+----+----+----+
   //          |  0 |  4 |  8 | 12 |
   // Slice 0: |  1 |  5 |  9 | 13 |
   //          |  2 |  6 | 10 | 14 |
   //          |  3 |  7 | 11 | 15 |
   //          +----+----+----+----+----+
   //          | 16 | 20 | 24 | 28 | 32 |
   // Slice 1: | 17 | 21 | 25 | 29 | 33 |
   //          | 18 | 22 | 26 | 30 | 34 |
   //          | 19 | 23 | 27 | 31 | 35 |
   //          +----+----+----+----+----+
   //
   // For this kernel we assume that SliceSize * ThreadsPerSegment >= WarpSize. Otherwise, we
   // would have to split one warp between multiple segments, which would break the coalesced memory
   // access pattern.
   //
   // For the description of this kernel, we assume that ThreadsPerSegment = 2 and so we need
   // to assume that warp size is 8. The mapping of threads to segment elements is as follows:
   //
   //          +------+-----+-----+-----+
   //          |   T0 |  T4 |  T0 |  T4 |
   // Slice 0: |   T1 |  T5 |  T1 |  T5 |
   //          |   T2 |  T6 |  T2 |  T6 |
   //          |   T3 |  T7 |  T3 |  T7 |
   //          +------+-----+-----+-----+-----+
   //          |   T8 | T12 |  T8 | T12 |  T8 |
   // Slice 1: |   T9 | T13 |  T9 | T13 |  T9 |
   //          |  T10 | T14 | T10 | T14 | T10 |
   //          |  T11 | T15 | T11 | T15 | T11 |
   //          +------+-----+-----+-----+-----+
   //
   // In this case each slice is processed by one warp (Slice 0 by threads T0-T7 and Slice 1 by threads T8-T15).
   // If we had ThreadsPerSegment = 4, then each slice would be processed by two warps which is also allowed in this
   // kernel. Since we need to perform reduction over all threads processing one segment (i.e. recution within rows),
   // in the later case we would need to perform reduction over more warps and that requires usage of shared memory.
   /////

   const Index firstSliceIdx = begin / SliceSize;
   const Index sliceIdx = firstSliceIdx + Backend::getGlobalThreadIdx_x( gridIdx ) / ( SliceSize * ThreadsPerSegment );
   const Index inSliceThreadIdx = threadIdx.x % ( SliceSize * ThreadsPerSegment );
   const Index inSliceSegmentIdx = inSliceThreadIdx % SliceSize;
   const Index segmentIdx = sliceIdx * SliceSize + inSliceSegmentIdx;
   ReturnType result = identity;
   if( segmentIdx >= begin && segmentIdx < end ) {
      TNL_ASSERT_LT( sliceIdx, segments.getSliceSegmentSizesView().getSize(), "" );
      TNL_ASSERT_LT( inSliceSegmentIdx, SliceSize, "" );
      TNL_ASSERT_LT( segmentIdx, segments.getSegmentsCount(), "" );

      Index localIdx = inSliceThreadIdx / SliceSize;
      const Index beginIdx = segments.getSliceOffsetsView()[ sliceIdx ] + inSliceSegmentIdx + localIdx * SliceSize;
      const Index endIdx = segments.getSliceOffsetsView()[ sliceIdx + 1 ];
      TNL_ASSERT_LE( endIdx, segments.getStorageSize(), "" );  // equality is for the case when both values equal 0

      if constexpr( argumentCount< Fetch >() == 3 ) {
         for( Index globalIdx = beginIdx; globalIdx < endIdx; globalIdx += ThreadsPerSegment * SliceSize ) {
            TNL_ASSERT_EQ( globalIdx, segments.getGlobalIndex( segmentIdx, localIdx ), "" );
            TNL_ASSERT_LT( globalIdx, endIdx, "" );
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );

            result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
            localIdx += ThreadsPerSegment;
         }
      }
      else {
         for( Index i = beginIdx; i < endIdx; i += ThreadsPerSegment * SliceSize ) {
            result = reduce( result, fetch( i ) );
         }
      }
   }

   // Parallel reduction
   if constexpr( SliceSize * ThreadsPerSegment <= Backend::getWarpSize() ) {
      /////
      // In this case, all threads participating in the reduction within one slice are in the same warp.
      // In case when ThreadsPerSegment = 2 and SliceSize = 4 warp size is 8, the mapping of threads
      // to segment elements is as follows:
      //
      //          +---------+--------+--------+--------+
      //          |   T0/W0 |  T4/W0 |  T0/W0 |  T4/W0 |
      // Slice 0: |   T1/W0 |  T5/W0 |  T1/W0 |  T5/W0 |
      //          |   T2/W0 |  T6/W0 |  T2/W0 |  T6/W0 |
      //          |   T3/W0 |  T7/W0 |  T3/W0 |  T7/W0 |
      //          +---------+--------+--------+--------+--------+
      //          |   T8/W1 | T12/W1 |  T8/W1 | T12/W1 |  T8/W1 |
      // Slice 1: |   T9/W1 | T13/W1 |  T9/W1 | T13/W1 |  T9/W1 |
      //          |  T10/W1 | T14/W1 | T10/W1 | T14/W1 | T10/W1 |
      //          |  T11/W1 | T15/W1 | T11/W1 | T15/W1 | T11/W1 |
      //          +---------+--------+--------+--------+--------+
      //
      // We can use warp shuffles to perform reduction within each row (i.e. within each segment).
      /////
      __syncthreads();
   #if defined( __HIP__ )
      if( ThreadsPerSegment > 16 ) {
         result = reduce( result, __shfl_down( result, 16 * SliceSize ) );
         result = reduce( result, __shfl_down( result, 8 * SliceSize ) );
         result = reduce( result, __shfl_down( result, 4 * SliceSize ) );
         result = reduce( result, __shfl_down( result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down( result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 8 ) {
         result = reduce( result, __shfl_down( result, 8 * SliceSize ) );
         result = reduce( result, __shfl_down( result, 4 * SliceSize ) );
         result = reduce( result, __shfl_down( result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down( result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 4 ) {
         result = reduce( result, __shfl_down( result, 4 * SliceSize ) );
         result = reduce( result, __shfl_down( result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down( result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 2 ) {
         result = reduce( result, __shfl_down( result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down( result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 1 )
         result = reduce( result, __shfl_down( result, SliceSize ) );
   #else
      if( ThreadsPerSegment > 16 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 8 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 4 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 2 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 * SliceSize ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, SliceSize ) );
      }
      else if( ThreadsPerSegment > 1 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, SliceSize ) );
      }
   #endif
      // Write the result
      if( inSliceThreadIdx < SliceSize && segmentIdx >= begin && segmentIdx < end ) {
         keep( segmentIdx, result );
      }
   }
   else {  // more than one warp is involved in the reduction - use shared memory
      /////
      // In this case we first store intermediate results from each thread into shared memory
      // and then we reshufle the data in shared memory so that each row contains all
      // intermediate results for one segment. Finally, we perform reduction within each row using
      // intrawarp reduction. We demonstrate this for the case when ThreadsPerSegment = 2 and
      // SliceSize = 4 (as described above) but the warp size is just 4 (to keep the example simple).
      // In this case the mapping of threads to segment elements is as follows:
      //
      //          +---------+--------+--------+--------+
      //          |   T0/W0 |  T4/W1 |  T0/W0 |  T4/W1 |
      // Slice 0: |   T1/W0 |  T5/W1 |  T1/W0 |  T5/W1 |
      //          |   T2/W0 |  T6/W1 |  T2/W0 |  T6/W1 |
      //          |   T3/W0 |  T7/W1 |  T3/W0 |  T7/W1 |
      //          +---------+--------+--------+--------+--------+
      //          |   T8/W2 | T12/W3 |  T8/W2 | T12/W3 |  T8/W2 |
      // Slice 1: |   T9/W2 | T13/W3 |  T9/W2 | T13/W3 |  T9/W2 |
      //          |  T10/W2 | T14/W3 | T10/W2 | T14/W3 | T10/W2 |
      //          |  T11/W2 | T15/W3 | T11/W2 | T15/W3 | T11/W2 |
      //          +---------+--------+--------+--------+--------+
      //
      // Threads within the same warp have already performed reduction when fetching segment elements.
      // Now we need to perform the rest of the reduction, i.e. the following threads:
      //
      //          +---------+--------+
      //          |   T0/W0 |  T4/W1 |
      // Slice 0: |   T1/W0 |  T5/W1 |
      //          |   T2/W0 |  T6/W1 |
      //          |   T3/W0 |  T7/W1 |
      //          +---------+--------+
      //          |   T8/W2 | T12/W3 |
      // Slice 1: |   T9/W2 | T13/W3 |
      //          |  T10/W2 | T14/W3 |
      //          |  T11/W2 | T15/W3 |
      //          +---------+--------+
      //
      // This is, in fact, a 2D array having BlockSize/ThreadsPerSegment rows (we ignore the parameters `begin` and `end`
      // for simplicity) and ThreadsPerSegment columns. First, each thread writes its intermediate result into shared
      // memory at position [row][column], where where `row` and `column` are given as follows:
      /////
      const Index column = inSliceThreadIdx / SliceSize;
      const Index row = ( threadIdx.x / ( SliceSize * ThreadsPerSegment ) ) * SliceSize  // the first row of this slice
                      + inSliceThreadIdx % SliceSize;                                    // the in-slice index within the row
      TNL_ASSERT_LT( row, BlockSize / ThreadsPerSegment, "" );
      __shared__ ReturnType sharedResults[ BlockSize ];

      if( column < ThreadsPerSegment ) {
         TNL_ASSERT_LT( row * ThreadsPerSegment + column, BlockSize, "" );
         sharedResults[ row * ThreadsPerSegment + column ] = result;
      }
      __syncthreads();

      /////
      // Now we read the values from the array in such a way that the elements which need to be reduced
      // are located in the same warp, i.e. the mapping of threads to segment elements is as follows:
      //
      //          +---------+--------+
      //          |   T0/W0 |  T1/W0 |
      // Slice 0: |   T2/W0 |  T3/W0 |
      //          |   T4/W1 |  T5/W1 |
      //          |   T6/W1 |  T7/W1 |
      //          +---------+--------+
      //          |   T8/W2 |  T9/W2 |
      // Slice 1: |  T10/W2 | T11/W2 |
      //          |  T12/W3 | T13/W3 |
      //          |  T14/W3 | T15/W3 |
      //          +---------+--------+
      /////
      result = sharedResults[ threadIdx.x ];
      __syncwarp();
   #if defined( __HIP__ )
      if( ThreadsPerSegment > 16 ) {
         result = reduce( result, __shfl_down( result, 16 ) );
         result = reduce( result, __shfl_down( result, 8 ) );
         result = reduce( result, __shfl_down( result, 4 ) );
         result = reduce( result, __shfl_down( result, 2 ) );
         result = reduce( result, __shfl_down( result, 1 ) );
      }
      else if( ThreadsPerSegment > 8 ) {
         result = reduce( result, __shfl_down( result, 8 ) );
         result = reduce( result, __shfl_down( result, 4 ) );
         result = reduce( result, __shfl_down( result, 2 ) );
         result = reduce( result, __shfl_down( result, 1 ) );
      }
      else if( ThreadsPerSegment > 4 ) {
         result = reduce( result, __shfl_down( result, 4 ) );
         result = reduce( result, __shfl_down( result, 2 ) );
         result = reduce( result, __shfl_down( result, 1 ) );
      }
      else if( ThreadsPerSegment > 2 ) {
         result = reduce( result, __shfl_down( result, 2 ) );
         result = reduce( result, __shfl_down( result, 1 ) );
      }
      else if( ThreadsPerSegment > 1 )
         result = reduce( result, __shfl_down( result, 1 ) );
   #else
      if( ThreadsPerSegment > 16 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 16 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
      }
      else if( ThreadsPerSegment > 8 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 8 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
      }
      else if( ThreadsPerSegment > 4 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 4 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
      }
      else if( ThreadsPerSegment > 2 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 2 ) );
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
      }
      else if( ThreadsPerSegment > 1 ) {
         result = reduce( result, __shfl_down_sync( 0xFFFFFFFF, result, 1 ) );
      }
   #endif

      /////
      // Finaly, we write the result. The mapping of threads having the result of the reduction is as follows:
      //
      //          +---------+
      //          |   T0/W0 |
      // Slice 0: |   T2/W0 |
      //          |   T4/W1 |
      //          |   T6/W1 |
      //          +---------+
      //          |   T8/W2 |
      // Slice 1: |  T10/W2 |
      //          |  T12/W3 |
      //          |  T14/W3 |
      //          +---------+
      /////
      const Index firstSliceInBlock =
         ( ( blockIdx.x + gridIdx * Backend::getMaxGridXSize() ) * BlockSize ) / ( SliceSize * ThreadsPerSegment );
      const Index currentSegmentIdx =
         ( begin / SliceSize ) * SliceSize + firstSliceInBlock * SliceSize + threadIdx.x / ThreadsPerSegment;

      if( ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 && currentSegmentIdx >= begin && currentSegmentIdx < end )
         keep( currentSegmentIdx, result );
   }
#endif
}

template< int BlockSize,
          int ThreadsPerSegment,
          typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
reduceSegmentsSlicedEllpackKernel( const int gridIdx,
                                   const Segments segments,
                                   IndexBegin begin,
                                   IndexEnd end,
                                   Fetch fetch,
                                   Reduction reduce,
                                   ResultKeeper keep,
                                   const Value identity )
{
   static_assert( ThreadsPerSegment <= Backend::getWarpSize(),
                  "ThreadsPerSegment must be less than or equal to the warp size." );
   if constexpr( Segments::getOrganization() == RowMajorOrder )
      reduceSegmentsRowMajorSlicedEllpackKernel< BlockSize, ThreadsPerSegment >(
         gridIdx, segments, begin, end, fetch, reduce, keep, identity );
   else
      reduceSegmentsColumnMajorSlicedEllpackKernel< BlockSize, ThreadsPerSegment >(
         gridIdx, segments, begin, end, fetch, reduce, keep, identity );
}
}  // namespace TNL::Algorithms::Segments::detail
