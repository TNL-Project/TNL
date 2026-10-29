// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Algorithms/detail/CudaReductionKernel.h>
#include <TNL/Backend/LaunchHelpers.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel for the Uniform thread mapping, reducing a contiguous range of
 * segments of Ellpack segments with the row-major organization.
 *
 * `ThreadsPerSegment` consecutive threads are assigned to every
 * segment of the range [`begin`, `end`).
 * All segments have the same size, so the global index of the first element of
 * a segment is just `segmentIdx * segmentSize` and no offsets have to be read.
 * The threads of a segment process its contiguous elements with stride
 * `ThreadsPerSegment` and their partial results are combined by a warp-shuffle
 * reduction, so `ThreadsPerSegment` must not exceed the warp size.
 * The column-major organization does not use this kernel, since the elements
 * of a segment are not contiguous there.
 *
 * See \ref reduceSegments_CSR_Uniform for the paper that inspired the
 * uniform number of threads per segment.
 */
template<
   int ThreadsPerSegment,
   typename Segments,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegments_Ellpack_Uniform(
   int gridIdx,
   const Segments segments,
   IndexBegin begin,
   IndexEnd end,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment;
   const bool active = ( segmentIdx < end );

   ReturnType result = identity;
   if( active ) {
      const Index segmentSize = segments.getSegmentSize();
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

      begin = segmentIdx * segmentSize;  // reusing begin and end variables - now they define
      end = begin + segmentSize;         // the range of the global indices

      // Calculate the result
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = laneIdx;
         for( Index globalIdx = begin + laneIdx; globalIdx < end;
              globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment )
         {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
         }
      }
      else {
         for( Index i = begin + laneIdx; i < end; i += ThreadsPerSegment )
            result = reduce( result, fetch( i ) );
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
   result = BlockReduce::template warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 )
      store( segmentIdx, result );

#endif
}

/**
 * \brief Kernel for the Uniform thread mapping, reducing the
 * segments given explicitly by `segmentIndexes` of Ellpack
 * segments with the row-major organization.
 *
 * `ThreadsPerSegment` consecutive threads are assigned to every entry of
 * `segmentIndexes`; otherwise it works the same way as
 * \ref reduceSegments_Ellpack_Uniform.
 */
template<
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegments_Ellpack_Uniform_WithIndexes(
   int gridIdx,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment;
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index segmentSize = segments.getSegmentSize();
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

      Index begin = segmentIdx * segmentSize;
      Index end = begin + segmentSize;

      // Calculate the result
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = laneIdx;
         for( Index globalIdx = begin + laneIdx; globalIdx < end;
              globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment )
         {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            result = reduce( result, fetch( segmentIdx, localIdx, globalIdx ) );
         }
      }
      else {
         for( Index i = begin + laneIdx; i < end; i += ThreadsPerSegment )
            result = reduce( result, fetch( i ) );
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceShfl< 256, Reduction, ReturnType >;
   result = BlockReduce::template warpReduce< ThreadsPerSegment >( reduce, result );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 ) {
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      store( segmentIdx_idx, segmentIdx, result );
   }

#endif
}

/**
 * \brief Kernel for the Uniform thread mapping, reducing a contiguous range of
 * segments with argument of Ellpack segments with the row-major organization.
 *
 * It works the same way as \ref reduceSegments_Ellpack_Uniform, but every
 * thread also tracks the local index of its partial result (e.g. the
 * position of the maximum) and the pairs of values and arguments are
 * combined by a warp-shuffle reduction with argument.
 */
template<
   int ThreadsPerSegment,
   typename Segments,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegments_Ellpack_Uniform_WithArgument(
   int gridIdx,
   const Segments segments,
   IndexBegin begin,
   IndexEnd end,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx = begin + Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment;
   const bool active = ( segmentIdx < end );

   ReturnType result = identity;
   Index argument = 0;
   if( active ) {
      const Index segmentSize = segments.getSegmentSize();
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

      begin = segmentIdx * segmentSize;  // reusing begin and end variables - now they define
      end = begin + segmentSize;         // the range of the global indices

      // Calculate the result
      if constexpr( callableArgumentCount< Fetch >() == 3 ) {
         Index localIdx = laneIdx;
         for( Index globalIdx = begin + laneIdx; globalIdx < end;
              globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment )
         {
            TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
            reduce( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
         }
      }
      else {
         for( Index i = begin + laneIdx; i < end; i += ThreadsPerSegment )
            reduce( result, fetch( i ), argument, i - begin );
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::template warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 ) {
      const Index segmentSize = segments.getSegmentSize();
      bool emptySegment = ( segmentSize == 0 );
      store( segmentIdx, argument_, result_, emptySegment );
   }

#endif
}

/**
 * \brief Kernel for the Uniform thread mapping, reducing the segments
 * given explicitly by `segmentIndexes` with argument of Ellpack
 * segments with the row-major organization.
 *
 * It combines \ref reduceSegments_Ellpack_Uniform_WithIndexes and
 * \ref reduceSegments_Ellpack_Uniform_WithArgument.
 */
template<
   int ThreadsPerSegment,
   typename Segments,
   typename ArrayView,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegments_Ellpack_Uniform_WithIndexesAndArgument(
   int gridIdx,
   const Segments segments,
   const ArrayView segmentIndexes,
   Fetch fetch,
   Reduction reduce,
   ResultStorer store,
   const Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using Index = typename Segments::IndexType;
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx = Backend::getGlobalThreadIdx_x( gridIdx ) / ThreadsPerSegment;
   const bool active = ( segmentIdx_idx < segmentIndexes.getSize() );

   ReturnType result = identity;
   Index argument = 0;
   if( active ) {
      TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index segmentSize = segments.getSegmentSize();
      const Index laneIdx = threadIdx.x & ( ThreadsPerSegment - 1 );  // & is cheaper than %

      Index begin = segmentIdx * segmentSize;
      Index end = begin + segmentSize;

      // Calculate the result
      Index localIdx = laneIdx;
      for( Index globalIdx = begin + laneIdx; globalIdx < end; globalIdx += ThreadsPerSegment, localIdx += ThreadsPerSegment ) {
         TNL_ASSERT_LT( globalIdx, segments.getStorageSize(), "" );
         if constexpr( callableArgumentCount< Fetch >() == 3 )
            reduce( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
         else
            reduce( result, fetch( globalIdx ), argument, localIdx );
      }
   }

   // Parallel reduction - all threads must participate in shuffle on HIP
   using BlockReduce = Algorithms::detail::CudaBlockReduceWithArgument< 256, Reduction, ReturnType, Index >;
   auto [ result_, argument_ ] = BlockReduce::template warpReduceWithArgument< ThreadsPerSegment >( reduce, result, argument );

   // Write the result
   if( active && ( threadIdx.x & ( ThreadsPerSegment - 1 ) ) == 0 ) {
      const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
      const Index segmentSize = segments.getSegmentSize();
      bool emptySegment = ( segmentSize == 0 );
      store( segmentIdx_idx, segmentIdx, argument_, result_, emptySegment );
   }

#endif
}

}  // namespace TNL::Algorithms::Segments::detail
