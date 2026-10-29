// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/detail/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/detail/FetchLambdaAdapter.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Backend/SharedMemory.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Kernel reducing a contiguous range of ChunkedEllpack segments.
 *
 * The ChunkedEllpack format (see
 * \ref TNL::Algorithms::Segments::ChunkedEllpack) splits the segments into
 * slices of `chunksInSlice` chunks and every segment of a slice occupies
 * one or more chunks of the same size.
 * One block processes one slice and every thread of the block one chunk: it
 * reduces the elements of its chunk (contiguous for the row-major
 * organization, strided by `chunksInSlice` for the column-major organization)
 * and stores the partial result in shared memory.
 * Then one thread per segment of the slice reduces the partial
 * results of the chunks of its segment.
 * The shared memory size must be `blockDim.x * sizeof( ReturnType )`.
 */
template< typename SegmentsView, typename Index, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
__global__
void
reduceSegments_ChunkedEllpack(
   SegmentsView segments,
   Index gridIdx,
   Index begin,
   Index end,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index firstSlice = segments.getSegmentToSliceMappingView()[ begin ];
   const Index lastSlice = segments.getSegmentToSliceMappingView()[ end - 1 ];

   const Index sliceIdx = firstSlice + gridIdx * Backend::getMaxGridXSize() + blockIdx.x;
   if( sliceIdx > lastSlice )
      return;

   ReturnType* chunksResults = Backend::getSharedMemory< ReturnType >();
   __shared__ Segments::detail::ChunkedEllpackSliceInfo< Index > sliceInfo;

   if( threadIdx.x == 0 )
      sliceInfo = segments.getSlicesView()[ sliceIdx ];
   chunksResults[ threadIdx.x ] = identity;
   __syncthreads();

   const Index sliceOffset = sliceInfo.pointer;
   const Index chunkSize = sliceInfo.chunkSize;

   if constexpr( callableArgumentCount< Fetch >() == 3 ) {
      const Index chunkIdx = sliceIdx * segments.getChunksInSlice() + threadIdx.x;
      const Index segmentIdx = segments.getChunksToSegmentsMappingView()[ chunkIdx ];
      Index firstChunkOfSegment = 0;
      if( segmentIdx != sliceInfo.firstSegment )
         firstChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx - 1 ];
      Index localIdx = ( threadIdx.x - firstChunkOfSegment ) * chunkSize;

      if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
         Index begin = sliceOffset + threadIdx.x * chunkSize;  // threadIdx.x = chunkIdx within the slice
         Index end = begin + chunkSize;
         for( Index j = begin; j < end; j++ )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j ) );
      }
      else {
         const Index begin = sliceOffset + threadIdx.x;  // threadIdx.x = chunkIdx within the slice
         const Index end = begin + segments.getChunksInSlice() * chunkSize;
         for( Index j = begin; j < end; j += segments.getChunksInSlice() )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( segmentIdx, localIdx++, j ) );
      }
   }
   else {
      if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
         Index begin = sliceOffset + threadIdx.x * chunkSize;  // threadIdx.x = chunkIdx within the slice
         Index end = begin + chunkSize;
         for( Index j = begin; j < end; j++ )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( j ) );
      }
      else {
         const Index begin = sliceOffset + threadIdx.x;  // threadIdx.x = chunkIdx within the slice
         const Index end = begin + segments.getChunksInSlice() * chunkSize;
         for( Index j = begin; j < end; j += segments.getChunksInSlice() )
            chunksResults[ threadIdx.x ] = reduction( chunksResults[ threadIdx.x ], fetch( j ) );
      }
   }

   __syncthreads();

   if( threadIdx.x < sliceInfo.size ) {
      const Index segment = sliceInfo.firstSegment + threadIdx.x;
      Index chunkIndex = 0;
      if( threadIdx.x != 0 )
         chunkIndex = segments.getSegmentToChunkMappingView()[ segment - 1 ];
      const Index lastChunk = segments.getSegmentToChunkMappingView()[ segment ];
      ReturnType result = identity;
      while( chunkIndex < lastChunk )
         result = reduction( result, chunksResults[ chunkIndex++ ] );
      if( segment >= begin && segment < end )
         storer( segment, result );
   }
#endif
}

/**
 * \brief Kernel reducing the ChunkedEllpack segments given
 * explicitly by `segmentIndexes`.
 *
 * Unlike \ref reduceSegments_ChunkedEllpack, one thread is assigned to every
 * entry of `segmentIndexes` and it sequentially reduces all chunks of its
 * segment, since the given segments are not grouped into slices.
 */
template<
   typename SegmentsView,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegments_ChunkedEllpack_WithIndexes(
   SegmentsView segments,
   ArrayView segmentIndexes,
   Index gridIdx,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentIdx_idx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
      return;
   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index sliceIdx = segments.getSegmentToSliceMappingView()[ segmentIdx ];

   Index firstChunkOfSegment( 0 );
   if( segmentIdx != segments.getSlicesView()[ sliceIdx ].firstSegment ) {
      firstChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx - 1 ];
   }

   const Index lastChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx ];
   const Index segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
   const Index sliceOffset = segments.getSlicesView()[ sliceIdx ].pointer;
   const Index chunkSize = segments.getSlicesView()[ sliceIdx ].chunkSize;

   const Index segmentSize = segmentChunksCount * chunkSize;
   Value result = identity;
   if( SegmentsView::getOrganization() == RowMajorOrder ) {
      Index begin = sliceOffset + firstChunkOfSegment * chunkSize;
      Index end = begin + segmentSize;
      Index localIdx = 0;
      for( Index j = begin; j < end; j++ )
         result = reduction( result, FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
   }
   else {
      Index localIdx = 0;
      for( Index chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
         Index begin = sliceOffset + firstChunkOfSegment + chunkIdx;
         Index end = begin + segments.getChunksInSlice() * chunkSize;
         for( Index j = begin; j < end; j += segments.getChunksInSlice() ) {
            result = reduction( result, FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx++, j ) );
         }
      }
   }
   storer( segmentIdx_idx, segmentIdx, result );
#endif
}

/**
 * \brief Kernel reducing a contiguous range of
 * ChunkedEllpack segments with argument.
 *
 * It works the same way as \ref reduceSegments_ChunkedEllpack, but every
 * thread also stores the local index of the partial result of its chunk
 * in shared memory, right after the partial results, so the shared
 * memory size must account for both arrays.
 */
template< typename SegmentsView, typename Index, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
__global__
void
reduceSegments_ChunkedEllpack_WithArgument(
   SegmentsView segments,
   Index gridIdx,
   Index begin,
   Index end,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index firstSlice = segments.getSegmentToSliceMappingView()[ begin ];
   const Index lastSlice = segments.getSegmentToSliceMappingView()[ end - 1 ];

   const Index sliceIdx = firstSlice + gridIdx * Backend::getMaxGridXSize() + blockIdx.x;
   if( sliceIdx > lastSlice )
      return;

   ReturnType* chunksResults = Backend::getSharedMemory< ReturnType >();
   // Align the arguments array to alignof(Index) since sizeof(ReturnType) may
   // not be a multiple of Index alignment (e.g. sizeof(float)=4 but alignof(long)=8).
   // Round up to the next aligned boundary: (value + align-1) rounds up, & ~(align-1)
   // clears the low bits to produce a multiple of align.
   const std::size_t argumentsOffset = ( blockDim.x * sizeof( ReturnType ) + alignof( Index ) - 1 ) & ~( alignof( Index ) - 1 );
   Index* chunksArguments = reinterpret_cast< Index* >( reinterpret_cast< unsigned char* >( chunksResults ) + argumentsOffset );
   __shared__ Segments::detail::ChunkedEllpackSliceInfo< Index > sliceInfo;

   if( threadIdx.x == 0 )
      sliceInfo = segments.getSlicesView()[ sliceIdx ];
   chunksResults[ threadIdx.x ] = identity;
   chunksArguments[ threadIdx.x ] = 0;
   __syncthreads();

   const Index sliceOffset = sliceInfo.pointer;
   const Index chunkSize = sliceInfo.chunkSize;

   const Index chunkIdx = sliceIdx * segments.getChunksInSlice() + threadIdx.x;
   const Index segmentIdx = segments.getChunksToSegmentsMappingView()[ chunkIdx ];
   Index firstChunkOfSegment = 0;
   if( segmentIdx != sliceInfo.firstSegment )
      firstChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx - 1 ];
   Index localIdx = ( threadIdx.x - firstChunkOfSegment ) * chunkSize;

   if constexpr( SegmentsView::getOrganization() == Segments::RowMajorOrder ) {
      Index begin = sliceOffset + threadIdx.x * chunkSize;  // threadIdx.x = chunkIdx within the slice
      Index end = begin + chunkSize;
      for( Index j = begin; j < end; j++, localIdx++ )
         reduction(
            chunksResults[ threadIdx.x ],
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, j ),
            chunksArguments[ threadIdx.x ],
            localIdx );
   }
   else {
      const Index begin = sliceOffset + threadIdx.x;  // threadIdx.x = chunkIdx within the slice
      const Index end = begin + segments.getChunksInSlice() * chunkSize;
      for( Index j = begin; j < end; j += segments.getChunksInSlice(), localIdx++ )
         reduction(
            chunksResults[ threadIdx.x ],
            detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, j ),
            chunksArguments[ threadIdx.x ],
            localIdx );
   }

   __syncthreads();

   if( threadIdx.x < sliceInfo.size ) {
      const Index segment = sliceInfo.firstSegment + threadIdx.x;
      Index chunkIndex = 0;
      if( threadIdx.x != 0 )
         chunkIndex = segments.getSegmentToChunkMappingView()[ segment - 1 ];
      const Index lastChunk = segments.getSegmentToChunkMappingView()[ segment ];
      ReturnType result = identity;
      Index argument = 0;
      while( chunkIndex < lastChunk ) {
         reduction( result, chunksResults[ chunkIndex ], argument, chunksArguments[ chunkIndex ] );
         chunkIndex++;
      }
      if( segment >= begin && segment < end ) {
         bool emptySegment = ( chunkSize == 0 );
         storer( segment, argument, result, emptySegment );
      }
   }
#endif
}

/**
 * \brief Kernel reducing the ChunkedEllpack segments given explicitly
 * by `segmentIndexes` with argument.
 *
 * One thread is assigned to every entry of `segmentIndexes` and it sequentially
 * reduces all chunks of its segment together with the local index of the
 * partial result, like \ref reduceSegments_ChunkedEllpack_WithIndexes.
 */
template<
   typename SegmentsView,
   typename ArrayView,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename ResultStorer,
   typename Value >
__global__
void
reduceSegments_ChunkedEllpack_WithIndexesAndArgument(
   SegmentsView segments,
   ArrayView segmentIndexes,
   Index gridIdx,
   Fetch fetch,
   Reduction reduction,
   ResultStorer storer,
   Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   const Index segmentIdx_idx = ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( segmentIdx_idx >= segmentIndexes.getSize() )
      return;
   TNL_ASSERT_LT( segmentIdx_idx, segmentIndexes.getSize(), "" );
   const Index segmentIdx = segmentIndexes[ segmentIdx_idx ];
   const Index sliceIdx = segments.getSegmentToSliceMappingView()[ segmentIdx ];

   Index firstChunkOfSegment( 0 );
   if( segmentIdx != segments.getSlicesView()[ sliceIdx ].firstSegment ) {
      firstChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx - 1 ];
   }

   const Index lastChunkOfSegment = segments.getSegmentToChunkMappingView()[ segmentIdx ];
   const Index segmentChunksCount = lastChunkOfSegment - firstChunkOfSegment;
   const Index sliceOffset = segments.getSlicesView()[ sliceIdx ].pointer;
   const Index chunkSize = segments.getSlicesView()[ sliceIdx ].chunkSize;

   const Index segmentSize = segmentChunksCount * chunkSize;
   Value result = identity;
   Index argument = 0;
   if( SegmentsView::getOrganization() == RowMajorOrder ) {
      Index begin = sliceOffset + firstChunkOfSegment * chunkSize;
      Index end = begin + segmentSize;
      Index localIdx = 0;
      for( Index j = begin; j < end; j++ ) {
         reduction( result, FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, j ), argument, localIdx );
         localIdx++;
      }
   }
   else {
      Index localIdx = 0;
      for( Index chunkIdx = 0; chunkIdx < segmentChunksCount; chunkIdx++ ) {
         Index begin = sliceOffset + firstChunkOfSegment + chunkIdx;
         Index end = begin + segments.getChunksInSlice() * chunkSize;
         for( Index j = begin; j < end; j += segments.getChunksInSlice() ) {
            reduction( result, FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, j ), argument, localIdx );
            localIdx++;
         }
      }
   }
   bool emptySegment = ( chunkSize == 0 );
   storer( segmentIdx_idx, segmentIdx, argument, result, emptySegment );
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
