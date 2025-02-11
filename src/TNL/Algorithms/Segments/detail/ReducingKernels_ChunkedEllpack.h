// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments::detail {

template< typename SegmentsView, typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
__global__
void
ChunkedEllpackReduceSegmentsKernel( SegmentsView segments,
                                    Index gridIdx,
                                    Index begin,
                                    Index end,
                                    Fetch fetch,
                                    Reduction reduction,
                                    ResultKeeper keeper,
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

   if constexpr( argumentCount< Fetch >() == 3 ) {
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
         keeper( segment, result );
   }
#endif
}

template< typename SegmentsView,
          typename ArrayView,
          typename Index,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value >
__global__
void
ChunkedEllpackReduceSegmentsKernelWithIndexes( SegmentsView segments,
                                               ArrayView segmentIndexes,
                                               Index gridIdx,
                                               Index begin,
                                               Index end,
                                               Fetch fetch,
                                               Reduction reduction,
                                               ResultKeeper keeper,
                                               Value identity )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using ReturnType = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType;

   const Index segmentIdx_idx = begin + ( gridIdx * Backend::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( segmentIdx_idx >= end )
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
   keeper( segmentIdx_idx, segmentIdx, result );
#endif
}

template< typename SegmentsView, typename Index, typename Fetch, typename Reduction, typename ResultKeeper, typename Value >
__global__
void
ChunkedEllpackReduceSegmentsKernelWithArgument( SegmentsView segments,
                                                Index gridIdx,
                                                Index begin,
                                                Index end,
                                                Fetch fetch,
                                                Reduction reduction,
                                                ResultKeeper keeper,
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
   Index* chunksArguments = &chunksResults[ blockDim.x ];
   __shared__ Segments::detail::ChunkedEllpackSliceInfo< Index > sliceInfo;

   if( threadIdx.x == 0 )
      sliceInfo = segments.getSlicesView()[ sliceIdx ];
   chunksResults[ threadIdx.x ] = identity;
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
         reduction( chunksResults[ threadIdx.x ],
                    detail::FetchLambdaAdapter< Index, Fetch >::call( fetch, segmentIdx, localIdx, j ),
                    chunksArguments[ threadIdx.x ],
                    localIdx );
   }
   else {
      const Index begin = sliceOffset + threadIdx.x;  // threadIdx.x = chunkIdx within the slice
      const Index end = begin + segments.getChunksInSlice() * chunkSize;
      for( Index j = begin; j < end; j += segments.getChunksInSlice(), localIdx++ )
         reduction( chunksResults[ threadIdx.x ],
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
      if( segment >= begin && segment < end )
         keeper( segment, result, argument );
   }
#endif
}

}  // namespace TNL::Algorithms::Segments::detail
