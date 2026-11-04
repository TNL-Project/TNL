// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <type_traits>
#include <vector>

#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Allocators/CudaHost.h>
#include <TNL/Allocators/HipHost.h>
#include <TNL/Backend/DeviceInfo.h>
#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include "MultiReducingKernels_CSR_LanesOverReductions.h"
#include "MultiReducingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct MultiReducingOperations< CSRView< Device, Index > > : public MultiReducingOperationsBase< CSRView< Device, Index > >
{
   using SegmentsViewType = CSRView< Device, Index >;
   using Base = MultiReducingOperationsBase< SegmentsViewType >;
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;

   //! \brief The block size of the kernels of \ref multiReduceAllSegmentsLanesOverReductions.
   static constexpr int lanesOverReductionsBlockSize = 256;

   /**
    * \brief The number of elements of the pieces of the long segments in
    * \ref multiReduceAllSegmentsLanesOverReductions.
    *
    * Every group of `Lanes` threads of the block processes about 64 of them.
    */
   template< int Lanes >
   static constexpr int
   lanesOverReductionsPieceSize()
   {
      return lanesOverReductionsBlockSize / Lanes * 64;
   }

   /**
    * \brief The default maximal size of the segments processed by one group of threads in
    * \ref multiReduceAllSegmentsLanesOverReductions on GPUs, the longer segments are split into pieces.
    *
    * A group of threads processes one segment, so a segment much longer than the average number of
    * elements per group of all groups running at once would delay the whole kernel. The segments are
    * split only when they are several times longer than this average.
    *
    * \param elements The number of elements of all segments.
    */
   template< int Lanes >
   static IndexType
   getDefaultMaxSegmentSize( IndexType elements )
   {
      const int device = Backend::getDevice();
      // the number of groups which can run at once, assuming 2048 threads per multiprocessor
      const std::size_t groups = static_cast< std::size_t >( Backend::getDeviceMultiprocessors( device ) ) * 2048 / Lanes;
      const std::size_t averageGroupElements = roundUpDivision( static_cast< std::size_t >( elements ), groups );
      return static_cast< IndexType >(
         TNL::max( 8 * averageGroupElements, static_cast< std::size_t >( lanesOverReductionsPieceSize< Lanes >() ) ) );
   }

   /**
    * \brief Implementation of \ref TNL::Algorithms::Segments::multiReduceAllSegmentsLanesOverReductions.
    *
    * On GPUs, a group of `Lanes` threads processes every segment by
    * \ref multiReduceSegments_CSR_LanesOverReductions, the segments with more than `maxSegmentSize`
    * elements are processed afterwards by \ref multiReduceLongSegmentsLanesOverReductions. On the host,
    * the general implementation is used.
    */
   template< int Lanes, int ReductionsPerLane, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   multiReduceAllSegmentsLanesOverReductions(
      const SegmentsViewType& segments,
      int reductions,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      IndexType maxSegmentSize )
   {
      if constexpr( std::is_same_v< DeviceType, Devices::GPU > ) {
         using FetchType = std::decay_t< Fetch >;
         using ValueType = MultiReductionFetchValue< IndexType, FetchType >;

         const IndexType segmentsCount = segments.getSegmentCount();
         if( segmentsCount == 0 )
            return;
         Backend::LaunchConfiguration launchConfig;
         launchConfig.blockSize.x = lanesOverReductionsBlockSize;
         dim3 blocksCount;
         dim3 gridsCount;
         Backend::setupThreads(
            launchConfig.blockSize, blocksCount, gridsCount, static_cast< std::size_t >( segmentsCount ) * Lanes );
         if( maxSegmentSize <= 0 )
            maxSegmentSize = getDefaultMaxSegmentSize< Lanes >( segments.getOffsets().getElement( segmentsCount ) );
         // The number of the long segments skipped by the kernel is counted in the mapped page-locked memory, so that the
         // host can reset and read it without copying, which would add a noticeable latency to the product of small
         // matrices. The buffer is allocated once, hence the function must not be called concurrently from several
         // host threads.
#if defined( __HIP__ )
         using PinnedAllocator = Allocators::HipHost< int >;
#else
         using PinnedAllocator = Allocators::CudaHost< int >;
#endif
         static Containers::Array< int, Devices::Host, int, PinnedAllocator > longSegmentsCount( 1 );
         longSegmentsCount[ 0 ] = 0;
         constexpr auto kernel = multiReduceSegments_CSR_LanesOverReductions<
            Lanes,
            ReductionsPerLane,
            SegmentsViewType,
            IndexType,
            FetchType,
            std::decay_t< Reduction >,
            std::decay_t< ResultStorer >,
            ValueType >;
         for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
            Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
            Backend::launchKernelAsync(
               kernel,
               launchConfig,
               gridIdx,
               segments,
               IndexType( 0 ),
               segmentsCount,
               maxSegmentSize,
               longSegmentsCount.getData(),
               reductions,
               fetch,
               reduction,
               storer,
               static_cast< ValueType >( identity ) );
         }
         Backend::streamSynchronize( launchConfig.stream );
         if( longSegmentsCount[ 0 ] > 0 )
            multiReduceLongSegmentsLanesOverReductions< Lanes, ReductionsPerLane >(
               segments, maxSegmentSize, reductions, fetch, reduction, storer, static_cast< ValueType >( identity ) );
      }
      else
         Base::template multiReduceAllSegmentsLanesOverReductions< Lanes, ReductionsPerLane >(
            segments,
            reductions,
            std::forward< Fetch >( fetch ),
            std::forward< Reduction >( reduction ),
            std::forward< ResultStorer >( storer ),
            identity,
            maxSegmentSize );
   }

   /**
    * \brief Computes the reductions of the segments longer than `maxSegmentSize`, which are skipped
    * by \ref multiReduceSegments_CSR_LanesOverReductions.
    *
    * The segments are split into pieces processed by thread blocks with
    * \ref multiReduceSegments_CSR_LanesOverReductions_Pieces and the partial results of the pieces
    * are combined by \ref multiReduceSegments_CSR_LanesOverReductions_CombinePieces.
    */
   template< int Lanes, int ReductionsPerLane, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   multiReduceLongSegmentsLanesOverReductions(
      const SegmentsViewType& segments,
      IndexType maxSegmentSize,
      int reductions,
      const Fetch& fetch,
      const Reduction& reduction,
      const ResultStorer& storer,
      const Value& identity )
   {
      using IndexVector = Containers::Vector< IndexType, Devices::GPU, IndexType >;
      using HostIndexVector = Containers::Vector< IndexType, Devices::Host, IndexType >;
      using ValueVector = Containers::Vector< Value, Devices::GPU, IndexType >;
      constexpr int tileSize = Lanes * ReductionsPerLane;
      // the pieces are smaller only when the maximal size is set explicitly to a small value
      const IndexType pieceSize =
         TNL::max( IndexType( 1 ), TNL::min( maxSegmentSize, IndexType( lanesOverReductionsPieceSize< Lanes >() ) ) );

      const auto offsets = segments.getOffsets();
      auto isLong = [ = ] __cuda_callable__( IndexType segmentIdx ) -> bool
      {
         return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ] > maxSegmentSize;
      };
      const IndexType segmentsCount = segments.getSegmentCount();
      const IndexVector longSegments = Algorithms::compress< IndexVector >( IndexType( 0 ), segmentsCount, isLong );
      const IndexType longCount = longSegments.getSize();

      // the pieces are set up on the host, there are only a few long segments
      IndexVector bounds( 2 * longCount );
      auto boundsView = bounds.getView();
      const auto longSegmentsView = longSegments.getConstView();
      Algorithms::parallelFor< Devices::GPU >(
         IndexType( 0 ),
         longCount,
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            boundsView[ 2 * i ] = offsets[ longSegmentsView[ i ] ];
            boundsView[ 2 * i + 1 ] = offsets[ longSegmentsView[ i ] + 1 ];
         } );
      const HostIndexVector hostBounds( bounds );
      const HostIndexVector hostLongSegments( longSegments );
      HostIndexVector hostFirstPieces( longCount + 1 );
      std::vector< IndexType > pieceSegments;
      std::vector< IndexType > pieceBegins;
      std::vector< IndexType > pieceEnds;
      for( IndexType i = 0; i < longCount; i++ ) {
         hostFirstPieces[ i ] = pieceSegments.size();
         for( IndexType begin = hostBounds[ 2 * i ]; begin < hostBounds[ 2 * i + 1 ]; begin += pieceSize ) {
            pieceSegments.push_back( hostLongSegments[ i ] );
            pieceBegins.push_back( begin );
            pieceEnds.push_back( TNL::min( begin + pieceSize, hostBounds[ 2 * i + 1 ] ) );
         }
      }
      const IndexType piecesCount = pieceSegments.size();
      hostFirstPieces[ longCount ] = piecesCount;
      const IndexVector firstPieces( hostFirstPieces );
      IndexVector devicePieceSegments( piecesCount );
      IndexVector devicePieceBegins( piecesCount );
      IndexVector devicePieceEnds( piecesCount );
      devicePieceSegments = pieceSegments;
      devicePieceBegins = pieceBegins;
      devicePieceEnds = pieceEnds;
      ValueVector partialResults( piecesCount * tileSize );

      Backend::LaunchConfiguration piecesConfig;
      piecesConfig.blockSize.x = lanesOverReductionsBlockSize;
      piecesConfig.gridSize.x = piecesCount;
      constexpr auto piecesKernel = multiReduceSegments_CSR_LanesOverReductions_Pieces<
         lanesOverReductionsBlockSize,
         Lanes,
         ReductionsPerLane,
         SegmentsViewType,
         typename IndexVector::ConstViewType,
         typename ValueVector::ViewType,
         IndexType,
         Fetch,
         Reduction,
         Value >;
      Backend::launchKernelAsync(
         piecesKernel,
         piecesConfig,
         segments,
         devicePieceSegments.getConstView(),
         devicePieceBegins.getConstView(),
         devicePieceEnds.getConstView(),
         reductions,
         partialResults.getView(),
         fetch,
         reduction,
         identity );

      Backend::LaunchConfiguration combineConfig;
      combineConfig.blockSize.x = 256;
      combineConfig.gridSize.x = roundUpDivision( longCount * tileSize, IndexType( 256 ) );
      constexpr auto combineKernel = multiReduceSegments_CSR_LanesOverReductions_CombinePieces<
         tileSize,
         typename IndexVector::ConstViewType,
         typename ValueVector::ConstViewType,
         IndexType,
         Reduction,
         ResultStorer >;
      Backend::launchKernelSync(
         combineKernel,
         combineConfig,
         longSegments.getConstView(),
         firstPieces.getConstView(),
         reductions,
         partialResults.getConstView(),
         reduction,
         storer );
   }
};

}  // namespace TNL::Algorithms::Segments::detail
