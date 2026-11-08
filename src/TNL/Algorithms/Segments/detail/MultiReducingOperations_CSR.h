// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
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
#include "CSRHostScheduling.h"
#include "MultiReducingKernels_CSR_LanesOverReductions.h"
#include "MultiReducingKernels_CSR_Uniform.h"
#include "MultiReducingOperationsBase.h"
#include "ReducingOperations_CSR.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct MultiReducingOperations< CSRView< Device, Index > > : public MultiReducingOperationsBase< CSRView< Device, Index > >
{
   using SegmentsViewType = CSRView< Device, Index >;
   using Base = MultiReducingOperationsBase< SegmentsViewType >;
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;

   //! \brief Returns `true` for the devices on the host, i.e. \ref TNL::Devices::Host and \ref TNL::Devices::Sequential.
   static constexpr bool
   isHostDevice()
   {
      return std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential >;
   }

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
    * \brief Implementation of \ref TNL::Algorithms::Segments::multiReduceAllSegmentsInTiles.
    *
    * On GPUs with the Uniform mapping of at most a warp of threads per segment, all tiles are
    * processed at once by \ref multiReduceSegments_CSR_Uniform. Otherwise, the general
    * implementation is used.
    */
   template< int TileSize, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   multiReduceAllSegmentsInTiles(
      const SegmentsViewType& segments,
      int reductions,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      if constexpr( std::is_same_v< DeviceType, Devices::GPU > ) {
         ReducingOperations< SegmentsViewType >::resolveLaunchConfiguration( launchConfig );
         const int threadsPerSegment = launchConfig.getThreadsPerSegmentCount();
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && threadsPerSegment <= Backend::getWarpSize( Backend::getDevice() ) && threadsPerSegment <= 32 )
         {
            using FetchType = std::decay_t< Fetch >;
            using ValueType = MultiReductionFetchValue< IndexType, FetchType >;
            const IndexType segmentsCount = segments.getSegmentCount();
            if( segmentsCount == 0 )
               return;
            auto launchTiles = [ & ]( auto threadsPerSegment_, auto tileSize_ )
            {
               constexpr int TPS = decltype( threadsPerSegment_ )::value;
               constexpr int ActualTileSize = decltype( tileSize_ )::value;
               constexpr auto kernel = multiReduceSegments_CSR_Uniform<
                  TPS,
                  ActualTileSize,
                  SegmentsViewType,
                  IndexType,
                  FetchType,
                  std::decay_t< Reduction >,
                  std::decay_t< ResultStorer >,
                  ValueType >;
               Backend::LaunchConfiguration config;
               config.blockSize.x = 256;
               dim3 blocksCount;
               dim3 gridsCount;
               Backend::setupThreads(
                  config.blockSize, blocksCount, gridsCount, static_cast< std::size_t >( segmentsCount ) * TPS );
               const int tiles = roundUpDivision( reductions, ActualTileSize );
               const int maxGridY = Backend::getMaxGridYSize();
               for( int firstTile = 0; firstTile < tiles; firstTile += maxGridY )
                  for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
                     Backend::setupGrid( blocksCount, gridsCount, gridIdx, config.gridSize );
                     config.gridSize.y = TNL::min( maxGridY, tiles - firstTile );
                     Backend::launchKernelAsync(
                        kernel,
                        config,
                        gridIdx,
                        segments,
                        IndexType( 0 ),
                        segmentsCount,
                        firstTile,
                        reductions,
                        fetch,
                        reduction,
                        storer,
                        static_cast< ValueType >( identity ) );
                  }
               Backend::streamSynchronize( config.stream );
            };
            // when all reductions fit into one tile, the tile has exactly their size, so that the threads do not
            // keep and check the unused partial results
            auto launch = [ & ]( auto threadsPerSegment_ )
            {
               if( reductions < TileSize )
                  dispatchMultiReductionSize< TileSize - 1 >(
                     reductions,
                     [ & ]( auto tileSize_ )
                     {
                        launchTiles( threadsPerSegment_, tileSize_ );
                     } );
               else
                  launchTiles( threadsPerSegment_, std::integral_constant< int, TileSize >{} );
            };
            switch( threadsPerSegment ) {
               case 1:
                  launch( std::integral_constant< int, 1 >{} );
                  return;
               case 2:
                  launch( std::integral_constant< int, 2 >{} );
                  return;
               case 4:
                  launch( std::integral_constant< int, 4 >{} );
                  return;
               case 8:
                  launch( std::integral_constant< int, 8 >{} );
                  return;
               case 16:
                  launch( std::integral_constant< int, 16 >{} );
                  return;
               case 32:
                  launch( std::integral_constant< int, 32 >{} );
                  return;
               default:
                  break;
            }
         }
      }
      if constexpr( isHostDevice() ) {
         multiReduceSegmentsHost< TileSize, false >( segments, reductions, fetch, reduction, storer, identity );
         return;
      }
      Base::template multiReduceAllSegmentsInTiles< TileSize >(
         segments,
         reductions,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< ResultStorer >( storer ),
         identity,
         launchConfig );
   }

   /**
    * \brief Implementation of \ref Segments_multiReduceAllSegments_identity.
    *
    * On the host, \ref multiReduceSegmentsHost is used. Otherwise, the general implementation is used.
    */
   template< int Size, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   multiReduceAllSegments(
      const SegmentsViewType& segments,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      if constexpr( isHostDevice() )
         multiReduceSegmentsHost< Size, true >( segments, Size, fetch, reduction, storer, identity );
      else
         Base::template multiReduceAllSegments< Size >(
            segments,
            std::forward< Fetch >( fetch ),
            std::forward< Reduction >( reduction ),
            std::forward< ResultStorer >( storer ),
            identity,
            launchConfig );
   }

   /**
    * \brief Multireduction in CSR segments on the host.
    *
    * Every thread processes a contiguous block of segments with approximately the same number of
    * elements (see \ref forSegmentsHost), sequentially for \ref TNL::Devices::Sequential. The
    * reductions are computed in tiles of `TileSize` reductions:
    *
    * - With `TilesInSegment`, all tiles of a segment are computed one after another, so the
    *   elements of the segment are read from the cache for all tiles but the first one. This suits
    *   the fetch of the reduction `j` reading consecutive data for consecutive `j` (e.g. a row of a
    *   row-major dense matrix), which are then read only once.
    * - Otherwise, every tile is computed for all segments before the next one, so only the data of
    *   one tile are read at once. This suits the fetch of the reduction `j` reading data far from
    *   those for `j + 1` (e.g. a column of a column-major dense matrix), which stay in the cache
    *   for the elements of the next segments.
    *
    * Within a tile, the inner loop over the reductions has a constant trip count and the partial
    * results are kept in a local array, so the compiler can vectorize it. The tiles of at most 8
    * reductions are computed with two independent partial results per reduction, so that the loop
    * is not bound by the latency of the reduction operation (e.g. the FMA in SpMM).
    */
   template< int TileSize, bool TilesInSegment, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   multiReduceSegmentsHost(
      const SegmentsViewType& segments,
      int reductions,
      Fetch fetch,
      Reduction reduction,
      ResultStorer storer,
      const Value& identity )
   {
      using FetchType = std::decay_t< Fetch >;
      using ValueType = MultiReductionFetchValue< IndexType, FetchType >;
      const auto offsets = segments.getOffsets();
      const IndexType segmentsCount = segments.getSegmentCount();
      const int tiles = roundUpDivision( reductions, TileSize );

      // computes the reductions firstReduction, ..., firstReduction + count - 1 of the segment
      auto reduceTile =
         [ offsets, fetch, reduction, storer, identity ]( IndexType segmentIdx, int firstReduction, auto count ) mutable
      {
         const IndexType begin = offsets[ segmentIdx ];
         const IndexType end = offsets[ segmentIdx + 1 ];
         auto fetchValue = [ & ]( IndexType globalIdx, int j )
         {
            if constexpr( callableArgumentCount< FetchType >() == 4 )
               return fetch( segmentIdx, globalIdx - begin, globalIdx, firstReduction + j );
            else
               return fetch( globalIdx, firstReduction + j );
         };
         // the elements are processed by Unroll independent sets of the partial results, so that there are at
         // least 8 independent reductions in the inner loop
         constexpr int Unroll = TileSize <= 4 ? 4 : TileSize <= 8 ? 2 : 1;
         ValueType results[ Unroll ][ TileSize ];
         for( int u = 0; u < Unroll; u++ )
            for( int j = 0; j < TileSize; j++ )
               results[ u ][ j ] = identity;
         IndexType globalIdx = begin;
         if constexpr( Unroll > 1 ) {
            for( ; globalIdx + Unroll - 1 < end; globalIdx += Unroll )
               for( int u = 0; u < Unroll; u++ )
                  for( int j = 0; j < count; j++ )
                     results[ u ][ j ] = reduction( results[ u ][ j ], fetchValue( globalIdx + u, j ) );
            for( int u = 1; u < Unroll; u++ )
               for( int j = 0; j < count; j++ )
                  results[ 0 ][ j ] = reduction( results[ 0 ][ j ], results[ u ][ j ] );
         }
         for( ; globalIdx < end; globalIdx++ )
            for( int j = 0; j < count; j++ )
               results[ 0 ][ j ] = reduction( results[ 0 ][ j ], fetchValue( globalIdx, j ) );
         for( int j = 0; j < count; j++ )
            storer( segmentIdx, firstReduction + j, results[ 0 ][ j ] );
      };
      // the full tiles have the constant trip count of the inner loop
      auto reduceSegmentTile = [ reduceTile, reductions ]( IndexType segmentIdx, int tile ) mutable
      {
         const int firstReduction = tile * TileSize;
         if( firstReduction + TileSize <= reductions )
            reduceTile( segmentIdx, firstReduction, std::integral_constant< int, TileSize >{} );
         else
            reduceTile( segmentIdx, firstReduction, reductions - firstReduction );
      };
      auto forAllSegments = [ & ]( auto& f )
      {
         if constexpr( std::is_same_v< DeviceType, Devices::Sequential > ) {
            for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ )
               f( segmentIdx );
         }
         else
            forSegmentsHost( offsets, IndexType( 0 ), segmentsCount, f );
      };

      if constexpr( TilesInSegment ) {
         auto processSegment = [ reduceSegmentTile, tiles ]( IndexType segmentIdx ) mutable
         {
            for( int tile = 0; tile < tiles; tile++ )
               reduceSegmentTile( segmentIdx, tile );
         };
         forAllSegments( processSegment );
      }
      else {
         for( int tile = 0; tile < tiles; tile++ ) {
            auto processSegment = [ reduceSegmentTile, tile ]( IndexType segmentIdx ) mutable
            {
               reduceSegmentTile( segmentIdx, tile );
            };
            forAllSegments( processSegment );
         }
      }
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
         // the long segments skipped by the kernel are counted and stored in the mapped page-locked buffers
         auto& longSegmentsCount = getLongSegmentsCountBuffer();
         auto& longSegments = getLongSegmentsBuffer();
         longSegmentsCount[ 0 ] = 0;
         const int longSegmentsCapacity = longSegments.getSize() / 3;
         constexpr auto kernel = multiReduceSegments_CSR_LanesOverReductions<
            Lanes,
            ReductionsPerLane,
            SegmentsViewType,
            IndexType,
            FetchType,
            std::decay_t< Reduction >,
            std::decay_t< ResultStorer >,
            ValueType >;
         // the tiles of reductions are processed by the blocks with the same blockIdx.y, so that all of them
         // run at once even when there are only a few segments
         const int tiles = roundUpDivision( reductions, Lanes * ReductionsPerLane );
         const int maxGridY = Backend::getMaxGridYSize();
         for( int firstTile = 0; firstTile < tiles; firstTile += maxGridY )
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launchConfig.gridSize );
               launchConfig.gridSize.y = TNL::min( maxGridY, tiles - firstTile );
               Backend::launchKernelAsync(
                  kernel,
                  launchConfig,
                  gridIdx,
                  segments,
                  IndexType( 0 ),
                  segmentsCount,
                  maxSegmentSize,
                  longSegmentsCount.getData(),
                  longSegments.getData(),
                  longSegmentsCapacity,
                  firstTile,
                  reductions,
                  fetch,
                  reduction,
                  storer,
                  static_cast< ValueType >( identity ) );
            }
         Backend::streamSynchronize( launchConfig.stream );
         const int longCount = longSegmentsCount[ 0 ];
         if( longCount > 0 ) {
            // when the buffer was too small, the long segments are searched for again and the buffer is enlarged for
            // the next calls
            const bool stored = longCount <= longSegmentsCapacity;
            multiReduceLongSegmentsLanesOverReductions< Lanes, ReductionsPerLane >(
               segments,
               maxSegmentSize,
               reductions,
               fetch,
               reduction,
               storer,
               static_cast< ValueType >( identity ),
               stored ? longSegments.getData() : nullptr,
               longCount );
            if( ! stored )
               longSegments.setSize( 3 * longCount );
         }
      }
      else if constexpr( isHostDevice() ) {
         // on the host, a thread processes all reductions of its segments
         multiReduceSegmentsHost< Lanes * ReductionsPerLane, true >( segments, reductions, fetch, reduction, storer, identity );
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
    *
    * The long segments are given by `longSegments` as stored by \ref multiReduceSegments_CSR_LanesOverReductions,
    * or they are searched for when it is `nullptr`. The pieces are set up on the host in the mapped
    * page-locked memory and the partial results are stored in a buffer kept for the next calls, so that
    * there is no allocation and copying in the common case, which would take much longer than the
    * reduction of a few long segments.
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
      const Value& identity,
      const IndexType* longSegments,
      int longCount )
   {
      using IndexVector = Containers::Vector< IndexType, Devices::GPU, IndexType >;
      using HostIndexVector = Containers::Vector< IndexType, Devices::Host, IndexType >;
      using ValueVector = Containers::Vector< Value, Devices::GPU, IndexType >;
      using IndexView = Containers::ArrayView< const IndexType, Devices::GPU, IndexType >;
      constexpr int tileSize = Lanes * ReductionsPerLane;
      // the pieces are smaller only when the maximal size is set explicitly to a small value
      const IndexType pieceSize =
         TNL::max( IndexType( 1 ), TNL::min( maxSegmentSize, IndexType( lanesOverReductionsPieceSize< Lanes >() ) ) );

      // the long segments as triples ( segment index, first element, end of the elements )
      std::vector< std::array< IndexType, 3 > > longList( longCount );
      if( longSegments != nullptr ) {
         for( int i = 0; i < longCount; i++ )
            longList[ i ] = { longSegments[ 3 * i ], longSegments[ 3 * i + 1 ], longSegments[ 3 * i + 2 ] };
      }
      else {
         const auto offsets = segments.getOffsets();
         auto isLong = [ = ] __cuda_callable__( IndexType segmentIdx ) -> bool
         {
            return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ] > maxSegmentSize;
         };
         const IndexVector found = Algorithms::compress< IndexVector >( IndexType( 0 ), segments.getSegmentCount(), isLong );
         IndexVector bounds( 2 * longCount );
         auto boundsView = bounds.getView();
         const auto foundView = found.getConstView();
         Algorithms::parallelFor< Devices::GPU >(
            IndexType( 0 ),
            IndexType( longCount ),
            [ = ] __cuda_callable__( IndexType i ) mutable
            {
               boundsView[ 2 * i ] = offsets[ foundView[ i ] ];
               boundsView[ 2 * i + 1 ] = offsets[ foundView[ i ] + 1 ];
            } );
         const HostIndexVector hostFound( found );
         const HostIndexVector hostBounds( bounds );
         for( int i = 0; i < longCount; i++ )
            longList[ i ] = { hostFound[ i ], hostBounds[ 2 * i ], hostBounds[ 2 * i + 1 ] };
      }
      // the kernel stores the segments in an arbitrary order, the order of the pieces does not matter for the results
      // but sorting makes the computation deterministic
      std::sort( longList.begin(), longList.end() );

      // the pieces are set up in the mapped page-locked buffer: the segments, the first and the end elements of the
      // pieces, the long segments and the first pieces of the long segments
      IndexType piecesCount = 0;
      for( const auto& [ segmentIdx, begin, end ] : longList )
         piecesCount += roundUpDivision( end - begin, pieceSize );
      auto& buffer = getPiecesBuffer();
      const IndexType bufferSize = 3 * piecesCount + 2 * longCount + 1;
      if( buffer.getSize() < bufferSize )
         buffer.setSize( bufferSize );
      IndexType* pieceSegments = buffer.getData();
      IndexType* pieceBegins = pieceSegments + piecesCount;
      IndexType* pieceEnds = pieceBegins + piecesCount;
      IndexType* longSegmentIndexes = pieceEnds + piecesCount;
      IndexType* firstPieces = longSegmentIndexes + longCount;
      IndexType piece = 0;
      for( int i = 0; i < longCount; i++ ) {
         const auto& [ segmentIdx, begin, end ] = longList[ i ];
         longSegmentIndexes[ i ] = segmentIdx;
         firstPieces[ i ] = piece;
         for( IndexType pieceBegin = begin; pieceBegin < end; pieceBegin += pieceSize ) {
            pieceSegments[ piece ] = segmentIdx;
            pieceBegins[ piece ] = pieceBegin;
            pieceEnds[ piece ] = TNL::min( pieceBegin + pieceSize, end );
            piece++;
         }
      }
      firstPieces[ longCount ] = piecesCount;

      // the partial results of all reductions of a piece are stored contiguously, including the padding of the last tile
      const int tiles = roundUpDivision( reductions, tileSize );
      const IndexType stride = static_cast< IndexType >( tiles ) * tileSize;
      static ValueVector partialResults;
      if( partialResults.getSize() < piecesCount * stride )
         partialResults.setSize( piecesCount * stride );

      Backend::LaunchConfiguration piecesConfig;
      piecesConfig.blockSize.x = lanesOverReductionsBlockSize;
      piecesConfig.gridSize.x = piecesCount;
      const int maxGridY = Backend::getMaxGridYSize();
      constexpr auto piecesKernel = multiReduceSegments_CSR_LanesOverReductions_Pieces<
         lanesOverReductionsBlockSize,
         Lanes,
         ReductionsPerLane,
         SegmentsViewType,
         IndexView,
         typename ValueVector::ViewType,
         IndexType,
         Fetch,
         Reduction,
         Value >;
      for( int firstTile = 0; firstTile < tiles; firstTile += maxGridY ) {
         piecesConfig.gridSize.y = TNL::min( maxGridY, tiles - firstTile );
         Backend::launchKernelAsync(
            piecesKernel,
            piecesConfig,
            segments,
            IndexView( pieceSegments, piecesCount ),
            IndexView( pieceBegins, piecesCount ),
            IndexView( pieceEnds, piecesCount ),
            firstTile,
            reductions,
            partialResults.getView(),
            stride,
            fetch,
            reduction,
            identity );
      }

      Backend::LaunchConfiguration combineConfig;
      combineConfig.blockSize.x = 256;
      combineConfig.gridSize.x = roundUpDivision( longCount * stride, IndexType( 256 ) );
      constexpr auto combineKernel = multiReduceSegments_CSR_LanesOverReductions_CombinePieces<
         IndexView,
         typename ValueVector::ConstViewType,
         IndexType,
         Reduction,
         ResultStorer >;
      Backend::launchKernelSync(
         combineKernel,
         combineConfig,
         IndexView( longSegmentIndexes, longCount ),
         IndexView( firstPieces, longCount + 1 ),
         reductions,
         partialResults.getConstView(),
         stride,
         reduction,
         storer );
   }

private:
#if defined( __HIP__ )
   template< typename T >
   using PinnedAllocator = Allocators::HipHost< T >;
#else
   template< typename T >
   using PinnedAllocator = Allocators::CudaHost< T >;
#endif
   template< typename T >
   using PinnedArray = Containers::Array< T, Devices::Host, IndexType, PinnedAllocator< T > >;

   // The buffers in the page-locked memory mapped to the address space of the device, so that the host can read and
   // write them without copying, which would add a noticeable latency to the product of small matrices. They are
   // allocated once, hence multiReduceAllSegmentsLanesOverReductions must not be called concurrently from several
   // host threads.

   // The number of the long segments found by multiReduceSegments_CSR_LanesOverReductions.
   static PinnedArray< int >&
   getLongSegmentsCountBuffer()
   {
      static PinnedArray< int > buffer( 1 );
      return buffer;
   }

   // The long segments stored by multiReduceSegments_CSR_LanesOverReductions, see its description.
   static PinnedArray< IndexType >&
   getLongSegmentsBuffer()
   {
      static PinnedArray< IndexType > buffer( 3 * 256 );
      return buffer;
   }

   // The pieces of the long segments, see multiReduceLongSegmentsLanesOverReductions.
   static PinnedArray< IndexType >&
   getPiecesBuffer()
   {
      static PinnedArray< IndexType > buffer;
      return buffer;
   }
};

}  // namespace TNL::Algorithms::Segments::detail
