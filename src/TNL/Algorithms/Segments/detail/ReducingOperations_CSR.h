// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include "LaunchConfiguration.h"
#include <TNL/TypeTraits.h>
#include "CSRHostScheduling.h"
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_CSR_Tiered.h"
#include "ReducingKernels_CSR_Uniform.h"
#include "ReducingOperationsBase.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct ReducingOperations< CSRView< Device, Index > > : public ReducingOperationsBase< CSRView< Device, Index > >
{
   using SegmentsViewType = CSRView< Device, Index >;
   using ConstViewType = typename SegmentsViewType::ConstViewType;
   using DeviceType = Device;
   using IndexType = std::remove_const_t< Index >;
   using ConstOffsetsView = typename SegmentsViewType::ConstOffsetsView;

   /**
    * \brief Resolves the `Default` mapping of `launchConfig` in place and
    * rejects the mappings without a GPU reduction kernel.
    *
    * The `Default` mapping resolves to the `Uniform` mapping with a warp of
    * threads per segment, the library-wide GPU default for CSR-based formats.
    * CSR has no dedicated kernel for the `Adaptive` mapping (that strategy
    * belongs to AdaptiveCSR, which falls back to CSR when its kernel cannot be
    * used), so it resolves to the `Default` mapping as well.
    * The `Tiered` mapping supports at most 256 threads per
    * segment, the block size of its kernels.
    */
   static void
   resolveLaunchConfiguration( LaunchConfiguration& launchConfig )
   {
      switch( launchConfig.getThreadsToSegmentsMapping() ) {
         case ThreadsToSegmentsMapping::Default:
         case ThreadsToSegmentsMapping::Adaptive:
            launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Uniform );
            launchConfig.setThreadsPerSegmentCountToWarpSize();
            break;
         case ThreadsToSegmentsMapping::Uniform:
            break;
         case ThreadsToSegmentsMapping::Tiered:
            // a block of the Tiered kernels with 256 threads processes 256 / threadsPerSegment segments
            if( launchConfig.getThreadsPerSegmentCount() > 256 )
               throw std::invalid_argument(
                  "Unsupported threads per segment count ( " + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                  + " ) for the Tiered reduction of CSR segments." );
            break;
         default:
            throw std::invalid_argument( "Unsupported threads to segments mapping for the reduction of CSR segments." );
      }
   }

   /**
    * \brief Launches the `Uniform` reduction kernel for `threadsPerSegment`
    * threads per segment.
    *
    * `singleWarpKernel` and `multipleWarpsKernel` return the kernel for the
    * threads per segment count given as `std::integral_constant`, and `launch`
    * launches the returned kernel.
    * The single-warp kernel is used when the threads of a segment fit into one
    * warp of the current device, i.e. also for 64 threads per segment on AMD
    * GPUs with 64-wide wavefronts, the multiple-warps kernel otherwise.
    */
   template< typename SingleWarpKernel, typename MultipleWarpsKernel, typename Launch >
   static void
   launchUniformKernel(
      int threadsPerSegment,
      int warpSize,
      SingleWarpKernel&& singleWarpKernel,
      MultipleWarpsKernel&& multipleWarpsKernel,
      Launch&& launch )
   {
      dispatchThreadsPerSegment< 2, 4, 8, 16, 32, 64, 128 >(
         threadsPerSegment,
         [ & ]( auto threadsPerSegment )
         {
            constexpr int TPS = decltype( threadsPerSegment )::value;
            if constexpr( TPS <= Backend::getMaxWarpSize() ) {
               if( TPS <= warpSize ) {
                  launch( singleWarpKernel( threadsPerSegment ) );
                  return;
               }
            }
            // the multiple-warps kernel needs at least one whole warp per segment
            if constexpr( TPS > 32 )
               launch( multipleWarpsKernel( threadsPerSegment ) );
         },
         "CSR" );
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsSequential(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      const LaunchConfiguration& launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();

      if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > || std::is_same_v< Device, TNL::Devices::Host > ) {
         // Long segments are reduced into several independent partial results, so that the loop
         // is not bound by the latency of the reduction operation (e.g. the FMA in SpMV). Short
         // segments are faster with a single partial result.
         auto l = [ offsets, fetch, reduction, storer, identity ]( const Index segmentIdx ) mutable
         {
            const IndexType begin = offsets[ segmentIdx ];
            const IndexType end = offsets[ segmentIdx + 1 ];
            using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
            ReturnType aux0 = identity;
            IndexType globalIdx = begin;
            if( end - begin >= 16 ) {
               ReturnType aux1 = identity;
               ReturnType aux2 = identity;
               ReturnType aux3 = identity;
               for( ; globalIdx + 3 < end; globalIdx += 4 ) {
                  if constexpr( callableArgumentCount< Fetch >() == 3 ) {
                     const IndexType localIdx = globalIdx - begin;
                     aux0 = reduction( aux0, fetch( segmentIdx, localIdx, globalIdx ) );
                     aux1 = reduction( aux1, fetch( segmentIdx, localIdx + 1, globalIdx + 1 ) );
                     aux2 = reduction( aux2, fetch( segmentIdx, localIdx + 2, globalIdx + 2 ) );
                     aux3 = reduction( aux3, fetch( segmentIdx, localIdx + 3, globalIdx + 3 ) );
                  }
                  else {
                     aux0 = reduction( aux0, fetch( globalIdx ) );
                     aux1 = reduction( aux1, fetch( globalIdx + 1 ) );
                     aux2 = reduction( aux2, fetch( globalIdx + 2 ) );
                     aux3 = reduction( aux3, fetch( globalIdx + 3 ) );
                  }
               }
               // the reduction may take its first argument by a non-const reference
               aux0 = reduction( aux0, aux1 );
               aux2 = reduction( aux2, aux3 );
               aux0 = reduction( aux0, aux2 );
            }
            for( ; globalIdx < end; globalIdx++ ) {
               if constexpr( callableArgumentCount< Fetch >() == 3 )
                  aux0 = reduction( aux0, fetch( segmentIdx, globalIdx - begin, globalIdx ) );
               else
                  aux0 = reduction( aux0, fetch( globalIdx ) );
            }
            storer( segmentIdx, aux0 );
         };

         if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
            for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ )
               l( segmentIdx );
         }
         else {
            forSegmentsHost( offsets, begin, end, l );
         }
      }
      else {
         auto l = [ offsets, fetch, reduction, storer, identity ] __cuda_callable__( const Index segmentIdx ) mutable
         {
            const IndexType begin = offsets[ segmentIdx ];
            const IndexType end = offsets[ segmentIdx + 1 ];
            using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
            ReturnType aux = identity;
            if constexpr( callableArgumentCount< Fetch >() == 3 ) {
               IndexType localIdx = 0;
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  aux = reduction( aux, fetch( segmentIdx, localIdx++, globalIdx ) );
            }
            else {
               for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
                  aux = reduction( aux, fetch( globalIdx ) );
            }
            storer( segmentIdx, aux );
         };
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegments(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
         if( end <= begin )
            return;
         resolveLaunchConfiguration( launchConfig );
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            reduceSegmentsSequential( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform ) {
                  auto launch = [ & ]( auto kernel )
                  {
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        gridIdx,
                        segments.getConstView(),
                        begin,
                        end,
                        fetch,
                        reduction,
                        storer,
                        identity );
                  };
                  launchUniformKernel(
                     launchConfig.getThreadsPerSegmentCount(),
                     warpSize,
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform<
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform_MultipleWarps<
                           256,
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     launch );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered ) {
                  constexpr auto kernel = reduceSegments_CSR_Tiered<
                     ConstViewType,
                     IndexType,
                     std::remove_reference_t< Fetch >,
                     std::remove_reference_t< Reduction >,
                     std::remove_reference_t< ResultStorer >,
                     Value,
                     256 >;
                  Backend::launchKernelAsync(
                     kernel,
                     launch_config,
                     gridIdx,
                     launchConfig.getThreadsPerSegmentCount(),
                     segments.getConstView(),
                     begin,
                     end,
                     fetch,
                     reduction,
                     storer,
                     identity );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else {
         reduceSegmentsSequential( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
      }
   }

   template< typename Array, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   reduceSegmentsWithIndexesSequential(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();
      auto segmentIndexes_view = segmentIndexes.getConstView();

      auto l = [ offsets, segmentIndexes_view, fetch, reduction, storer, identity ] __cuda_callable__(
                  const Index segmentIdx_idx ) mutable
      {
         const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
         const IndexType begin = offsets[ segmentIdx ];
         const IndexType end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
         ReturnType result = identity;
         if constexpr( callableArgumentCount< Fetch >() == 3 ) {
            IndexType localIdx = 0;
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               result = reduction( result, fetch( segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               result = reduction( result, fetch( globalIdx ) );
         }
         storer( segmentIdx_idx, segmentIdx, result );
      };

      if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
         for( IndexType segmentIdx = 0; segmentIdx < segmentIndexes.getSize(); segmentIdx++ )
            l( segmentIdx );
      }
      else if constexpr( std::is_same_v< Device, TNL::Devices::Host > ) {
#ifdef HAVE_OPENMP
         #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( IndexType segmentIdx = 0; segmentIdx < segmentIndexes.getSize(); segmentIdx++ )
            l( segmentIdx );
      }
      else {
         Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
      }
   }

   template< typename Array, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   reduceSegmentsWithSegmentIndexes(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ArrayView = typename Array::ConstViewType;
      if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
         resolveLaunchConfiguration( launchConfig );
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            reduceSegmentsWithIndexesSequential( segments, segmentIndexes, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = segmentIndexes.getSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );

            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform ) {
                  auto launch = [ & ]( auto kernel )
                  {
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        gridIdx,
                        segments.getConstView(),
                        segmentIndexes.getConstView(),
                        fetch,
                        reduction,
                        storer,
                        identity );
                  };
                  launchUniformKernel(
                     launchConfig.getThreadsPerSegmentCount(),
                     warpSize,
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform_WithIndexes<
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           ArrayView,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform_MultipleWarps_WithIndexes<
                           256,
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           ArrayView,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     launch );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered ) {
                  constexpr auto kernel = reduceSegments_CSR_Tiered_WithIndexes<
                     ConstViewType,
                     ArrayView,
                     IndexType,
                     std::remove_reference_t< Fetch >,
                     std::remove_reference_t< Reduction >,
                     std::remove_reference_t< ResultStorer >,
                     Value,
                     256 >;
                  Backend::launchKernelAsync(
                     kernel,
                     launch_config,
                     gridIdx,
                     launchConfig.getThreadsPerSegmentCount(),
                     segments.getConstView(),
                     segmentIndexes.getConstView(),
                     fetch,
                     reduction,
                     storer,
                     identity );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else {
         reduceSegmentsWithIndexesSequential( segments, segmentIndexes, fetch, reduction, storer, identity, launchConfig );
      }
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsSequentialWithArgument(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      const LaunchConfiguration& launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();

      auto l = [ offsets, fetch, reduction, storer, identity ] __cuda_callable__( const Index segmentIdx ) mutable
      {
         const IndexType begin = offsets[ segmentIdx ];
         const IndexType end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
         ReturnType result = identity;
         IndexType argument = 0;
         IndexType localIdx = 0;
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
            if constexpr( callableArgumentCount< Fetch >() == 3 )
               reduction( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
            else
               reduction( result, fetch( globalIdx ), argument, localIdx );
            localIdx++;
         }
         bool emptySegment = ( begin == end );
         storer( segmentIdx, argument, result, emptySegment );
      };

      if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
         for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
      }
      else if constexpr( std::is_same_v< Device, TNL::Devices::Host > ) {
         forSegmentsHost( offsets, begin, end, l );
      }
      else {
         Algorithms::parallelFor< Device >( begin, end, l );
      }
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Fetch,
      typename Reduction,
      typename ResultStorer,
      typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument(
      const ConstViewType& segments,
      IndexBegin begin,
      IndexEnd end,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
         if( end <= begin )
            return;
         resolveLaunchConfiguration( launchConfig );
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            reduceSegmentsSequentialWithArgument( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform ) {
                  auto launch = [ & ]( auto kernel )
                  {
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        gridIdx,
                        segments.getConstView(),
                        begin,
                        end,
                        fetch,
                        reduction,
                        storer,
                        identity );
                  };
                  launchUniformKernel(
                     launchConfig.getThreadsPerSegmentCount(),
                     warpSize,
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform_WithArgument<
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform_MultipleWarps_WithArgument<
                           256,
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     launch );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered ) {
                  constexpr auto kernel = reduceSegments_CSR_Tiered_WithArgument<
                     ConstViewType,
                     IndexType,
                     std::remove_reference_t< Fetch >,
                     std::remove_reference_t< Reduction >,
                     std::remove_reference_t< ResultStorer >,
                     Value,
                     256 >;
                  Backend::launchKernelAsync(
                     kernel,
                     launch_config,
                     gridIdx,
                     launchConfig.getThreadsPerSegmentCount(),
                     segments.getConstView(),
                     begin,
                     end,
                     fetch,
                     reduction,
                     storer,
                     identity );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else {
         reduceSegmentsSequentialWithArgument( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
      }
   }

   template< typename Array, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   reduceSegmentsWithIndexesAndArgumentSequential(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();
      auto segmentIndexes_view = segmentIndexes.getConstView();

      auto l = [ offsets, segmentIndexes_view, fetch, reduction, storer, identity ] __cuda_callable__(
                  const Index segmentIdx_idx ) mutable
      {
         const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
         const IndexType begin = offsets[ segmentIdx ];
         const IndexType end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
         ReturnType result = identity;
         IndexType argument = 0;
         {
            IndexType localIdx = 0;
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++, localIdx++ )
               if constexpr( callableArgumentCount< Fetch >() == 3 )
                  reduction( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
               else
                  reduction( result, fetch( globalIdx ), argument, localIdx );
         }
         bool emptySegment = ( begin == end );
         storer( segmentIdx_idx, segmentIdx, argument, result, emptySegment );
      };

      if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
         for( IndexType segmentIdx = 0; segmentIdx < segmentIndexes.getSize(); segmentIdx++ )
            l( segmentIdx );
      }
      else if constexpr( std::is_same_v< Device, TNL::Devices::Host > ) {
#ifdef HAVE_OPENMP
         #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( IndexType segmentIdx = 0; segmentIdx < segmentIndexes.getSize(); segmentIdx++ )
            l( segmentIdx );
      }
      else {
         Algorithms::parallelFor< Device >( 0, segmentIndexes.getSize(), l );
      }
   }

   template< typename Array, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
   static void
   reduceSegmentsWithSegmentIndexesAndArgument(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      Fetch&& fetch,
      Reduction&& reduction,
      ResultStorer&& storer,
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ArrayView = typename Array::ConstViewType;
      if constexpr( std::is_same_v< Device, TNL::Devices::GPU > ) {
         resolveLaunchConfiguration( launchConfig );
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
             && launchConfig.getThreadsPerSegmentCount() == 1 )
         {
            reduceSegmentsWithIndexesAndArgumentSequential(
               segments, segmentIndexes, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = segmentIndexes.getSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );

            const int warpSize = Backend::getWarpSize( Backend::getDevice() );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Uniform ) {
                  auto launch = [ & ]( auto kernel )
                  {
                     Backend::launchKernelAsync(
                        kernel,
                        launch_config,
                        gridIdx,
                        segments.getConstView(),
                        segmentIndexes.getConstView(),
                        fetch,
                        reduction,
                        storer,
                        identity );
                  };
                  launchUniformKernel(
                     launchConfig.getThreadsPerSegmentCount(),
                     warpSize,
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform_WithIndexesAndArgument<
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           ArrayView,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     []( auto threadsPerSegment )
                     {
                        return reduceSegments_CSR_Uniform_MultipleWarps_WithIndexesAndArgument<
                           256,
                           decltype( threadsPerSegment )::value,
                           ConstViewType,
                           ArrayView,
                           IndexType,
                           std::remove_reference_t< Fetch >,
                           std::remove_reference_t< Reduction >,
                           std::remove_reference_t< ResultStorer >,
                           Value >;
                     },
                     launch );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Tiered ) {
                  constexpr auto kernel = reduceSegments_CSR_Tiered_WithIndexesAndArgument<
                     ConstViewType,
                     ArrayView,
                     IndexType,
                     std::remove_reference_t< Fetch >,
                     std::remove_reference_t< Reduction >,
                     std::remove_reference_t< ResultStorer >,
                     Value,
                     256 >;
                  Backend::launchKernelAsync(
                     kernel,
                     launch_config,
                     gridIdx,
                     launchConfig.getThreadsPerSegmentCount(),
                     segments.getConstView(),
                     segmentIndexes.getConstView(),
                     fetch,
                     reduction,
                     storer,
                     identity );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else {
         reduceSegmentsWithIndexesAndArgumentSequential(
            segments, segmentIndexes, fetch, reduction, storer, identity, launchConfig );
      }
   }
};
}  // namespace TNL::Algorithms::Segments::detail
