// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_CSR.h"
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

      if constexpr( std::is_same_v< Device, TNL::Devices::Sequential > ) {
         for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
      }
      else if constexpr( std::is_same_v< Device, TNL::Devices::Host > ) {
#ifdef HAVE_OPENMP
         #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
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
         // The unspecified Default mapping resolves to a warp of threads per segment - the
         // library-wide GPU default for CSR-based formats.
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Default ) {
            launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Fixed );
            launchConfig.setThreadsPerSegmentCountToWarpSize();
         }
         // CSR has no dedicated kernel for the Adaptive mapping (that strategy belongs to
         // AdaptiveCSR), so it is treated as a synonym for the plain sequential {Fixed, 1} strategy.
         if( ( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
               && launchConfig.getThreadsPerSegmentCount() == 1 )
             || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Adaptive )
         {
            reduceSegmentsSequential( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel<
                              2,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel<
                              4,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel<
                              8,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel<
                              16,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel<
                              32,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernel<
                              256,
                              64,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernel<
                              256,
                              128,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }

                     default:
                        throw std::runtime_error(
                           "Unsupported number of threads per segment"
                           + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                           + ". It can be only 2, 4, 8, 16 or 32." );
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernel<
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
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
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
         // The unspecified Default mapping resolves to a warp of threads per segment - the
         // library-wide GPU default for CSR-based formats.
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Default ) {
            launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Fixed );
            launchConfig.setThreadsPerSegmentCountToWarpSize();
         }
         // CSR has no dedicated kernel for the Adaptive mapping (that strategy belongs to
         // AdaptiveCSR), so it is treated as a synonym for the plain sequential {Fixed, 1} strategy.
         if( ( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
               && launchConfig.getThreadsPerSegmentCount() == 1 )
             || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Adaptive )
         {
            reduceSegmentsWithIndexesSequential( segments, segmentIndexes, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = segmentIndexes.getSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes<
                              2,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes<
                              4,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes<
                              8,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes<
                              16,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes<
                              32,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexes<
                              256,
                              64,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexes<
                              256,
                              128,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }

                     default:
                        throw std::runtime_error(
                           "Unsupported number of threads per segment"
                           + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                           + ". It can be only 2, 4, 8, 16 or 32." );
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernelWithIndexes<
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
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
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
#ifdef HAVE_OPENMP
         #pragma omp parallel for firstprivate( l ) schedule( dynamic, 100 ), if( Devices::Host::isOMPEnabled() )
#endif
         for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ )
            l( segmentIdx );
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
         // The unspecified Default mapping resolves to a warp of threads per segment - the
         // library-wide GPU default for CSR-based formats.
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Default ) {
            launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Fixed );
            launchConfig.setThreadsPerSegmentCountToWarpSize();
         }
         // CSR has no dedicated kernel for the Adaptive mapping (that strategy belongs to
         // AdaptiveCSR), so it is treated as a synonym for the plain sequential {Fixed, 1} strategy.
         if( ( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
               && launchConfig.getThreadsPerSegmentCount() == 1 )
             || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Adaptive )
         {
            reduceSegmentsSequentialWithArgument( segments, begin, end, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument<
                              2,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument<
                              4,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument<
                              8,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument<
                              16,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument<
                              32,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithArgument<
                              256,
                              64,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithArgument<
                              256,
                              128,
                              ConstViewType,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }

                     default:
                        throw std::runtime_error(
                           "Unsupported number of threads per segment"
                           + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                           + ". It can be only 2, 4, 8, 16 or 32." );
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernelWithArgument<
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
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
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
         // The unspecified Default mapping resolves to a warp of threads per segment - the
         // library-wide GPU default for CSR-based formats.
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Default ) {
            launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Fixed );
            launchConfig.setThreadsPerSegmentCountToWarpSize();
         }
         // CSR has no dedicated kernel for the Adaptive mapping (that strategy belongs to
         // AdaptiveCSR), so it is treated as a synonym for the plain sequential {Fixed, 1} strategy.
         if( ( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
               && launchConfig.getThreadsPerSegmentCount() == 1 )
             || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Adaptive )
         {
            reduceSegmentsWithIndexesAndArgumentSequential(
               segments, segmentIndexes, fetch, reduction, storer, identity, launchConfig );
         }
         else {
            std::size_t threadsCount = segmentIndexes.getSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
                || launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping )
               threadsCount *= static_cast< std::size_t >( launchConfig.getThreadsPerSegmentCount() );
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < static_cast< IndexType >( gridsCount.x ); gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument<
                              2,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument<
                              4,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument<
                              8,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument<
                              16,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument<
                              32,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexesAndArgument<
                              256,
                              64,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexesAndArgument<
                              256,
                              128,
                              ConstViewType,
                              ArrayView,
                              IndexType,
                              std::remove_reference_t< Fetch >,
                              std::remove_reference_t< Reduction >,
                              std::remove_reference_t< ResultStorer >,
                              Value >;
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
                           break;
                        }

                     default:
                        throw std::runtime_error(
                           "Unsupported number of threads per segment"
                           + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                           + ". It can be only 2, 4, 8, 16 or 32." );
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernelWithIndexesAndArgument<
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
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
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
