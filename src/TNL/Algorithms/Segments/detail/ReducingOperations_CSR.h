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
   using IndexType = typename std::remove_const< Index >::type;
   using ConstOffsetsView = typename SegmentsViewType::ConstOffsetsView;

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsSequential( const ConstViewType& segments,
                             IndexBegin begin,
                             IndexEnd end,
                             Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                             Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                             ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                             const Value& identity,
                             const LaunchConfiguration& launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();

      auto l = [ offsets, fetch, reduction, keeper, identity ] __cuda_callable__( const Index segmentIdx ) mutable
      {
         const IndexType begin = offsets[ segmentIdx ];
         const IndexType end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
         ReturnType aux = identity;
         if constexpr( argumentCount< Fetch >() == 3 ) {
            IndexType localIdx = 0;
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               aux = reduction( aux, fetch( segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               aux = reduction( aux, fetch( globalIdx ) );
         }
         keeper( segmentIdx, aux );
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
      else
         Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegments( const ConstViewType& segments,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                   Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                   ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                   const Value& identity,
                   const LaunchConfiguration& launchConfig )
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
            reduceSegmentsSequential( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp )
               threadsCount *= (std::size_t) Backend::getWarpSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp ) {
                  constexpr auto kernel =
                     reduceSegmentsCSRVectorKernel< ConstViewType, IndexType, Fetch, Reduction, ResultKeeper, Value >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getConstView(), begin, end, fetch, reduction, keeper, identity );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel< 2,
                                                                                          ConstViewType,
                                                                                          IndexType,
                                                                                          Fetch,
                                                                                          Reduction,
                                                                                          ResultKeeper,
                                                                                          Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel< 4,
                                                                                          ConstViewType,
                                                                                          IndexType,
                                                                                          Fetch,
                                                                                          Reduction,
                                                                                          ResultKeeper,
                                                                                          Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel< 8,
                                                                                          ConstViewType,
                                                                                          IndexType,
                                                                                          Fetch,
                                                                                          Reduction,
                                                                                          ResultKeeper,
                                                                                          Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel< 16,
                                                                                          ConstViewType,
                                                                                          IndexType,
                                                                                          Fetch,
                                                                                          Reduction,
                                                                                          ResultKeeper,
                                                                                          Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernel< 32,
                                                                                          ConstViewType,
                                                                                          IndexType,
                                                                                          Fetch,
                                                                                          Reduction,
                                                                                          ResultKeeper,
                                                                                          Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernel< 256,
                                                                                            64,
                                                                                            ConstViewType,
                                                                                            IndexType,
                                                                                            Fetch,
                                                                                            Reduction,
                                                                                            ResultKeeper,
                                                                                            Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernel< 256,
                                                                                            128,
                                                                                            ConstViewType,
                                                                                            IndexType,
                                                                                            Fetch,
                                                                                            Reduction,
                                                                                            ResultKeeper,
                                                                                            Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }

                     default:
                        throw std::runtime_error( "Unsupported number of threads per segment"
                                                  + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                  + ". It can be only 2, 4, 8, 16 or 32." );
                        break;
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernel< ConstViewType,
                                                                                  IndexType,
                                                                                  Fetch,
                                                                                  Reduction,
                                                                                  ResultKeeper,
                                                                                  Value,
                                                                                  256 >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getConstView(), begin, end, fetch, reduction, keeper, identity );
               }
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         reduceSegmentsSequential( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value >
   static void
   reduceSegmentsWithIndexesSequential( const ConstViewType& segments,
                                        const Array& segmentIndexes,
                                        IndexBegin begin,
                                        IndexEnd end,
                                        Fetch&& fetch,
                                        Reduction&& reduction,
                                        ResultKeeper&& keeper,
                                        const Value& identity,
                                        LaunchConfiguration launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();
      auto segmentIndexes_view = segmentIndexes.getConstView();

      auto l = [ offsets, segmentIndexes_view, fetch, reduction, keeper, identity ] __cuda_callable__(
                  const Index segmentIdx_idx ) mutable
      {
         const IndexType segmentIdx = segmentIndexes_view[ segmentIdx_idx ];
         const IndexType begin = offsets[ segmentIdx ];
         const IndexType end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
         ReturnType result = identity;
         if constexpr( argumentCount< Fetch >() == 3 ) {
            IndexType localIdx = 0;
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               result = reduction( result, fetch( segmentIdx, localIdx++, globalIdx ) );
         }
         else {
            for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ )
               result = reduction( result, fetch( globalIdx ) );
         }
         keeper( segmentIdx_idx, segmentIdx, result );
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
      else
         Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value >
   static void
   reduceSegmentsWithSegmentIndexes( const ConstViewType& segments,
                                     const Array& segmentIndexes,
                                     IndexBegin begin,
                                     IndexEnd end,
                                     Fetch fetch,          // TODO: Fetch&& fetch does not work here with CUDA
                                     Reduction reduction,  // TODO: Reduction&& reduction does not work here with CUDA
                                     ResultKeeper keeper,  // TODO: ResultKeeper&& keeper does not work here with CUDA
                                     const Value& identity,
                                     LaunchConfiguration launchConfig )
   {
      using ArrayView = typename Array::ConstViewType;
      if constexpr( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
            reduceSegmentsWithIndexesSequential(
               segments, segmentIndexes, begin, end, fetch, reduction, keeper, identity, launchConfig );
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp )
               threadsCount *= (std::size_t) Backend::getWarpSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp ) {
                  constexpr auto kernel = reduceSegmentsCSRVectorKernelWithIndexes< ConstViewType,
                                                                                    ArrayView,
                                                                                    IndexType,
                                                                                    Fetch,
                                                                                    Reduction,
                                                                                    ResultKeeper,
                                                                                    Value >;
                  Backend::launchKernelAsync( kernel,
                                              launch_config,
                                              gridIdx,
                                              segments.getConstView(),
                                              segmentIndexes.getConstView(),
                                              begin,
                                              end,
                                              fetch,
                                              reduction,
                                              keeper,
                                              identity );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes< 2,
                                                                                                     ConstViewType,
                                                                                                     ArrayView,
                                                                                                     IndexType,
                                                                                                     Fetch,
                                                                                                     Reduction,
                                                                                                     ResultKeeper,
                                                                                                     Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes< 4,
                                                                                                     ConstViewType,
                                                                                                     ArrayView,
                                                                                                     IndexType,
                                                                                                     Fetch,
                                                                                                     Reduction,
                                                                                                     ResultKeeper,
                                                                                                     Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes< 8,
                                                                                                     ConstViewType,
                                                                                                     ArrayView,
                                                                                                     IndexType,
                                                                                                     Fetch,
                                                                                                     Reduction,
                                                                                                     ResultKeeper,
                                                                                                     Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes< 16,
                                                                                                     ConstViewType,
                                                                                                     ArrayView,
                                                                                                     IndexType,
                                                                                                     Fetch,
                                                                                                     Reduction,
                                                                                                     ResultKeeper,
                                                                                                     Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexes< 32,
                                                                                                     ConstViewType,
                                                                                                     ArrayView,
                                                                                                     IndexType,
                                                                                                     Fetch,
                                                                                                     Reduction,
                                                                                                     ResultKeeper,
                                                                                                     Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexes< 256,
                                                                                                       64,
                                                                                                       ConstViewType,
                                                                                                       ArrayView,
                                                                                                       IndexType,
                                                                                                       Fetch,
                                                                                                       Reduction,
                                                                                                       ResultKeeper,
                                                                                                       Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexes< 256,
                                                                                                       128,
                                                                                                       ConstViewType,
                                                                                                       ArrayView,
                                                                                                       IndexType,
                                                                                                       Fetch,
                                                                                                       Reduction,
                                                                                                       ResultKeeper,
                                                                                                       Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }

                     default:
                        throw std::runtime_error( "Unsupported number of threads per segment"
                                                  + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                  + ". It can be only 2, 4, 8, 16 or 32." );
                        break;
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernelWithIndexes< ConstViewType,
                                                                                             ArrayView,
                                                                                             IndexType,
                                                                                             Fetch,
                                                                                             Reduction,
                                                                                             ResultKeeper,
                                                                                             Value,
                                                                                             256 >;
                  Backend::launchKernelAsync( kernel,
                                              launch_config,
                                              gridIdx,
                                              segments.getConstView(),
                                              segmentIndexes.getConstView(),
                                              begin,
                                              end,
                                              fetch,
                                              reduction,
                                              keeper,
                                              identity );
               }
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         reduceSegmentsWithIndexesSequential(
            segments, segmentIndexes, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsSequentialWithArgument( const ConstViewType& segments,
                                         IndexBegin begin,
                                         IndexEnd end,
                                         Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                                         Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                                         ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                                         const Value& identity,
                                         const LaunchConfiguration& launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();

      auto l = [ offsets, fetch, reduction, keeper, identity ] __cuda_callable__( const Index segmentIdx ) mutable
      {
         const IndexType begin = offsets[ segmentIdx ];
         const IndexType end = offsets[ segmentIdx + 1 ];
         using ReturnType = typename detail::FetchLambdaAdapter< IndexType, Fetch >::ReturnType;
         ReturnType result = identity;
         IndexType argument = 0;
         IndexType localIdx = 0;
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
            if constexpr( argumentCount< Fetch >() == 3 )
               reduction( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
            else
               reduction( result, fetch( globalIdx ), argument, localIdx );
            localIdx++;
         }
         keeper( segmentIdx, argument, result );
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
      else
         Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value = typename detail::FetchLambdaAdapter< Index, Fetch >::ReturnType >
   static void
   reduceSegmentsWithArgument( const ConstViewType& segments,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch fetch,          // TODO Fetch&& fetch does not work here with CUDA
                               Reduction reduction,  // TODO Reduction&& reduction does not work here with CUDA
                               ResultKeeper keeper,  // TODO ResultKeeper&& keeper does not work here with CUDA
                               const Value& identity,
                               const LaunchConfiguration& launchConfig )
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
            reduceSegmentsSequentialWithArgument( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp )
               threadsCount *= (std::size_t) Backend::getWarpSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp ) {
                  constexpr auto kernel = reduceSegmentsCSRVectorKernelWithArgument< ConstViewType,
                                                                                     IndexType,
                                                                                     Fetch,
                                                                                     Reduction,
                                                                                     ResultKeeper,
                                                                                     Value >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getConstView(), begin, end, fetch, reduction, keeper, identity );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument< 2,
                                                                                                      ConstViewType,
                                                                                                      IndexType,
                                                                                                      Fetch,
                                                                                                      Reduction,
                                                                                                      ResultKeeper,
                                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument< 4,
                                                                                                      ConstViewType,
                                                                                                      IndexType,
                                                                                                      Fetch,
                                                                                                      Reduction,
                                                                                                      ResultKeeper,
                                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument< 8,
                                                                                                      ConstViewType,
                                                                                                      IndexType,
                                                                                                      Fetch,
                                                                                                      Reduction,
                                                                                                      ResultKeeper,
                                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument< 16,
                                                                                                      ConstViewType,
                                                                                                      IndexType,
                                                                                                      Fetch,
                                                                                                      Reduction,
                                                                                                      ResultKeeper,
                                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithArgument< 32,
                                                                                                      ConstViewType,
                                                                                                      IndexType,
                                                                                                      Fetch,
                                                                                                      Reduction,
                                                                                                      ResultKeeper,
                                                                                                      Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithArgument< 256,
                                                                                                        64,
                                                                                                        ConstViewType,
                                                                                                        IndexType,
                                                                                                        Fetch,
                                                                                                        Reduction,
                                                                                                        ResultKeeper,
                                                                                                        Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithArgument< 256,
                                                                                                        128,
                                                                                                        ConstViewType,
                                                                                                        IndexType,
                                                                                                        Fetch,
                                                                                                        Reduction,
                                                                                                        ResultKeeper,
                                                                                                        Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }

                     default:
                        throw std::runtime_error( "Unsupported number of threads per segment"
                                                  + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                  + ". It can be only 2, 4, 8, 16 or 32." );
                        break;
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernelWithArgument< ConstViewType,
                                                                                              IndexType,
                                                                                              Fetch,
                                                                                              Reduction,
                                                                                              ResultKeeper,
                                                                                              Value,
                                                                                              256 >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getConstView(), begin, end, fetch, reduction, keeper, identity );
               }
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
               }
               Backend::streamSynchronize( launch_config.stream );
            }
         }
      }
      else
         reduceSegmentsSequentialWithArgument( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value >
   static void
   reduceSegmentsWithIndexesAndArgumentSequential( const ConstViewType& segments,
                                                   const Array& segmentIndexes,
                                                   IndexBegin begin,
                                                   IndexEnd end,
                                                   Fetch&& fetch,
                                                   Reduction&& reduction,
                                                   ResultKeeper&& keeper,
                                                   const Value& identity,
                                                   LaunchConfiguration launchConfig )
   {
      using OffsetsView = typename SegmentsViewType::ConstOffsetsView;
      OffsetsView offsets = segments.getOffsets();
      auto segmentIndexes_view = segmentIndexes.getConstView();

      auto l = [ offsets, segmentIndexes_view, fetch, reduction, keeper, identity ] __cuda_callable__(
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
               if constexpr( argumentCount< Fetch >() == 3 )
                  reduction( result, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
               else
                  reduction( result, fetch( globalIdx ), argument, localIdx );
         }
         keeper( segmentIdx_idx, segmentIdx, argument, result );
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
      else
         Algorithms::parallelFor< Device >( begin, end, l );
   }

   template< typename Array,
             typename IndexBegin,
             typename IndexEnd,
             typename Fetch,
             typename Reduction,
             typename ResultKeeper,
             typename Value >
   static void
   reduceSegmentsWithSegmentIndexesAndArgument(
      const ConstViewType& segments,
      const Array& segmentIndexes,
      IndexBegin begin,
      IndexEnd end,
      Fetch fetch,          // TODO: Fetch&& fetch does not work here with CUDA
      Reduction reduction,  // TODO: Reduction&& reduction does not work here with CUDA
      ResultKeeper keeper,  // TODO: ResultKeeper&& keeper does not work here with CUDA
      const Value& identity,
      LaunchConfiguration launchConfig )
   {
      using ArrayView = typename Array::ConstViewType;
      if constexpr( std::is_same_v< Device, TNL::Devices::Cuda > || std::is_same_v< Device, TNL::Devices::Hip > ) {
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed
             && launchConfig.getThreadsPerSegmentCount() == 1 )
            reduceSegmentsWithIndexesAndArgumentSequential(
               segments, segmentIndexes, begin, end, fetch, reduction, keeper, identity, launchConfig );
         else {
            std::size_t threadsCount = end - begin;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp )
               threadsCount *= (std::size_t) Backend::getWarpSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed )
               threadsCount *= (std::size_t) launchConfig.getThreadsPerSegmentCount();
            if( threadsCount > std::numeric_limits< IndexType >::max() )
               throw std::runtime_error( "The number of GPU threads exceeds the maximum limit of the IndexType." );

            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Warp ) {
                  constexpr auto kernel = reduceSegmentsCSRVectorKernelWithIndexesAndArgument< ConstViewType,
                                                                                               ArrayView,
                                                                                               IndexType,
                                                                                               Fetch,
                                                                                               Reduction,
                                                                                               ResultKeeper,
                                                                                               Value >;
                  Backend::launchKernelAsync( kernel,
                                              launch_config,
                                              gridIdx,
                                              segments.getConstView(),
                                              segmentIndexes.getConstView(),
                                              begin,
                                              end,
                                              fetch,
                                              reduction,
                                              keeper,
                                              identity );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::Fixed ) {
                  switch( launchConfig.getThreadsPerSegmentCount() ) {
                     case 2:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument< 2,
                                                                                                                ConstViewType,
                                                                                                                ArrayView,
                                                                                                                IndexType,
                                                                                                                Fetch,
                                                                                                                Reduction,
                                                                                                                ResultKeeper,
                                                                                                                Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 4:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument< 4,
                                                                                                                ConstViewType,
                                                                                                                ArrayView,
                                                                                                                IndexType,
                                                                                                                Fetch,
                                                                                                                Reduction,
                                                                                                                ResultKeeper,
                                                                                                                Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 8:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument< 8,
                                                                                                                ConstViewType,
                                                                                                                ArrayView,
                                                                                                                IndexType,
                                                                                                                Fetch,
                                                                                                                Reduction,
                                                                                                                ResultKeeper,
                                                                                                                Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 16:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument< 16,
                                                                                                                ConstViewType,
                                                                                                                ArrayView,
                                                                                                                IndexType,
                                                                                                                Fetch,
                                                                                                                Reduction,
                                                                                                                ResultKeeper,
                                                                                                                Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 32:
                        {
                           constexpr auto kernel = reduceSegmentsCSRVariableVectorKernelWithIndexesAndArgument< 32,
                                                                                                                ConstViewType,
                                                                                                                ArrayView,
                                                                                                                IndexType,
                                                                                                                Fetch,
                                                                                                                Reduction,
                                                                                                                ResultKeeper,
                                                                                                                Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 64:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexesAndArgument< 256,
                                                                                                                  64,
                                                                                                                  ConstViewType,
                                                                                                                  ArrayView,
                                                                                                                  IndexType,
                                                                                                                  Fetch,
                                                                                                                  Reduction,
                                                                                                                  ResultKeeper,
                                                                                                                  Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }
                     case 128:
                        {
                           constexpr auto kernel = reduceSegmentsCSRLightMultivectorKernelWithIndexesAndArgument< 256,
                                                                                                                  128,
                                                                                                                  ConstViewType,
                                                                                                                  ArrayView,
                                                                                                                  IndexType,
                                                                                                                  Fetch,
                                                                                                                  Reduction,
                                                                                                                  ResultKeeper,
                                                                                                                  Value >;
                           Backend::launchKernelAsync( kernel,
                                                       launch_config,
                                                       gridIdx,
                                                       segments.getConstView(),
                                                       segmentIndexes.getConstView(),
                                                       begin,
                                                       end,
                                                       fetch,
                                                       reduction,
                                                       keeper,
                                                       identity );
                           break;
                        }

                     default:
                        throw std::runtime_error( "Unsupported number of threads per segment"
                                                  + std::to_string( launchConfig.getThreadsPerSegmentCount() )
                                                  + ". It can be only 2, 4, 8, 16 or 32." );
                        break;
                  }
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::DynamicGrouping ) {
                  constexpr auto kernel = reduceSegmentsCSRDynamicGroupingKernelWithIndexesAndArgument< ConstViewType,
                                                                                                        ArrayView,
                                                                                                        IndexType,
                                                                                                        Fetch,
                                                                                                        Reduction,
                                                                                                        ResultKeeper,
                                                                                                        Value,
                                                                                                        256 >;
                  Backend::launchKernelAsync( kernel,
                                              launch_config,
                                              gridIdx,
                                              segments.getConstView(),
                                              segmentIndexes.getConstView(),
                                              begin,
                                              end,
                                              fetch,
                                              reduction,
                                              keeper,
                                              identity );
               }
               else {
                  throw std::runtime_error( "Unsupported threads to segments mapping strategy." );
               }
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         reduceSegmentsWithIndexesAndArgumentSequential(
            segments, segmentIndexes, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }
};
}  //namespace TNL::Algorithms::Segments::detail
