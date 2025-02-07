// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>
#include "FetchLambdaAdapter.h"
#include "ReducingKernels_CSR.h"

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
struct ReducingOperations< CSRView< Device, Index > >
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
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::ThreadPerSegment
             || ( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::UserDefined
                  && launchConfig.getThreadsPerSegmentCount() == 1 ) )
            reduceSegmentsSequential( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
         else {
            std::size_t threadsCount = 0;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment )
               threadsCount = ( end - begin ) * Backend::getWarpSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::UserDefined )
               threadsCount = ( end - begin ) * launchConfig.getThreadsPerSegmentCount();
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
                  constexpr auto kernel =
                     reduceSegmentsCSRVectorKernel< ConstViewType, IndexType, Fetch, Reduction, ResultKeeper, Value >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getConstView(), begin, end, fetch, reduction, keeper, identity );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::UserDefined ) {
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
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         reduceSegmentsSequential( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
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
         ReturnType aux = identity;
         IndexType argument = 0;
         IndexType localIdx = 0;
         for( IndexType globalIdx = begin; globalIdx < end; globalIdx++ ) {
            if constexpr( argumentCount< Fetch >() == 3 )
               reduction( aux, fetch( segmentIdx, localIdx, globalIdx ), argument, localIdx );
            else
               reduction( aux, fetch( globalIdx ), argument, localIdx );
            localIdx++;
         }
         keeper( segmentIdx, aux, argument );
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
         if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::ThreadPerSegment
             || ( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::UserDefined
                  && launchConfig.getThreadsPerSegmentCount() == 1 ) )
            reduceSegmentsSequentialWithArgument( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
         else {
            std::size_t threadsCount = 0;
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment )
               threadsCount = ( end - begin ) * Backend::getWarpSize();
            if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::UserDefined )
               threadsCount = ( end - begin ) * launchConfig.getThreadsPerSegmentCount();
            Backend::LaunchConfiguration launch_config;
            launch_config.blockSize.x = 256;
            dim3 blocksCount;
            dim3 gridsCount;
            Backend::setupThreads( launch_config.blockSize, blocksCount, gridsCount, threadsCount );
            for( IndexType gridIdx = 0; gridIdx < (Index) gridsCount.x; gridIdx++ ) {
               Backend::setupGrid( blocksCount, gridsCount, gridIdx, launch_config.gridSize );
               if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::WarpPerSegment ) {
                  constexpr auto kernel = reduceSegmentsCSRVectorKernelWithArgument< ConstViewType,
                                                                                     IndexType,
                                                                                     Fetch,
                                                                                     Reduction,
                                                                                     ResultKeeper,
                                                                                     Value >;
                  Backend::launchKernelAsync(
                     kernel, launch_config, gridIdx, segments.getConstView(), begin, end, fetch, reduction, keeper, identity );
               }
               else if( launchConfig.getThreadsToSegmentsMapping() == ThreadsToSegmentsMapping::UserDefined ) {
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
            }
            Backend::streamSynchronize( launch_config.stream );
         }
      }
      else
         reduceSegmentsSequentialWithArgument( segments, begin, end, fetch, reduction, keeper, identity, launchConfig );
   }
};

}  //namespace TNL::Algorithms::Segments::detail
