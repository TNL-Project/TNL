// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRScalarKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRVectorKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRLightKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/CSRAdaptiveKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/EllpackKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/SlicedEllpackKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/BiEllpackKernel.h>
#include <TNL/Algorithms/SegmentsReductionKernels/ChunkedEllpackKernel.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixOperations.h>
#include "LaunchConfigurationsSetup.h"

namespace TNL::Benchmarks::Segments {

template< typename Index = int >
struct SegmentsBenchmark
{
   using IndexType = Index;
   using HostVector = TNL::Containers::Vector< Index, TNL::Devices::Host, Index >;
   //using HostRealVector = TNL::Containers::Vector< Real, TNL::Devices::Host, Index >;

   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using CSRSegments = TNL::Algorithms::Segments::CSR< Device_, Index_, IndexAllocator_ >;
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using EllpackSegments = TNL::Algorithms::Segments::Ellpack< Device_, Index_, IndexAllocator_ >;
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using SlicedEllpackSegments = TNL::Algorithms::Segments::SlicedEllpack< Device_, Index_, IndexAllocator_ >;
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using BiEllpackSegments = TNL::Algorithms::Segments::BiEllpack< Device_, Index_, IndexAllocator_ >;
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using ChunkedEllpackSegments = TNL::Algorithms::Segments::ChunkedEllpack< Device_, Index_, IndexAllocator_ >;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Benchmark settings:" );
      config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-segments.log" );
      config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
      config.addEntryEnum( "append" );
      config.addEntryEnum( "overwrite" );
      config.addEntry< TNL::String >( "segments-setup", "Segments setup used for benchmarking.", "all" );
      config.addEntryEnum( "all" );
      config.addEntryEnum( "constant" );
      config.addEntryEnum( "linear" );
      config.addEntryEnum( "quadratic" );
      config.addEntry< int >( "min-segment-size", "Minimum segment size.", 1 );
      config.addEntry< int >( "max-segment-size", "Maximum segment size.", 128 );
      config.addEntry< int >( "min-segments-count", "Minimum number of segments.", 1 << 8 );
      config.addEntry< int >( "max-segments-count", "Maximum number of segments.", 1 << 20 );
      //config.addEntry< bool >( "with-bfs", "Run breadth-first search benchmark.", true );

      config.addDelimiter( "Device settings:" );
      config.addEntry< TNL::String >( "device", "Device the computation will run on.", "all" );
      config.addEntryEnum< TNL::String >( "all" );
      config.addEntryEnum< TNL::String >( "host" );
      config.addEntryEnum< TNL::String >( "sequential" );
      config.addEntryEnum< TNL::String >( "cuda" );
      TNL::Devices::Host::configSetup( config );
      TNL::Devices::Cuda::configSetup( config );

      config.addEntry< int >( "loops", "Number of iterations for every computation.", 10 );
      config.addEntry< int >( "verbose", "Verbose mode.", 1 );
   }

   SegmentsBenchmark( const TNL::Config::ParameterContainer& parameters_ ) : parameters( parameters_ ) {}

   template< typename Device,
             template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments,
             template< typename Index_, typename Device_ > class SegmentsKernel >
   void
   TNLBenchmarks( const HostVector& hostSegmentsSizes,
                  TNL::Benchmarks::Benchmark<>& benchmark,
                  const TNL::String& device,
                  const TNL::String& segmentsType )
   {
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;
      using IndexAllocator = typename TNL::Allocators::Default< Device >::template Allocator< Index >;
      using SegmentsType = Segments< Device, Index, IndexAllocator >;

      //using KernelType = SegmentsKernel< Index, Device >;
      // TODO: Find a way how to use various reduction kernels for segments in the algorithms.

      IndexVector segmentsSizes( hostSegmentsSizes );
      auto segmentsSizes_view = segmentsSizes.getConstView();
      SegmentsType segments( segmentsSizes );
      IndexVector data( segments.getStorageSize(), 0 );
      auto dataView = data.getView();
      benchmark.setMetadataElement( { "segments type", segmentsType } );

      benchmark.setMetadataElement( { "function", "forElements" } );
      benchmark.setDatasetSize( sum( segmentsSizes ) * sizeof( Index ) );
      for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
         benchmark.setMetadataElement( { "threads mapping", tag } );
         auto segmentsView = segments.getView();
         auto f = [ & ]() mutable
         {
            TNL::Algorithms::Segments::forAllElements(
               segmentsView,
               [ = ] __cuda_callable__(
                  const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
               {
                  dataView[ globalIdx ] = segmentIdx + localIdx;
               },
               launchConfig );
         };
         benchmark.time< Device >( device, f );
         HostVector dataHost( data );
         for( IndexType segmentIdx = 0; segmentIdx < segmentsSizes.getSize(); segmentIdx++ ) {
            for( IndexType localIdx = 0; localIdx < segmentsSizes.getElement( segmentIdx ); localIdx++ )
               if( dataHost.getElement( segments.getGlobalIndex( segmentIdx, localIdx ) ) != segmentIdx + localIdx )
                  throw std::runtime_error( "Error in forElements" );
         }
      }

      for( auto stride : { 2, 4, 8 } ) {
         benchmark.setMetadataElement( { "function", "forElements with indexes stride " + convertToString( stride ) } );
         IndexVector segmentIndexes( segmentsSizes.getSize() / stride );
         auto segmentIndexes_view = segmentIndexes.getView();
         segmentIndexes.forAllElements(
            [ = ] __cuda_callable__( IndexType idx, IndexType & value )
            {
               value = stride * idx;
            } );
         benchmark.setDatasetSize( TNL::Algorithms::reduce< Device >(
                                      0,
                                      segmentIndexes.getSize(),
                                      [ = ] __cuda_callable__( Index idx )
                                      {
                                         return segmentsSizes_view[ segmentIndexes_view[ idx ] ];
                                      },
                                      TNL::Plus{} )
                                   * sizeof( Index ) );
         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );
            auto segmentsView = segments.getView();
            auto f = [ & ]() mutable
            {
               TNL::Algorithms::Segments::forElements(
                  segmentsView,
                  segmentIndexes_view,
                  [ = ] __cuda_callable__(
                     const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
                  {
                     dataView[ globalIdx ] = 1;
                  },
                  launchConfig );
            };
            benchmark.time< Device >( device, f );
         }
      }

      for( auto stride : { 2, 4, 8 } ) {
         benchmark.setMetadataElement( { "function", "forElementsIf stride " + convertToString( stride ) } );
         benchmark.setDatasetSize( TNL::Algorithms::reduce< Device >(
                                      0,
                                      segmentsSizes.getSize(),
                                      [ = ] __cuda_callable__( Index idx )
                                      {
                                         return ( idx % stride == 0 ) ? segmentsSizes_view[ idx ] : 0;
                                      },
                                      TNL::Plus{} )
                                   * sizeof( Index ) );

         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );
            auto segmentsView = segments.getView();
            auto f = [ & ]() mutable
            {
               TNL::Algorithms::Segments::forAllElementsIf(
                  segmentsView,
                  [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
                  {
                     return segmentIdx % stride == 0;
                  },
                  [ = ] __cuda_callable__(
                     const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
                  {
                     dataView[ globalIdx ] = 1;
                  },
                  launchConfig );
            };
            benchmark.time< Device >( device, f );
         }

         benchmark.setMetadataElement( { "function", "forElementsIfSparse with stride " + convertToString( stride ) } );
         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );
            auto segmentsView = segments.getView();
            auto f = [ & ]() mutable
            {
               TNL::Algorithms::Segments::forAllElementsIfSparse(
                  segmentsView,
                  [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
                  {
                     return segmentIdx % stride == 0;
                  },
                  [ = ] __cuda_callable__(
                     const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
                  {
                     dataView[ globalIdx ] = 1;
                  },
                  launchConfig );
            };
            benchmark.time< Device >( device, f );
         }
      }

      IndexVector result( segmentsSizes.getSize(), 0 );
      auto resultView = result.getView();
      TNL::Algorithms::Segments::forAllElements(
         segments,
         [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType localIdx, const IndexType globalIdx ) mutable
         {
            if( localIdx < segmentsSizes_view[ segmentIdx ] ) {
               dataView[ globalIdx ] = 1;
            }
            else
               dataView[ globalIdx ] = 0;
         } );
      benchmark.setMetadataElement( { "function", "reduceSegments" } );
      benchmark.setDatasetSize( sum( segmentsSizes ) * sizeof( Index ) );
      for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
         benchmark.setMetadataElement( { "threads mapping", tag } );
         auto segmentsView = segments.getView();
         auto f = [ & ]() mutable
         {
            TNL::Algorithms::Segments::reduceAllSegments(
               segmentsView,
               [ = ] __cuda_callable__( const IndexType globalIdx ) mutable
               {
                  return dataView[ globalIdx ];
               },
               TNL::Plus{},
               [ = ] __cuda_callable__( const IndexType segmentIdx, const IndexType result ) mutable
               {
                  resultView[ segmentIdx ] = result;
               },
               launchConfig );
         };
         benchmark.time< Device >( device, f );
         HostVector resultHost( result );
         for( IndexType segmentIdx = 0; segmentIdx < segmentsSizes.getSize(); segmentIdx++ ) {
            if( resultHost[ segmentIdx ] != segmentsSizes.getElement( segmentIdx ) )
               throw std::runtime_error( "Error in reduceSegments" );
         }
      }

      for( auto stride : { 2, 4, 8 } ) {
         result = 0;
         benchmark.setMetadataElement( { "function", "reduceSegmentsWithIndexes stride " + convertToString( stride ) } );
         IndexVector segmentIndexes( segmentsSizes.getSize() / stride );
         auto segmentIndexes_view = segmentIndexes.getConstView();
         segmentIndexes.forAllElements(
            [ = ] __cuda_callable__( IndexType idx, IndexType & value )
            {
               value = stride * idx;
            } );
         benchmark.setDatasetSize( TNL::Algorithms::reduce< Device >(
                                      0,
                                      segmentIndexes.getSize(),
                                      [ = ] __cuda_callable__( Index idx )
                                      {
                                         return segmentsSizes_view[ segmentIndexes_view[ idx ] ];
                                      },
                                      TNL::Plus{} )
                                   * sizeof( Index ) );
         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );
            auto segmentsView = segments.getView();
            auto f = [ & ]() mutable
            {
               TNL::Algorithms::Segments::reduceSegments(
                  segmentsView,
                  segmentIndexes,
                  [ = ] __cuda_callable__( const IndexType globalIdx ) mutable
                  {
                     return dataView[ globalIdx ];
                  },
                  TNL::Plus{},
                  [ = ] __cuda_callable__(
                     const IndexType segmentIdx_idx, const IndexType segmentIdx, const IndexType result ) mutable
                  {
                     resultView[ segmentIdx ] = result;
                  },
                  launchConfig );
            };
            benchmark.time< Device >( device, f );
            HostVector resultHost( result );
            for( IndexType segmentIdx = 0; segmentIdx < segmentsSizes.getSize(); segmentIdx++ ) {
               if( segmentIdx % stride == 0 ) {
                  if( resultHost[ segmentIdx ] != segmentsSizes.getElement( segmentIdx ) )
                     throw std::runtime_error( "Error in reduceSegments" );
               }
               else {
                  if( resultHost[ segmentIdx ] != 0 )
                     throw std::runtime_error( "Error in reduceSegments" );
               }
            }
         }
      }

      for( auto stride : { 2, 4, 8 } ) {
         result = 0;
         benchmark.setMetadataElement( { "function", "reduceSegmentIf stride " + convertToString( stride ) } );
         benchmark.setDatasetSize( TNL::Algorithms::reduce< Device >(
                                      0,
                                      segmentsSizes.getSize(),
                                      [ = ] __cuda_callable__( Index idx )
                                      {
                                         return ( idx % stride == 0 ) ? segmentsSizes_view[ idx ] : 0;
                                      },
                                      TNL::Plus{} )
                                   * sizeof( Index ) );
         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );
            auto segmentsView = segments.getView();
            auto f = [ & ]() mutable
            {
               TNL::Algorithms::Segments::reduceAllSegmentsIf(
                  segmentsView,
                  [ = ] __cuda_callable__( const IndexType segmentIdx ) -> bool
                  {
                     return segmentIdx % stride == 0;
                  },
                  [ = ] __cuda_callable__( const IndexType globalIdx ) mutable
                  {
                     return dataView[ globalIdx ];
                  },
                  TNL::Plus{},
                  [ = ] __cuda_callable__(
                     const IndexType segmentIdx_idx, const IndexType segmentIdx, const IndexType result ) mutable
                  {
                     resultView[ segmentIdx ] = result;
                  },
                  launchConfig );
            };
            benchmark.time< Device >( device, f );
            HostVector resultHost( result );
            for( IndexType segmentIdx = 0; segmentIdx < segmentsSizes.getSize(); segmentIdx++ ) {
               if( segmentIdx % stride == 0 ) {
                  if( resultHost[ segmentIdx ] != segmentsSizes.getElement( segmentIdx ) )
                     throw std::runtime_error( "Error in reduceSegments" );
               }
               else {
                  if( resultHost[ segmentIdx ] != 0 )
                     throw std::runtime_error( "Error in reduceSegments" );
               }
            }
         }
      }
   }

   void
   runBenchmark( TNL::Benchmarks::Benchmark<>& benchmark, const HostVector& segmentsSizes, std::string segmentsSetup )
   {
      auto device = parameters.getParameter< TNL::String >( "device" );

      benchmark.setMetadataColumns( {
         { "segments setup", segmentsSetup },
         { "segments count", convertToString( segmentsSizes.getSize() ) },
         { "max segment size", convertToString( max( segmentsSizes ) ) },
         { "elements count", convertToString( sum( segmentsSizes ) ) },
      } );
      benchmark.setMetadataWidths( {
         { "segments setup", 16 },
         { "segments count", 16 },
         { "max segment size", 18 },
         { "elements count", 16 },
         { "segments type", 25 },
         { "function", 35 },
         { "threads mapping", 44 },
      } );

      if( device == "sequential" || device == "all" )
         TNLBenchmarks< TNL::Devices::Sequential, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            segmentsSizes, benchmark, "sequential", "CSR" );
      if( device == "host" || device == "all" )
         TNLBenchmarks< TNL::Devices::Host, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            segmentsSizes, benchmark, "host", "CSR" );
#ifdef __CUDACC__
      if( device == "cuda" || device == "all" ) {
         TNLBenchmarks< TNL::Devices::Cuda, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            segmentsSizes, benchmark, "cuda", "CSR" );
         TNLBenchmarks< TNL::Devices::Cuda, EllpackSegments, TNL::Algorithms::SegmentsReductionKernels::EllpackKernel >(
            segmentsSizes, benchmark, "cuda", "Ellpack" );
         TNLBenchmarks< TNL::Devices::Cuda,
                        SlicedEllpackSegments,
                        TNL::Algorithms::SegmentsReductionKernels::SlicedEllpackKernel >(
            segmentsSizes, benchmark, "cuda", "SlicedEllpack" );
         TNLBenchmarks< TNL::Devices::Cuda, BiEllpackSegments, TNL::Algorithms::SegmentsReductionKernels::BiEllpackKernel >(
            segmentsSizes, benchmark, "cuda", "BiEllpack" );
         TNLBenchmarks< TNL::Devices::Cuda,
                        ChunkedEllpackSegments,
                        TNL::Algorithms::SegmentsReductionKernels::ChunkedEllpackKernel >(
            segmentsSizes, benchmark, "cuda", "ChunkedEllpack" );
      }
#endif
   }

   void
   setupBenchmark()
   {
      const auto logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const auto outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const int loops = parameters.getParameter< int >( "loops" );
      const int verbose = parameters.getParameter< int >( "verbose" );
      const auto segmentsSetup = parameters.getParameter< TNL::String >( "segments-setup" );
      const int minSegmentsCount = parameters.getParameter< int >( "min-segments-count" );
      const int maxSegmentsCount = parameters.getParameter< int >( "max-segments-count" );
      const int minSegmentSize = parameters.getParameter< int >( "min-segment-size" );
      const int maxSegmentSize = parameters.getParameter< int >( "max-segment-size" );

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      // Constant segments
      if( segmentsSetup == "constant" || segmentsSetup == "all" ) {
         for( Index segmentsCount = minSegmentsCount; segmentsCount < maxSegmentsCount; segmentsCount *= 2 ) {
            for( Index segmentSize = minSegmentSize; segmentSize < maxSegmentSize; segmentSize *= 2 ) {
               runBenchmark( benchmark, HostVector( segmentsCount, segmentSize ), "constant" );
            }
         }
      }

      // Linear segments
      if( segmentsSetup == "linear" || segmentsSetup == "all" ) {
         for( Index segmentsCount = minSegmentsCount; segmentsCount < maxSegmentsCount; segmentsCount *= 2 ) {
            for( Index segmentSize = minSegmentSize; segmentSize < maxSegmentSize; segmentSize *= 2 ) {
               HostVector segmentSizes( segmentsCount );
               segmentSizes.forAllElements(
                  [ = ] __cuda_callable__( Index i, Index & x ) mutable
                  {
                     x = i % segmentSize + 1;
                  } );
               runBenchmark( benchmark, segmentSizes, "linear" );
            }
         }
      }

      // Quadratic segments
      if( segmentsSetup == "quadratic" || segmentsSetup == "all" ) {
         for( Index segmentsCount = minSegmentsCount; segmentsCount < maxSegmentsCount; segmentsCount *= 2 ) {
            for( Index segmentSize = minSegmentSize; segmentSize < maxSegmentSize; segmentSize *= 2 ) {
               HostVector segmentSizes( segmentsCount );
               segmentSizes.forAllElements(
                  [ = ] __cuda_callable__( Index i, Index & x ) mutable
                  {
                     std::size_t val = i * i;
                     x = val % segmentSize + 1;
                  } );
               runBenchmark( benchmark, segmentSizes, "quadratic" );
            }
         }
      }
   }

protected:
   const TNL::Config::ParameterContainer& parameters;
};

}  //namespace TNL::Benchmarks::Segments
