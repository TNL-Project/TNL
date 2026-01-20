// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Graphs/GraphOperations.h>
#include <TNL/Graphs/Readers/EdgeListReader.h>
#include <TNL/Graphs/Readers/MtxReader.h>
#include <TNL/Graphs/Writers/EdgeListWriter.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Algorithms/Segments/TraversingLaunchConfigurations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <type_traits>
#include <utility>
#include <vector>

//#define WITH_ROW_MAJOR_SLICED_ELLPACK
//#define WITH_SORTED_SEGMENTS

namespace TNL::Benchmarks::Graphs {

template< typename Real, typename Index, typename Derived >  // CRTP – Curiously Recurring Template Pattern
class GraphBenchmarkBase
{
public:
   using RealType = Real;
   using IndexType = Index;
   using HostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, Index >;
   using HostGraph = TNL::Graphs::Graph< Real, TNL::Devices::Host, Index, TNL::Graphs::UndirectedGraph >;
   using HostDigraph = TNL::Graphs::Graph< Real, TNL::Devices::Host, Index, TNL::Graphs::DirectedGraph >;
   using HostIndexVector = TNL::Containers::Vector< Index, TNL::Devices::Host, Index >;
   using HostRealVector = TNL::Containers::Vector< Real, TNL::Devices::Host, Index >;

   // Template aliases for segments types
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using CSRSegments = TNL::Algorithms::Segments::CSR< Device_, Index_, IndexAllocator_ >;

   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using EllpackSegments = TNL::Algorithms::Segments::Ellpack< Device_, Index_, IndexAllocator_ >;

   template< int SliceSize >
   struct RowMajorSlicedEllpackSegments
   {
      template< typename Device_, typename Index_, typename IndexAllocator_ >
      using type = TNL::Algorithms::Segments::RowMajorSlicedEllpack< Device_, Index_, IndexAllocator_, SliceSize >;
   };

   template< int SliceSize >
   struct ColumnMajorSlicedEllpackSegments
   {
      template< typename Device_, typename Index_, typename IndexAllocator_ >
      using type = TNL::Algorithms::Segments::RowMajorSlicedEllpack< Device_, Index_, IndexAllocator_, SliceSize >;
   };

   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using BiEllpackSegments = TNL::Algorithms::Segments::BiEllpack< Device_, Index_, IndexAllocator_ >;

   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using ChunkedEllpackSegments = TNL::Algorithms::Segments::ChunkedEllpack< Device_, Index_, IndexAllocator_ >;

   // Template aliases for sorted segments types
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using SortedCSRSegments = TNL::Algorithms::Segments::SortedCSR< Device_, Index_, IndexAllocator_ >;

   template< int SliceSize >
   struct SortedRowMajorSlicedEllpackSegments
   {
      template< typename Device_, typename Index_, typename IndexAllocator_ >
      using type = TNL::Algorithms::Segments::SortedRowMajorSlicedEllpack< Device_, Index_, IndexAllocator_, SliceSize >;
   };

   template< int SliceSize >
   struct SortedColumnMajorSlicedEllpackSegments
   {
      template< typename Device_, typename Index_, typename IndexAllocator_ >
      using type = TNL::Algorithms::Segments::SortedColumnMajorSlicedEllpack< Device_, Index_, IndexAllocator_, SliceSize >;
   };

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Benchmark settings:" );
      config.addEntry< TNL::String >( "input-file", "Input file with the graph." );
      config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-graphs.log" );
      config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
      config.addEntryEnum( "append" );
      config.addEntryEnum( "overwrite" );
#ifdef WITH_SORTED_SEGMENTS
      config.addEntry< bool >( "with-sorted-segments", "Run benchmark with sorted segments.", true );
#endif
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

   GraphBenchmarkBase( const TNL::Config::ParameterContainer& parameters_ )
   : parameters( parameters_ )
   {}

   virtual ~GraphBenchmarkBase() = default;

   bool
   runBenchmark()
   {
      auto inputFile = parameters.getParameter< TNL::String >( "input-file" );
      const auto logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const auto outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const int loops = parameters.getParameter< int >( "loops" );
      verbose = parameters.getParameter< int >( "verbose" );
#ifdef WITH_SORTED_SEGMENTS
      withSortedSegments = parameters.getParameter< bool >( "with-sorted-segments" );
#endif

      size_t dotPosition = inputFile.find_last_of( '.' );
      std::string inputFileExtension;
      if( dotPosition != std::string::npos )
         inputFileExtension = inputFile.substr( dotPosition + 1 );

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      this->errors = 0;

      auto device = parameters.getParameter< TNL::String >( "device" );

      std::cout << "Graphs benchmark with " << TNL::getType< Real >() << " precision and device: " << device << std::endl;

      HostDigraph digraph;
      std::cout << "Reading graph from file " << inputFile << '\n';
      if( inputFileExtension == "mtx" )
         TNL::Graphs::Readers::MtxReader< HostDigraph >::read( inputFile, digraph );
      else
         TNL::Graphs::Readers::EdgeListReader< HostDigraph >::read( inputFile, digraph );
      // Make all weights positive because of benchmarking SSSP
      digraph.getAdjacencyMatrix().getValues() = abs( digraph.getAdjacencyMatrix().getValues() );

      auto symmetrizedAdjacencyMatrix = TNL::Matrices::getSymmetricPart< HostMatrix >( digraph.getAdjacencyMatrix() );
      HostGraph graph( symmetrizedAdjacencyMatrix );

      HostIndexVector nodeDegrees( digraph.getVertexCount(), 0 );
      graph.getAdjacencyMatrix().getCompressedRowLengths( nodeDegrees );
      Index largest = TNL::argMax( nodeDegrees ).second;
      Index smallest = TNL::argMax( greater( nodeDegrees, 0 ) ).second;
      std::cout << "Smallest degree is " << nodeDegrees[ smallest ] << " at position " << smallest << std::endl;
      std::cout << "Largest degree is " << nodeDegrees[ largest ] << " at position " << largest << std::endl;

      benchmark.setMetadataColumns( {
         { "graph name", inputFile },
         { "precision", getType< Real >() },
         { "index type", TNL::getType< Index >() },
         { "nodes", convertToString( graph.getAdjacencyMatrix().getRows() ) },
         { "edges", convertToString( graph.getAdjacencyMatrix().getNonzeroElementsCount() ) },
      } );
      benchmark.setMetadataWidths( {
         { "graph name", 32 },
         { "format", 26 },
         { "threads", 5 },
      } );

      auto& derived = static_cast< Derived& >( *this );

      // Run benchmarks for different libraries and devices
      derived.runOtherBenchmarks( digraph, graph, smallest, largest, benchmark );

      if( device == "sequential" || device == "all" )
         runTNLBenchmarks< TNL::Devices::Sequential, CSRSegments >(
            digraph, graph, smallest, largest, benchmark, "sequential" );
      if( device == "host" || device == "all" )
         runTNLBenchmarks< TNL::Devices::Host, CSRSegments >( digraph, graph, smallest, largest, benchmark, "host" );

#ifdef __CUDACC__
      if( device == "cuda" || device == "all" ) {
         runTNLBenchmarks< TNL::Devices::Cuda, CSRSegments >( digraph, graph, smallest, largest, benchmark, "cuda" );
         runTNLBenchmarks< TNL::Devices::Cuda, EllpackSegments >( digraph, graph, smallest, largest, benchmark, "cuda" );

   #ifdef WITH_ROW_MAJOR_SLICED_ELLPACK
         // Row-major sliced Ellpack with various segment sizes
         runTNLBenchmarks< TNL::Devices::Cuda, RowMajorSlicedEllpackSegments< 2 >::template type >(
            digraph, graph, smallest, largest, benchmark, "cuda" );
         //runTNLBenchmarks< TNL::Devices::Cuda, RowMajorSlicedEllpackSegments< 4 >::template type >(
         //   digraph, graph, smallest, largest, benchmark, "cuda" );
         runTNLBenchmarks< TNL::Devices::Cuda, RowMajorSlicedEllpackSegments< 8 >::template type >(
            digraph, graph, smallest, largest, benchmark, "cuda" );
         //runTNLBenchmarks< TNL::Devices::Cuda, RowMajorSlicedEllpackSegments< 16 >::template type >(
         //   digraph, graph, smallest, largest, benchmark, "cuda" );
         runTNLBenchmarks< TNL::Devices::Cuda, RowMajorSlicedEllpackSegments< 32 >::template type >(
            digraph, graph, smallest, largest, benchmark, "cuda" );
   #endif

         // Column-major sliced Ellpack with various segment sizes
         runTNLBenchmarks< TNL::Devices::Cuda, ColumnMajorSlicedEllpackSegments< 2 >::template type >(
            digraph, graph, smallest, largest, benchmark, "cuda" );
         //runTNLBenchmarks< TNL::Devices::Cuda, ColumnMajorSlicedEllpackSegments< 4 >::template type >(
         //   digraph, graph, smallest, largest, benchmark, "cuda" );
         runTNLBenchmarks< TNL::Devices::Cuda, ColumnMajorSlicedEllpackSegments< 8 >::template type >(
            digraph, graph, smallest, largest, benchmark, "cuda" );
         //runTNLBenchmarks< TNL::Devices::Cuda, ColumnMajorSlicedEllpackSegments< 16 >::template type >(
         //   digraph, graph, smallest, largest, benchmark, "cuda" );
         runTNLBenchmarks< TNL::Devices::Cuda, ColumnMajorSlicedEllpackSegments< 32 >::template type >(
            digraph, graph, smallest, largest, benchmark, "cuda" );

         runTNLBenchmarks< TNL::Devices::Cuda, BiEllpackSegments >( digraph, graph, smallest, largest, benchmark, "cuda" );
         runTNLBenchmarks< TNL::Devices::Cuda, ChunkedEllpackSegments >( digraph, graph, smallest, largest, benchmark, "cuda" );

   #ifdef WITH_SORTED_SEGMENTS
         if( withSortedSegments ) {
            runTNLBenchmarks< TNL::Devices::Cuda, SortedCSRSegments >( digraph, graph, smallest, largest, benchmark, "cuda" );

      #ifdef WITH_ROW_MAJOR_SLICED_ELLPACK
            // Row-major sliced Ellpack with various segment sizes
            runTNLBenchmarks< TNL::Devices::Cuda, SortedRowMajorSlicedEllpackSegments< 2 >::template type >(
               digraph, graph, smallest, largest, benchmark, "cuda" );
            //runTNLBenchmarks< TNL::Devices::Cuda, SortedRowMajorSlicedEllpackSegments< 4 >::template type >(
            //   digraph, graph, smallest, largest, benchmark, "cuda" );
            runTNLBenchmarks< TNL::Devices::Cuda, SortedRowMajorSlicedEllpackSegments< 8 >::template type >(
               digraph, graph, smallest, largest, benchmark, "cuda" );
            //runTNLBenchmarks< TNL::Devices::Cuda, SortedRowMajorSlicedEllpackSegments< 16 >::template type >(
            //   digraph, graph, smallest, largest, benchmark, "cuda" );
            runTNLBenchmarks< TNL::Devices::Cuda, SortedRowMajorSlicedEllpackSegments< 32 >::template type >(
               digraph, graph, smallest, largest, benchmark, "cuda" );
      #endif

            // Column-major sliced Ellpack with various segment sizes
            runTNLBenchmarks< TNL::Devices::Cuda, SortedColumnMajorSlicedEllpackSegments< 2 >::template type >(
               digraph, graph, smallest, largest, benchmark, "cuda" );
            //runTNLBenchmarks< TNL::Devices::Cuda, SortedColumnMajorSlicedEllpackSegments< 4 >::template type >(
            //   digraph, graph, smallest, largest, benchmark, "cuda" );
            runTNLBenchmarks< TNL::Devices::Cuda, SortedColumnMajorSlicedEllpackSegments< 8 >::template type >(
               digraph, graph, smallest, largest, benchmark, "cuda" );
            //runTNLBenchmarks< TNL::Devices::Cuda, SortedColumnMajorSlicedEllpackSegments< 16 >::template type >(
            //   digraph, graph, smallest, largest, benchmark, "cuda" );
            runTNLBenchmarks< TNL::Devices::Cuda, SortedColumnMajorSlicedEllpackSegments< 32 >::template type >(
               digraph, graph, smallest, largest, benchmark, "cuda" );
         }
   #endif  // WITH_SORTED_SEGMENTS
      }
#endif  // __CUDACC__

      if( errors == 0 )
         return true;
      return false;
   }

protected:
   template< typename Device, template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments >
   void
   runTNLBenchmarks( const HostDigraph& hostDigraph,
                     const HostGraph& hostGraph,
                     IndexType smallestNode,
                     IndexType largestNode,
                     TNL::Benchmarks::Benchmark<>& benchmark,
                     const TNL::String& device )
   {
      auto& derived = static_cast< Derived& >( *this );
      using Graph = TNL::Graphs::Graph< Real, Device, Index, TNL::Graphs::UndirectedGraph, Segments >;
      using Digraph = TNL::Graphs::Graph< Real, Device, Index, TNL::Graphs::DirectedGraph, Segments >;

      Digraph digraph( hostDigraph );
      Graph graph( hostGraph );
      benchmark.setMetadataElement( { "solver", "TNL" } );
      auto segments = graph.getAdjacencyMatrix().getSegments().getSegmentsType();

      // Call the derived class implementation
      derived.runTNLAlgorithm( digraph, graph, smallestNode, largestNode, benchmark, device, segments );
   }

   // Common data members
   const TNL::Config::ParameterContainer& parameters;
   int verbose = 0;
   int errors = 0;
   bool withSortedSegments = false;
};

}  // namespace TNL::Benchmarks::Graphs
