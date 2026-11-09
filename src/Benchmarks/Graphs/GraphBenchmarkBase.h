// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Benchmarks/Devices.h>
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
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Graphs/Algorithms/breadthFirstSearch.h>
#include <climits>
#include <new>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "GapGraphReader.h"
#include "GraphBenchmarkTools.h"

//#define WITH_ROW_MAJOR_SLICED_ELLPACK
//#define WITH_SORTED_SEGMENTS

namespace TNL::Benchmarks::Graphs {

namespace detail {

// Detects an opt-in `static constexpr bool needsNonNegativeWeights` on the
// CRTP Derived class (e.g. GraphBenchmarkSSSP, since Dijkstra requires
// non-negative weights); defaults to false so other benchmarks (CC, SCC,
// Coloring, MST, BFS) don't need to declare it explicitly.
template< typename Derived, typename = void >
struct NeedsNonNegativeWeights : std::false_type
{};

template< typename Derived >
struct NeedsNonNegativeWeights< Derived, std::void_t< decltype( Derived::needsNonNegativeWeights ) > >
: std::bool_constant< Derived::needsNonNegativeWeights >
{};

// Detects an opt-in `static constexpr int defaultTrials` on the CRTP Derived class, i.e. the number of trials
// prescribed by GAP (64 sources for BFS and SSSP, 16 trials for CC). Benchmarks without it (Coloring, SCC, MST)
// keep using the `loops` parameter.
template< typename Derived, typename = void >
struct DefaultTrials : std::integral_constant< int, 0 >
{};

template< typename Derived >
struct DefaultTrials< Derived, std::void_t< decltype( Derived::defaultTrials ) > >
: std::integral_constant< int, Derived::defaultTrials >
{};

// Detects an opt-in `static constexpr bool isSingleSource` on the CRTP Derived class (BFS, SSSP). The base class
// then picks the source vertices and computes the numbers of traversed edges for them.
template< typename Derived, typename = void >
struct IsSingleSource : std::false_type
{};

template< typename Derived >
struct IsSingleSource< Derived, std::void_t< decltype( Derived::isSingleSource ) > >
: std::bool_constant< Derived::isSingleSource >
{};

}  // namespace detail

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
      Benchmark::configSetup( config );
      config.addDelimiter( "Graph benchmark settings:" );
      config.addRequiredEntry< TNL::String >(
         "input-file",
         "Input file with the graph: .sg or .wsg file of the GAP benchmark (see get-gap-graphs.py), .mtx file, or edge "
         "list. The undirected variant of an .mtx file or an edge list is computed by symmetrization." );
      config.addEntry< TNL::String >(
         "undirected-input-file",
         "Input file with the undirected (symmetrized) graph for a directed .sg or .wsg input file. The default is "
         "<name>U.sg or <name>U.wsg in the directory of the input file.",
         "" );
      config.addEntry< TNL::String >( "orientation", "Which variants of the graph are benchmarked.", "both" );
      config.addEntryEnum< TNL::String >( "both" );
      config.addEntryEnum< TNL::String >( "directed" );
      config.addEntryEnum< TNL::String >( "undirected" );
      config.addEntry< int >(
         "trials",
         "Number of trials: the number of source vertices for BFS and SSSP, the number of repetitions for CC. "
         "0 means the number prescribed by the GAP benchmark (64 for BFS and SSSP, 16 for CC). Each trial is timed "
         "separately; the benchmarks without a GAP specification use the loops parameter instead.",
         0 );
      config.addEntry< TNL::String >( "verify", "Which trials are verified.", "all" );
      config.addEntryEnum< TNL::String >( "all" );
      config.addEntryEnum< TNL::String >( "first" );
      config.addEntryEnum< TNL::String >( "none" );
      config.addEntry< int >(
         "max-boost-edges",
         "Boost is skipped for graphs with more edges, since its adjacency list needs tens of bytes per edge.",
         200000000 );
      config.addEntry< TNL::String >( "segments", "Segments used on GPU.", "all" );
      config.addEntryEnum< TNL::String >( "all" );
      config.addEntryEnum< TNL::String >( "csr" );
#ifdef WITH_SORTED_SEGMENTS
      config.addEntry< bool >( "with-sorted-segments", "Run benchmark with sorted segments.", true );
#endif
      config.addDelimiter( "Device settings:" );
      config.addEntry< TNL::String >( "device", "Device the computation will run on.", "all" );
      config.addEntryEnum< TNL::String >( "host" );
      config.addEntryEnum< TNL::String >( "sequential" );
      config.addEntryEnum< TNL::String >( "cuda" );
      config.addEntryEnum< TNL::String >( "hip" );
      config.addEntryEnum< TNL::String >( "all" );
      TNL::Devices::Host::configSetup( config );
      TNL::Devices::GPU::configSetup( config );
   }

   GraphBenchmarkBase( const TNL::Config::ParameterContainer& parameters_ )
   : parameters( parameters_ )
   {}

   virtual ~GraphBenchmarkBase() = default;

   /**
    * \brief Returns the name of the file with the undirected variant of a directed GAP graph.
    */
   [[nodiscard]] static std::string
   getUndirectedInputFile( const TNL::Config::ParameterContainer& parameters )
   {
      const std::string file = parameters.getParameter< TNL::String >( "undirected-input-file" );
      if( ! file.empty() )
         return file;
      return getGapUndirectedFileName( parameters.getParameter< TNL::String >( "input-file" ) );
   }

   /**
    * \brief Checks whether the graphs to be benchmarked need a 64-bit index type.
    *
    * Only the .sg and .wsg files of the GAP benchmark are checked; the other formats are always read with 32-bit indexes.
    * The vertex count and the number of stored edges must fit in the index type, since TNL uses the same type for the
    * vertex indexes and for the offsets of the adjacency matrix.
    */
   [[nodiscard]] static bool
   requiresLongIndex( const TNL::Config::ParameterContainer& parameters )
   {
      const std::string inputFile = parameters.getParameter< TNL::String >( "input-file" );
      if( ! isGapGraphFile( inputFile ) )
         return false;
      const std::string orientation = parameters.getParameter< TNL::String >( "orientation" );
      const auto exceedsInt = []( const GapGraphHeader& header )
      {
         return header.vertices > INT_MAX || header.edges > INT_MAX;
      };
      const GapGraphHeader header = readGapGraphHeader( inputFile );
      bool result = false;
      if( header.directed && orientation != "undirected" )
         result = result || exceedsInt( header );
      if( orientation != "directed" ) {
         const GapGraphHeader undirectedHeader =
            header.directed ? readGapGraphHeader( getUndirectedInputFile( parameters ) ) : header;
         result = result || exceedsInt( undirectedHeader );
      }
      return result;
   }

   bool
   runBenchmark( TNL::Benchmarks::Benchmark& benchmark, const std::string& programName = "" )
   {
      const std::string inputFile = parameters.getParameter< TNL::String >( "input-file" );
#ifdef WITH_SORTED_SEGMENTS
      withSortedSegments = parameters.getParameter< bool >( "with-sorted-segments" );
#endif

      this->errors = 0;

      auto device = parameters.getParameter< TNL::String >( "device" );
      const std::string orientation = parameters.getParameter< TNL::String >( "orientation" );
      verifyMode = parameters.getParameter< TNL::String >( "verify" );
      loops = parameters.getParameter< int >( "loops" );
      warmupLoops = parameters.getParameter< int >( "warmup-loops" );
      trials = parameters.getParameter< int >( "trials" );
      if( trials <= 0 )
         trials = detail::DefaultTrials< Derived >::value > 0 ? detail::DefaultTrials< Derived >::value : loops;

      std::cout << "Graphs benchmark with " << TNL::getType< Real >() << " precision, " << TNL::getType< Index >()
                << " index and device: " << device << '\n';

      HostDigraph digraph;
      HostGraph graph;
      if( isGapGraphFile( inputFile ) ) {
         gapInput = true;
         const GapGraphHeader header = readGapGraphHeader( inputFile );
         runDirected = header.directed && orientation != "undirected";
         runUndirected = orientation != "directed";
         if( ! header.directed && orientation == "directed" )
            std::cout << "The graph " << inputFile << " is undirected, it has no directed variant to benchmark.\n";
         if( runDirected ) {
            directedInputFile = inputFile;
            std::cout << "Reading directed graph from file " << directedInputFile << '\n';
            HostMatrix matrix;
            readGapGraph( directedInputFile, matrix );
            digraph.setAdjacencyMatrix( std::move( matrix ) );
         }
         if( runUndirected ) {
            undirectedInputFile = header.directed ? getUndirectedInputFile( parameters ) : inputFile;
            std::cout << "Reading undirected graph from file " << undirectedInputFile << '\n';
            HostMatrix matrix;
            if( readGapGraph( undirectedInputFile, matrix ).directed )
               throw std::runtime_error( "The graph in " + undirectedInputFile + " is not undirected." );
            graph.setAdjacencyMatrix( std::move( matrix ) );
         }
      }
      else {
         size_t dotPosition = inputFile.find_last_of( '.' );
         std::string inputFileExtension;
         if( dotPosition != std::string::npos )
            inputFileExtension = inputFile.substr( dotPosition + 1 );
         std::cout << "Reading graph from file " << inputFile << '\n';
         if( inputFileExtension == "mtx" )
            TNL::Graphs::Readers::MtxReader< HostDigraph >::read( inputFile, digraph );
         else
            TNL::Graphs::Readers::EdgeListReader< HostDigraph >::read( inputFile, digraph );
         runDirected = orientation != "undirected";
         runUndirected = orientation != "directed";
         directedInputFile = inputFile;
      }

      // Dijkstra (used for SSSP) requires non-negative weights; other
      // algorithms don't need this and some (e.g. MST) actually care about
      // the true, possibly negative weights, so only SSSP opts in.
      if constexpr( detail::NeedsNonNegativeWeights< Derived >::value ) {
         digraph.getAdjacencyMatrix().getValues() = abs( digraph.getAdjacencyMatrix().getValues() );
         graph.getAdjacencyMatrix().getValues() = abs( graph.getAdjacencyMatrix().getValues() );
      }

      if( ! gapInput ) {
         auto symmetrizedAdjacencyMatrix = TNL::Matrices::getSymmetricPart< HostMatrix >( digraph.getAdjacencyMatrix() );
         graph = HostGraph( symmetrizedAdjacencyMatrix );
         if( ! runDirected )
            digraph = HostDigraph();
      }

      // The graph used for the metadata and for the benchmarks without a GAP specification.
      const auto& adjacencyMatrix =
         runUndirected ? graph.getAdjacencyMatrix() : static_cast< const HostMatrix& >( digraph.getAdjacencyMatrix() );
      HostIndexVector nodeDegrees( adjacencyMatrix.getRows(), 0 );
      adjacencyMatrix.getCompressedRowLengths( nodeDegrees );
      Index largest = 0;
      Index smallest = 0;
      if( nodeDegrees.getSize() > 0 ) {
         largest = TNL::argMax( nodeDegrees ).second;
         smallest = TNL::argMax( greater( nodeDegrees, 0 ) ).second;
         std::cout << "Smallest degree is " << nodeDegrees[ smallest ] << " at position " << smallest << '\n';
         std::cout << "Largest degree is " << nodeDegrees[ largest ] << " at position " << largest << '\n';
      }

      benchmark.setMetadataColumns(
         {
            { "graph name", inputFile },
            { "precision", getType< Real >() },
            { "index type", TNL::getType< Index >() },
            { "nodes", convertToString( adjacencyMatrix.getRows() ) },
            { "edges", convertToString( adjacencyMatrix.getNonzeroElementsCount() ) },
         } );

      hostDigraphPointer = &digraph;
      hostGraphPointer = &graph;

      if constexpr( detail::IsSingleSource< Derived >::value ) {
         if( runDirected )
            prepareSources( digraph, directedSources, directedTraversedEdges, "directed" );
         if( runUndirected )
            prepareSources( graph, undirectedSources, undirectedTraversedEdges, "undirected" );
      }

      auto& derived = static_cast< Derived& >( *this );

      // Run benchmarks for different libraries and devices
      derived.runOtherBenchmarks( digraph, graph, smallest, largest, benchmark );

      if( device == "sequential" || device == "all" )
         runTNLBenchmarks< TNL::Devices::Sequential, CSRSegments >( digraph, graph, smallest, largest, benchmark );
      if( device == "host" || device == "all" )
         runTNLBenchmarks< TNL::Devices::Host, CSRSegments >( digraph, graph, smallest, largest, benchmark );

#if defined( __CUDACC__ ) || defined( __HIP__ )
      if( device == "cuda" || device == "hip" || device == "all" ) {
         runTNLBenchmarks< TNL::Devices::GPU, CSRSegments >( digraph, graph, smallest, largest, benchmark );
      }
      if( ( device == "cuda" || device == "hip" || device == "all" )
          && parameters.getParameter< TNL::String >( "segments" ) == "all" )
      {
         runTNLBenchmarks< TNL::Devices::GPU, EllpackSegments >( digraph, graph, smallest, largest, benchmark );

   #ifdef WITH_ROW_MAJOR_SLICED_ELLPACK
         // Row-major sliced Ellpack with various segment sizes
         runTNLBenchmarks< TNL::Devices::GPU, RowMajorSlicedEllpackSegments< 2 >::template type >(
            digraph, graph, smallest, largest, benchmark );
         //runTNLBenchmarks< TNL::Devices::GPU, RowMajorSlicedEllpackSegments< 4 >::template type >(
         //   digraph, graph, smallest, largest, benchmark );
         runTNLBenchmarks< TNL::Devices::GPU, RowMajorSlicedEllpackSegments< 8 >::template type >(
            digraph, graph, smallest, largest, benchmark );
         //runTNLBenchmarks< TNL::Devices::GPU, RowMajorSlicedEllpackSegments< 16 >::template type >(
         //   digraph, graph, smallest, largest, benchmark );
         runTNLBenchmarks< TNL::Devices::GPU, RowMajorSlicedEllpackSegments< 32 >::template type >(
            digraph, graph, smallest, largest, benchmark );
   #endif

         // Column-major sliced Ellpack with various segment sizes
         runTNLBenchmarks< TNL::Devices::GPU, ColumnMajorSlicedEllpackSegments< 2 >::template type >(
            digraph, graph, smallest, largest, benchmark );
         //runTNLBenchmarks< TNL::Devices::GPU, ColumnMajorSlicedEllpackSegments< 4 >::template type >(
         //   digraph, graph, smallest, largest, benchmark );
         runTNLBenchmarks< TNL::Devices::GPU, ColumnMajorSlicedEllpackSegments< 8 >::template type >(
            digraph, graph, smallest, largest, benchmark );
         //runTNLBenchmarks< TNL::Devices::GPU, ColumnMajorSlicedEllpackSegments< 16 >::template type >(
         //   digraph, graph, smallest, largest, benchmark );
         runTNLBenchmarks< TNL::Devices::GPU, ColumnMajorSlicedEllpackSegments< 32 >::template type >(
            digraph, graph, smallest, largest, benchmark );

         runTNLBenchmarks< TNL::Devices::GPU, BiEllpackSegments >( digraph, graph, smallest, largest, benchmark );
         runTNLBenchmarks< TNL::Devices::GPU, ChunkedEllpackSegments >( digraph, graph, smallest, largest, benchmark );

   #ifdef WITH_SORTED_SEGMENTS
         if( withSortedSegments ) {
            runTNLBenchmarks< TNL::Devices::GPU, SortedCSRSegments >( digraph, graph, smallest, largest, benchmark );

      #ifdef WITH_ROW_MAJOR_SLICED_ELLPACK
            // Row-major sliced Ellpack with various segment sizes
            runTNLBenchmarks< TNL::Devices::GPU, SortedRowMajorSlicedEllpackSegments< 2 >::template type >(
               digraph, graph, smallest, largest, benchmark );
            //runTNLBenchmarks< TNL::Devices::GPU, SortedRowMajorSlicedEllpackSegments< 4 >::template type >(
            //   digraph, graph, smallest, largest, benchmark );
            runTNLBenchmarks< TNL::Devices::GPU, SortedRowMajorSlicedEllpackSegments< 8 >::template type >(
               digraph, graph, smallest, largest, benchmark );
            //runTNLBenchmarks< TNL::Devices::GPU, SortedRowMajorSlicedEllpackSegments< 16 >::template type >(
            //   digraph, graph, smallest, largest, benchmark );
            runTNLBenchmarks< TNL::Devices::GPU, SortedRowMajorSlicedEllpackSegments< 32 >::template type >(
               digraph, graph, smallest, largest, benchmark );
      #endif

            // Column-major sliced Ellpack with various segment sizes
            runTNLBenchmarks< TNL::Devices::GPU, SortedColumnMajorSlicedEllpackSegments< 2 >::template type >(
               digraph, graph, smallest, largest, benchmark );
            //runTNLBenchmarks< TNL::Devices::GPU, SortedColumnMajorSlicedEllpackSegments< 4 >::template type >(
            //   digraph, graph, smallest, largest, benchmark );
            runTNLBenchmarks< TNL::Devices::GPU, SortedColumnMajorSlicedEllpackSegments< 8 >::template type >(
               digraph, graph, smallest, largest, benchmark );
            //runTNLBenchmarks< TNL::Devices::GPU, SortedColumnMajorSlicedEllpackSegments< 16 >::template type >(
            //   digraph, graph, smallest, largest, benchmark );
            runTNLBenchmarks< TNL::Devices::GPU, SortedColumnMajorSlicedEllpackSegments< 32 >::template type >(
               digraph, graph, smallest, largest, benchmark );
         }
   #endif  // WITH_SORTED_SEGMENTS
      }
#endif  // __CUDACC__ || __HIP__

      hostDigraphPointer = nullptr;
      hostGraphPointer = nullptr;
      if( errors == 0 )
         return true;
      return false;
   }

protected:
   template< typename Device, template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments >
   void
   runTNLBenchmarks(
      const HostDigraph& hostDigraph,
      const HostGraph& hostGraph,
      IndexType smallestNode,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
      auto& derived = static_cast< Derived& >( *this );
      using Graph = TNL::Graphs::Graph< Real, Device, Index, TNL::Graphs::UndirectedGraph, Segments >;
      using Digraph = TNL::Graphs::Graph< Real, Device, Index, TNL::Graphs::DirectedGraph, Segments >;

      const std::string segments = typename Graph::AdjacencyMatrixType::SegmentsType{}.getSegmentsType();
      try {
         // A segments format may need too much memory for a graph (e.g. Ellpack pads all rows to the largest degree),
         // such a format is skipped.
         Digraph digraph;
         Graph graph;
         if( runDirected || ! gapInput )
            digraph = Digraph( hostDigraph );
         if( runUndirected || ! gapInput )
            graph = Graph( hostGraph );
         benchmark.setMetadataElement( { "solver", "TNL" } );

         // Call the derived class implementation
         derived.runTNLAlgorithm( digraph, graph, smallestNode, largestNode, benchmark, segments );
      }
      catch( const std::bad_alloc& e ) {
         // also TNL::Exceptions::BackendBadAlloc, which is derived from std::bad_alloc
         std::cerr << "Skipping the benchmarks on " << TNL::Benchmarks::getDeviceName< Device >() << " with " << segments
                   << " segments, the graph does not fit in the memory: " << e.what() << '\n';
      }
   }

   /**
    * \brief Picks the GAP source vertices of a graph and counts the edges traversed from them.
    *
    * The traversed edges are counted from the distances of a parallel host BFS, which are verified before they are used.
    */
   template< typename HostGraphType >
   void
   prepareSources(
      const HostGraphType& hostGraph,
      std::vector< Index >& sources,
      std::vector< double >& traversedEdges,
      const std::string& graphTag )
   {
      const auto& matrix = hostGraph.getAdjacencyMatrix();
      const auto& offsets = matrix.getSegments().getOffsets();
      sources = pickGapSources(
         hostGraph.getVertexCount(),
         trials,
         [ & ]( Index v )
         {
            return offsets[ v + 1 ] - offsets[ v ];
         } );
      traversedEdges.clear();
      HostIndexVector distances;
      for( const Index source : sources ) {
         TNL::Graphs::Algorithms::breadthFirstSearch( hostGraph, source, distances );
         const auto verification = verifyBFS( matrix, source, distances.getData() );
         if( verification.violations != 0 )
            throw std::runtime_error( "The BFS used for counting the traversed edges failed: " + verification.message );
         traversedEdges.push_back( verification.traversedEdges );
      }
      std::cout << "Picked " << sources.size() << " source vertices of the " << graphTag << " graph";
      if( ! sources.empty() ) {
         double sum = 0;
         for( const double edges : traversedEdges )
            sum += edges;
         std::cout << ", the average number of traversed edges is " << sum / sources.size();
      }
      std::cout << '\n';
   }

   [[nodiscard]] bool
   shouldVerify( int trial ) const
   {
      return verifyMode == "all" || ( verifyMode == "first" && trial == 0 );
   }

   void
   reportVerification( const GraphVerificationResult& result, const std::string& what )
   {
      if( result.violations == 0 )
         return;
      std::cout << "VERIFICATION FAILED: " << what << ": " << result.message << " (" << result.violations << " violations)\n";
      errors++;
   }

   /**
    * \brief Times a single-source kernel from all source vertices, each trial separately as prescribed by GAP.
    *
    * Each trial is one timed run of the kernel from one source vertex, logged with the `trial` and `source`
    * metadata. The first trial of each configuration is preceded by the configured number of warmup runs, the other
    * trials are not. The `reset` callable is called before each run outside of the measured time; it should release
    * the outputs, so that their allocation in `compute` is measured. The output of each verified trial is checked by
    * `check`.
    *
    * \param compute is called with the source vertex.
    * \param check is called with the source vertex and returns \ref GraphVerificationResult.
    */
   template< typename Device, typename Reset, typename Compute, typename Check >
   void
   timeSingleSourceTrials(
      TNL::Benchmarks::Benchmark& benchmark,
      const std::string& performer,
      bool directed,
      const std::string& what,
      Reset&& reset,
      Compute&& compute,
      Check&& check )
   {
      const auto& sources = directed ? directedSources : undirectedSources;
      const auto& traversedEdges = directed ? directedTraversedEdges : undirectedTraversedEdges;
      benchmark.setLoops( 1 );
      for( std::size_t trial = 0; trial < sources.size(); trial++ ) {
         const Index source = sources[ trial ];
         benchmark.setMetadataElement( { "trial", convertToString( trial ) } );
         benchmark.setMetadataElement( { "source", convertToString( source ) } );
         benchmark.setWarmupLoops( trial == 0 ? warmupLoops : 0 );
         GraphBenchmarkResult result;
         result.edges = traversedEdges[ trial ];
         auto computeTrial = [ & ]()
         {
            compute( source );
         };
         benchmark.time< Device >( reset, performer, computeTrial, result );
         if( shouldVerify( trial ) )
            reportVerification(
               check( source ),
               what + " from source " + std::to_string( source ) + " (trial " + std::to_string( trial ) + ")" );
      }
      benchmark.setLoops( loops );
      benchmark.setWarmupLoops( warmupLoops );
   }

   /**
    * \brief Times a whole-graph kernel (e.g. CC) in the number of trials prescribed by GAP.
    *
    * The trials are the loops of one measurement, preceded by the configured number of warmup runs. The `reset`
    * callable is called before each run outside of the measured time. The output of the last run is checked by `check`.
    */
   template< typename Device, typename Reset, typename Compute, typename Check >
   void
   timeWholeGraphTrials(
      TNL::Benchmarks::Benchmark& benchmark,
      const std::string& performer,
      const std::string& what,
      Reset&& reset,
      Compute&& compute,
      Check&& check )
   {
      benchmark.setMetadataElement( { "trial", "all" } );
      benchmark.setMetadataElement( { "source", "N/A" } );
      benchmark.setLoops( trials );
      benchmark.setWarmupLoops( warmupLoops );
      GraphBenchmarkResult result;
      benchmark.time< Device >( reset, performer, compute, result );
      if( verifyMode != "none" )
         reportVerification( check(), what );
      benchmark.setLoops( loops );
   }

   // Common data members
   const TNL::Config::ParameterContainer& parameters;
   int errors = 0;
   bool withSortedSegments = false;

   //! \brief The input are .sg or .wsg files of the GAP benchmark.
   bool gapInput = false;
   //! \brief The directed and undirected variants of the graph are benchmarked.
   bool runDirected = true;
   bool runUndirected = true;
   std::string directedInputFile;
   std::string undirectedInputFile;

   int trials = 0;
   int loops = 10;
   int warmupLoops = 1;
   std::string verifyMode = "all";

   //! \brief The graphs on the host, used for the verification. They are valid only during \ref runBenchmark.
   const HostDigraph* hostDigraphPointer = nullptr;
   const HostGraph* hostGraphPointer = nullptr;

   //! \brief Source vertices of the single-source kernels and the numbers of edges traversed from them.
   std::vector< Index > directedSources;
   std::vector< Index > undirectedSources;
   std::vector< double > directedTraversedEdges;
   std::vector< double > undirectedTraversedEdges;
};

}  // namespace TNL::Benchmarks::Graphs
