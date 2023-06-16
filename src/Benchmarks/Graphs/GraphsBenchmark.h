// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Graphs/GraphReader.h>
#include <TNL/Graphs/GraphWriter.h>
#include <TNL/Graphs/breadthFirstSearch.h>
#include <TNL/Graphs/singleSourceShortestPath.h>
#include <TNL/Graphs/minimumSpanningTree.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixOperations.h>
#include "BoostGraph.h"
#include "GunrockBenchmark.h"

namespace TNL::Benchmarks::Graphs {

template< typename Real = double,
          typename Index = int >
struct GraphsBenchmark
{
   using HostMatrix = TNL::Matrices::SparseMatrix<Real, TNL::Devices::Host, Index>;
   using HostGraph = TNL::Graphs::Graph< HostMatrix, TNL::Graphs::Undirected >;
   using HostDigraph = TNL::Graphs::Graph< HostMatrix, TNL::Graphs::Directed >;
   using HostIndexVector = TNL::Containers::Vector<Index, TNL::Devices::Host, Index>;
   using HostRealVector = TNL::Containers::Vector<Real, TNL::Devices::Host, Index>;

   static void configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter("Benchmark settings:");
      config.addEntry<TNL::String>("input-file", "Input file with the graph." );
      config.addEntry<TNL::String>("log-file", "Log file name.", "tnl-benchmark-graphs.log");
      config.addEntry<TNL::String>("output-mode", "Mode for opening the log file.", "overwrite");
      config.addEntryEnum("append");
      config.addEntryEnum("overwrite");

      config.addDelimiter( "Device settings:" );
      config.addEntry< TNL::String >( "device", "Device the computation will run on.", "all" );
      config.addEntryEnum< TNL::String >( "all" );
      config.addEntryEnum< TNL::String >( "host" );
      config.addEntryEnum< TNL::String >( "sequential" );
      config.addEntryEnum< TNL::String >( "cuda" );
      TNL::Devices::Host::configSetup( config );
      TNL::Devices::Cuda::configSetup( config );

      config.addEntry<int>("loops", "Number of iterations for every computation.", 10);
      config.addEntry<int>("verbose", "Verbose mode.", 1);
   }

   GraphsBenchmark( const TNL::Config::ParameterContainer& parameters_ ) : parameters( parameters_ ){}

   template< typename Device,
      template< typename Device_,
                typename Index_,
                typename IndexAllocator_ > class Segments = TNL::Algorithms::Segments::CSRScalar >
   void TNLBenchmarks( const HostDigraph& hostDigraph,
                       const HostGraph& hostGraph,
                       TNL::Benchmarks::Benchmark<>& benchmark,
                       const TNL::String& device,
                       const TNL::String& segments )
   {
      using Matrix = TNL::Matrices::SparseMatrix<Real, Device, Index, TNL::Matrices::GeneralMatrix, Segments >;
      using Graph = TNL::Graphs::Graph< Matrix, TNL::Graphs::Undirected >;
      using Digraph = TNL::Graphs::Graph< Matrix, TNL::Graphs::Directed >;
      using Graph = TNL::Graphs::Graph< Matrix, TNL::Graphs::Undirected >;
      using IndexVector = TNL::Containers::Vector<Index, Device, Index>;
      using RealVector = TNL::Containers::Vector<Real, Device, Index>;

      Digraph digraph( hostDigraph );

      // Benchmarking breadth-first search
      IndexVector bfsDistances( digraph.getNodeCount() );
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType<Index>() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "BFS TNL" ) } } ) );

      auto bfs_tnl = [&] () mutable {
         TNL::Graphs::breadthFirstSearch( digraph, 0, bfsDistances );
      };
      benchmark.time< Device >( device, bfs_tnl );
      if( bfsDistances != this->boostBfsDistances )
      {
         std::cout << "BFS distances from Boost and TNL are not equal!" << std::endl;
         this->errors++;
      }

      // Benchmarking single-source shortest paths
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType<Real>() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "SSSP TNL" ) } } ) );

      RealVector ssspDistances( digraph.getNodeCount(), 0 );
      auto sssp_tnl = [&] () mutable {
         TNL::Graphs::singleSourceShortestPath( digraph, 0, ssspDistances );
      };
      benchmark.time< Device >( device, sssp_tnl );

      if( ssspDistances != this->boostSSSPDistances )
      {
         std::cout << "SSSP distances from Boost and TNL are not equal!" << std::endl;
         std::cout << "Boost: " << this->boostSSSPDistances << std::endl;
         std::cout << "TNL: " << ssspDistances << std::endl;
         this->errors++;
         abort();
      }

      // Benchmarking minimum spanning tree
      HostMatrix symmetrizedAdjacencyMatrix;
      TNL::Matrices::makeSymmetric( digraph.getAdjacencyMatrix(), symmetrizedAdjacencyMatrix );
      Graph graph, mstGraph;
      graph.setAdjacencyMatrix( symmetrizedAdjacencyMatrix );
      IndexVector roots;
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType<Real>() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "MST TNL" ) } } ) );
      auto mst_tnl = [&] () mutable {
         TNL::Graphs::minimumSpanningTree( graph, mstGraph, roots );
      };
      benchmark.time< Device >( device, mst_tnl );
      auto filename = this->parameters.getParameter< TNL::String >( "input-file" );
      TNL::Graphs::GraphWriter< Graph >::writeEdgeList( filename + "-tnl-mst.txt", mstGraph );
      Real mstTotalWeight = mstGraph.getTotalWeight();
      if( mstTotalWeight != boostMSTTotalWeight )
      {
         std::cout << "ERROR: Total weights of boost MST and TNL MST do not match!" << std::endl;
         std::cout << "Boost MST total weight: " << boostMSTTotalWeight << std::endl;
         std::cout << "TNL MST total weight: " << mstTotalWeight << std::endl;
         this->errors++;
      }

   }

   void boostBenchmarks( const HostDigraph& digraph,
                         const HostGraph& graph,
                         TNL::Benchmarks::Benchmark<>& benchmark )
   {
      benchmark.setMetadataColumns(
      TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType<Index>() },
                                                       { "device", "sequential" },
                                                       { "format", "N/A" },
                                                       { "algorithm", std::string( "BFS Boost" ) } } ) );
      BoostGraph< Index, Real, TNL::Graphs::Directed > boostDiraph( digraph );
      BoostGraph< Index, Real, TNL::Graphs::Undirected > boostGraph( graph );

      // Benchmarking breadth-first search
      std::vector<Index> boostBfsDistances( digraph.getNodeCount() );
      auto bfs_boost = [&] () mutable {
         boostGraph.breadthFirstSearch( 0, boostBfsDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", bfs_boost );

      HostIndexVector boost_bfs_dist( boostBfsDistances );
      boost_bfs_dist.forAllElements( [] __cuda_callable__ ( Index i, Index& x ) { x = x == std::numeric_limits< Index >::max() ? -1 : x; } );
      this->boostBfsDistances = boost_bfs_dist;

      // Benchmarking single-source shortest paths
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType<Real>() },
                                                          { "device", "sequential" },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "SSSP Boost" ) } } ) );
      std::vector<Real> boostSSSPDistances( digraph.getNodeCount() );
      auto sssp_boost = [&] () mutable {
         boostGraph.singleSourceShortestPath( 0, boostSSSPDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", sssp_boost );
      HostRealVector boost_sssp_dist( boostSSSPDistances );
      boost_sssp_dist.forAllElements( [] __cuda_callable__ ( Index i, Real& x ) { x = x == std::numeric_limits< Real >::max() ? -1 : x; } );
      this->boostSSSPDistances = boost_sssp_dist;

      // Benchmarking minimum spanning tree
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType<Real>() },
                                                          { "device", "sequential" },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "MST Boost" ) } } ) );
      using BoostEdge = typename BoostGraph< Index, Real >::Edge;
      std::vector< BoostEdge > boostMstEdges;
      auto mst_boost = [&] () mutable {
         boostGraph.minimumSpanningTree( boostMstEdges );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", mst_boost );
      this->boostMSTTotalWeight = 0.0;
      for (auto &edge : boostMstEdges) {
        Real weight = boost::get(boost::edge_weight, boostGraph.getGraph(), edge);
        this->boostMSTTotalWeight += weight;
        //std::cout << boost::source(edge, g) << " <--> " << boost::target(edge, g) << " [weight = " << weight << "]" << std::endl;
      }
      auto filename = this->parameters.getParameter< TNL::String >( "input-file" );
      boostGraph.exportMst( boostMstEdges, filename + "-boost-mst.txt" );
   }

#ifdef HAVE_GUNROCK
   void gunrockBenchmarks( const HostDigraph& digraph, TNL::Benchmarks::Benchmark<>& benchmark )
   {
      auto filename = this->parameters.getParameter< TNL::String >( "input-file" );
      std::vector<int> row_offsets;
      std::vector<int> column_indices, values;

      const auto& adjacencyMatrix = digraph.getAdjacencyMatrix();
      TNL::copy( adjacencyMatrix.getSegments().getOffsets(), row_offsets );
      TNL::copy( adjacencyMatrix.getColumnIndexes(), column_indices );
      TNL::copy( adjacencyMatrix.getValues(), values );

      thrust::device_vector< Index > d_row_offsets( adjacencyMatrix.getRows() + 1 );
      thrust::device_vector< Index > d_column_indices( adjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< Real > d_values( adjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< Index > d_row_indices( adjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< Index > d_column_offsets( adjacencyMatrix.getColumns() + 1 );
      thrust::copy( row_offsets.begin(), row_offsets.end(), d_row_offsets.begin() );
      thrust::copy( column_indices.begin(), column_indices.end(), d_column_indices.begin() );
      thrust::copy( values.begin(), values.end(), d_values.begin() );

      auto graph =
        gunrock::graph::build::from_csr<gunrock::memory_space_t::device, gunrock::graph::view_t::csr >(
            adjacencyMatrix.getRows(),                 // rows
            adjacencyMatrix.getColumns(),              // columns
            adjacencyMatrix.getNonzeroElementsCount(), // nonzeros
            d_row_offsets.data().get(),                // row_offsets
            d_column_indices.data().get(),             // column_indices
            d_values.data().get(),                     // values
            d_row_indices.data().get(),                // row_indices
            d_column_offsets.data().get()              // column_offsets
        );

      GunrockBenchmark< Real, Index > gunrockBenchmark;
      Index start = 0;

      // Benchmarking breadth-first search
      benchmark.setDatasetSize( adjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType<Index>() },
                                                          { "device", std::string( "GPU" ) },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "BFS Gunrock" ) } } ) );
      std::vector< Index > bfsDistances( adjacencyMatrix.getRows() );
      gunrockBenchmark.breadthFirstSearch( benchmark, graph, start, adjacencyMatrix.getRows(), bfsDistances );
      this->gunrockBfsDistances = bfsDistances;

      benchmark.setDatasetSize( adjacencyMatrix.getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType<Index>() },
                                                          { "device", std::string( "GPU" ) },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "SSSP Gunrock" ) } } ) );
      std::vector< Real > ssspDistances( adjacencyMatrix.getRows() );
      gunrockBenchmark.singleSourceShortestPath( benchmark, graph, start, adjacencyMatrix.getRows(), ssspDistances );
      this->gunrockSSSPDistances = ssspDistances;

      if( this->boostBfsDistances != this->gunrockBfsDistances ) {
         std::cout << "BFS distances from Boost and Gunrock are not equal!" << std::endl;
         this->errors++;
      }
      if( this->boostSSSPDistances != this->gunrockSSSPDistances ) {
         std::cout << "SSSP distances from Boost and Gunrock are not equal!" << std::endl;
         this->errors++;
      }
   }
#endif

   bool runBenchmark()
   {
      auto inputFile = parameters.getParameter< TNL::String >( "input-file" );
      const TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const TNL::String outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const int loops = parameters.getParameter< int >("loops");
      const int verbose = parameters.getParameter< int >("verbose");

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark(logFile, loops, verbose);

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      this->errors = 0;

      TNL::String device = parameters.getParameter< TNL::String >( "device" );

      std::cout << "Graphs benchmark  with " << TNL::getType<Real>() << " precision and device: " << device << std::endl;

      HostDigraph digraph;
      std::cout << "Reading graph from file " << inputFile << std::endl;
      TNL::Graphs::GraphReader< HostDigraph >::readEdgeList( inputFile, digraph );

      HostMatrix symmetrizedAdjacencyMatrix;
      TNL::Matrices::makeSymmetric( digraph.getAdjacencyMatrix(), symmetrizedAdjacencyMatrix );
      HostGraph graph( symmetrizedAdjacencyMatrix );
      TNL::Graphs::GraphWriter< HostGraph >::writeEdgeList( inputFile+"-undirected.txt", graph );


      boostBenchmarks( digraph, graph, benchmark );
#ifdef HAVE_GUNROCK
      gunrockBenchmarks( digraph, benchmark );
#endif

      if( device == "sequential" || device == "all" )
         TNLBenchmarks< TNL::Devices::Sequential, TNL::Algorithms::Segments::CSRScalar >( digraph, graph, benchmark, "sequential", "CSRScalar" );
      if( device == "host" || device == "all" )
         TNLBenchmarks< TNL::Devices::Host, TNL::Algorithms::Segments::CSRScalar >( digraph, graph, benchmark, "host", "CSRScalar" );
#ifdef __CUDACC__
      if( device == "cuda" || device == "all" )
      {
         TNLBenchmarks< TNL::Devices::Cuda, TNL::Algorithms::Segments::CSRScalar     >( digraph, graph, benchmark, "cuda", "CSRScalar" );
         TNLBenchmarks< TNL::Devices::Cuda, TNL::Algorithms::Segments::CSRVector     >( digraph, graph, benchmark, "cuda", "CSRVector" );
         TNLBenchmarks< TNL::Devices::Cuda, TNL::Algorithms::Segments::CSRLight      >( digraph, graph, benchmark, "cuda", "CSRLight" );
         TNLBenchmarks< TNL::Devices::Cuda, TNL::Algorithms::Segments::CSRAdaptive   >( digraph, graph, benchmark, "cuda", "CSRAdaptive" );
         TNLBenchmarks< TNL::Devices::Cuda, TNL::Algorithms::Segments::Ellpack       >( digraph, graph, benchmark, "cuda", "Ellpack" );
         TNLBenchmarks< TNL::Devices::Cuda, TNL::Algorithms::Segments::SlicedEllpack >( digraph, graph, benchmark, "cuda", "SlicedEllpack" );
         TNLBenchmarks< TNL::Devices::Cuda, TNL::Algorithms::Segments::BiEllpack     >( digraph, graph, benchmark, "cuda", "BiEllpack" );
      }
#endif
      return true;
   }

protected:
   const TNL::Config::ParameterContainer& parameters;

   // These vectors serve as a reference solution for comparison with TNL
   HostIndexVector boostBfsDistances;
   HostRealVector boostSSSPDistances;

   HostIndexVector gunrockBfsDistances;
   HostRealVector gunrockSSSPDistances;

   Real boostMSTTotalWeight;

   int errors;
};

} // namespace TNL::Benchmarks::Graphs
