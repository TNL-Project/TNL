// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Graphs/GraphReader.h>
#include <TNL/Graphs/GraphWriter.h>
#include <TNL/Graphs/breadthFirstSearch.h>
#include <TNL/Graphs/singleSourceShortestPath.h>
#include <TNL/Graphs/minimumSpanningTree.h>
#include <TNL/Graphs/trees.h>
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
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixOperations.h>
#include "BoostGraph.h"
#include "GunrockBenchmark.h"

namespace TNL::Benchmarks::Graphs {

template< typename Real = double, typename Index = int >
struct GraphsBenchmark
{
   using RealType = Real;
   using IndexType = Index;
   using HostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, Index >;
   using HostGraph = TNL::Graphs::Graph< HostMatrix, TNL::Graphs::GraphTypes::Undirected >;
   using HostDigraph = TNL::Graphs::Graph< HostMatrix, TNL::Graphs::GraphTypes::Directed >;
   using HostIndexVector = TNL::Containers::Vector< Index, TNL::Devices::Host, Index >;
   using HostRealVector = TNL::Containers::Vector< Real, TNL::Devices::Host, Index >;

   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using CSRSegments = TNL::Algorithms::Segments::CSR< Device_, Index_, IndexAllocator_ >;
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using EllpackSegments = TNL::Algorithms::Segments::Ellpack< Device_, Index_, IndexAllocator_ >;
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using SlicedEllpackSegments = TNL::Algorithms::Segments::SlicedEllpack< Device_, Index_, IndexAllocator_ >;
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using BiEllpackSegments = TNL::Algorithms::Segments::BiEllpack< Device_, Index_, IndexAllocator_ >;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Benchmark settings:" );
      config.addEntry< TNL::String >( "input-file", "Input file with the graph." );
      config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-graphs.log" );
      config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
      config.addEntryEnum( "append" );
      config.addEntryEnum( "overwrite" );

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

   GraphsBenchmark( const TNL::Config::ParameterContainer& parameters_ ) : parameters( parameters_ ) {}

   template< typename Device,
             template< typename Device_, typename Index_, typename IndexAllocator_ >
             class Segments,
             template< typename Index_, typename Device_ >
             class SegmentsKernel >
   void
   TNLBenchmarks( const HostDigraph& hostDigraph,
                  const HostGraph& hostGraph,
                  TNL::Benchmarks::Benchmark<>& benchmark,
                  const TNL::String& device,
                  const TNL::String& segments )
   {
      using Matrix = TNL::Matrices::SparseMatrix< Real, Device, Index, TNL::Matrices::GeneralMatrix, Segments >;
      using Graph = TNL::Graphs::Graph< Matrix, TNL::Graphs::GraphTypes::Undirected >;
      using Digraph = TNL::Graphs::Graph< Matrix, TNL::Graphs::GraphTypes::Directed >;
      using Graph = TNL::Graphs::Graph< Matrix, TNL::Graphs::GraphTypes::Undirected >;
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;
      using RealVector = TNL::Containers::Vector< Real, Device, Index >;
      //using KernelType = SegmentsKernel< Index, Device >;

      // TODO: Find a way how to use various reduction kernels for segments in the algorithms.

      Digraph digraph( hostDigraph );
      Graph graph( hostGraph );

      // Benchmarking breadth-first search with directed graph
      IndexVector bfsDistances( digraph.getNodeCount() );
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "BFS TNL dir" ) } } ) );

      auto bfs_tnl_dir = [ & ]() mutable
      {
         TNL::Graphs::breadthFirstSearch( digraph, 0, bfsDistances );
      };
      benchmark.time< Device >( device, bfs_tnl_dir );
#ifdef HAVE_BOOST
      if( bfsDistances != this->boostBfsDistancesDirected ) {
         std::cout << "BFS distances of directed graph from Boost and TNL are not equal!" << std::endl;
         std::cout << "Boost: " << this->boostBfsDistancesDirected << std::endl;
         std::cout << "TNL:   " << bfsDistances << std::endl;
         this->errors++;
      }
#endif

      // Benchmarking breadth-first search with undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "BFS TNL undir" ) } } ) );

      auto bfs_tnl_undir = [ & ]() mutable
      {
         TNL::Graphs::breadthFirstSearch( graph, 0, bfsDistances );
      };
      benchmark.time< Device >( device, bfs_tnl_undir );
#ifdef HAVE_BOOST
      if( bfsDistances != this->boostBfsDistancesUndirected ) {
         std::cout << "BFS distances of undirected graph from Boost and TNL are not equal!" << std::endl;
         std::cout << "Boost: " << this->boostBfsDistancesUndirected << std::endl;
         std::cout << "TNL:   " << bfsDistances << std::endl;
         this->errors++;
      }
#endif

      // Benchmarking single-source shortest paths with directed graph
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< Real >() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "SSSP TNL dir" ) } } ) );

      RealVector ssspDistances( digraph.getNodeCount(), 0 );
      auto sssp_tnl_dir = [ & ]() mutable
      {
         TNL::Graphs::singleSourceShortestPath( digraph, 0, ssspDistances );
      };
      benchmark.time< Device >( device, sssp_tnl_dir );

#ifdef HAVE_BOOST
      if( ssspDistances != this->boostSSSPDistancesDirected ) {
         std::cout << "SSSP distances of directed graph from Boost and TNL are not equal!" << std::endl;
         std::cout << "Boost: " << this->boostSSSPDistancesDirected << std::endl;
         std::cout << "TNL:   " << ssspDistances << std::endl;
         this->errors++;
      }
#endif

      // Benchmarking single-source shortest paths with undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< Real >() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "SSSP TNL undir" ) } } ) );

      //RealVector ssspDistances( digraph.getNodeCount(), 0 );
      auto sssp_tnl_undir = [ & ]() mutable
      {
         TNL::Graphs::singleSourceShortestPath( graph, 0, ssspDistances );
      };
      benchmark.time< Device >( device, sssp_tnl_undir );

#ifdef HAVE_BOOST
      if( ssspDistances != this->boostSSSPDistancesUndirected ) {
         std::cout << "SSSP distances of undirected graph from Boost and TNL are not equal!" << std::endl;
         std::cout << "Boost: " << this->boostSSSPDistancesUndirected << std::endl;
         std::cout << "TNL:   " << ssspDistances << std::endl;
         this->errors++;
      }
#endif

      // Benchmarking minimum spanning tree
      Graph mstGraph;
      IndexVector roots;
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< Real >() },
                                                          { "device", device },
                                                          { "format", segments },
                                                          { "algorithm", std::string( "MST TNL undir" ) } } ) );
      auto mst_tnl = [ & ]() mutable
      {
         TNL::Graphs::minimumSpanningTree( graph, mstGraph, roots );
      };
      benchmark.time< Device >( device, mst_tnl );
      auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
      TNL::Graphs::GraphWriter< Graph >::writeEdgeList( filename + "-tnl-mst.txt", mstGraph );
      if( ! TNL::Graphs::isForest( mstGraph ) ) {
         std::cout << "ERROR: TNL MST is not a forest!" << std::endl;
         this->errors++;
      }
#ifdef HAVE_BOOST
      Real mstTotalWeight = mstGraph.getTotalWeight();
      if( mstTotalWeight != boostMSTTotalWeight ) {
         std::cout << "ERROR: Total weights of boost MST and TNL MST do not match!" << std::endl;
         std::cout << "Boost MST total weight: " << boostMSTTotalWeight << std::endl;
         std::cout << "TNL MST total weight: " << mstTotalWeight << std::endl;
         this->errors++;
      }
#endif
   }

   void
   boostBenchmarks( const HostDigraph& digraph, const HostGraph& graph, TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_BOOST
      BoostGraph< Index, Real, TNL::Graphs::GraphTypes::Directed > boostDigraph( digraph );
      BoostGraph< Index, Real, TNL::Graphs::GraphTypes::Undirected > boostGraph( graph );

      // Benchmarking breadth-first search of directed graph
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", "sequential" },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "BFS Boost dir" ) } } ) );
      std::vector< Index > boostBfsDistances( digraph.getNodeCount() );
      auto bfs_boost_dir = [ & ]() mutable
      {
         boostDigraph.breadthFirstSearch( 0, boostBfsDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", bfs_boost_dir );
      HostIndexVector boost_bfs_dist( boostBfsDistances );
      boost_bfs_dist.forAllElements(
         [] __cuda_callable__( Index i, Index & x )
         {
            x = x == std::numeric_limits< Index >::max() ? -1 : x;
         } );
      this->boostBfsDistancesDirected = boost_bfs_dist;

      // Benchmarking breadth-first search of undirected graph
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", "sequential" },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "BFS Boost undir" ) } } ) );
      auto bfs_boost_undir = [ & ]() mutable
      {
         boostGraph.breadthFirstSearch( 0, boostBfsDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", bfs_boost_undir );
      boost_bfs_dist = boostBfsDistances;
      boost_bfs_dist.forAllElements(
         [] __cuda_callable__( Index i, Index & x )
         {
            x = x == std::numeric_limits< Index >::max() ? -1 : x;
         } );
      this->boostBfsDistancesUndirected = boost_bfs_dist;

      // Benchmarking single-source shortest paths of directed graph
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< Real >() },
                                                          { "device", "sequential" },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "SSSP Boost dir" ) } } ) );
      std::vector< Real > boostSSSPDistances( digraph.getNodeCount() );
      auto sssp_boost_dir = [ & ]() mutable
      {
         boostDigraph.singleSourceShortestPath( 0, boostSSSPDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", sssp_boost_dir );
      HostRealVector boost_sssp_dist( boostSSSPDistances );
      boost_sssp_dist.forAllElements(
         [] __cuda_callable__( Index i, Real & x )
         {
            x = x == std::numeric_limits< Real >::max() ? -1 : x;
         } );
      this->boostSSSPDistancesDirected = boost_sssp_dist;

      // Benchmarking single-source shortest paths of undirected graph
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< Real >() },
                                                          { "device", "sequential" },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "SSSP Boost undir" ) } } ) );
      auto sssp_boost_undir = [ & ]() mutable
      {
         boostGraph.singleSourceShortestPath( 0, boostSSSPDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", sssp_boost_undir );
      boost_sssp_dist = boostSSSPDistances;
      boost_sssp_dist.forAllElements(
         [] __cuda_callable__( Index i, Real & x )
         {
            x = x == std::numeric_limits< Real >::max() ? -1 : x;
         } );
      this->boostSSSPDistancesUndirected = boost_sssp_dist;

      // Benchmarking minimum spanning tree
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "precision", TNL::getType< Real >() },
                                                          { "device", "sequential" },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "MST Boost" ) } } ) );
      using BoostEdge = typename BoostGraph< Index, Real, TNL::Graphs::GraphTypes::Undirected >::Edge;
      std::vector< BoostEdge > boostMstEdges;
      auto mst_boost = [ & ]() mutable
      {
         boostGraph.minimumSpanningTree( boostMstEdges );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", mst_boost );
      this->boostMSTTotalWeight = 0.0;
      for( auto& edge : boostMstEdges ) {
         Real weight = boost::get( boost::edge_weight, boostGraph.getGraph(), edge );
         this->boostMSTTotalWeight += weight;
      }
      //this->boostMSTTotalWeight = 2.0;
      auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
      boostGraph.exportMst( boostMstEdges, filename + "-boost-mst.txt" );
#endif
   }

   void
   gunrockBenchmarks( const HostDigraph& hostDigraph, const HostGraph& hostGraph, TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_GUNROCK
      auto filename = this->parameters.getParameter< TNL::String >( "input-file" );
      std::vector< IndexType > digraph_row_offsets, graph_row_offsets;
      std::vector< IndexType > digraph_column_indices, graph_column_indices;
      std::vector< RealType > digraph_values, graph_values;

      const auto& digraphAdjacencyMatrix = hostDigraph.getAdjacencyMatrix();
      TNL::copy( digraphAdjacencyMatrix.getSegments().getOffsets(), digraph_row_offsets );
      TNL::copy( digraphAdjacencyMatrix.getColumnIndexes(), digraph_column_indices );
      TNL::copy( digraphAdjacencyMatrix.getValues(), digraph_values );
      thrust::device_vector< IndexType > d_digraph_row_offsets( digraphAdjacencyMatrix.getRows() + 1 );
      thrust::device_vector< IndexType > d_digraph_column_indices( digraphAdjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< RealType > d_digraph_values( digraphAdjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< IndexType > d_digraph_row_indices( digraphAdjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< IndexType > d_digraph_column_offsets( digraphAdjacencyMatrix.getColumns() + 1 );
      thrust::copy( digraph_row_offsets.begin(), digraph_row_offsets.end(), d_digraph_row_offsets.begin() );
      thrust::copy( digraph_column_indices.begin(), digraph_column_indices.end(), d_digraph_column_indices.begin() );
      thrust::copy( digraph_values.begin(), digraph_values.end(), d_digraph_values.begin() );

      auto digraph = gunrock::graph::build::from_csr< gunrock::memory_space_t::device, gunrock::graph::view_t::csr >(
         digraphAdjacencyMatrix.getRows(),              // rows
         digraphAdjacencyMatrix.getColumns(),           // columns
         digraphAdjacencyMatrix.getValues().getSize(),  // nonzeros
         d_digraph_row_offsets.data().get(),            // row_offsets
         d_digraph_column_indices.data().get(),         // column_indices
         d_digraph_values.data().get(),                 // values
         d_digraph_row_indices.data().get(),            // row_indices
         d_digraph_column_offsets.data().get()          // column_offsets
      );

      const auto& graphAdjacencyMatrix = hostGraph.getAdjacencyMatrix();
      TNL::copy( graphAdjacencyMatrix.getSegments().getOffsets(), graph_row_offsets );
      TNL::copy( graphAdjacencyMatrix.getColumnIndexes(), graph_column_indices );
      TNL::copy( graphAdjacencyMatrix.getValues(), graph_values );
      thrust::device_vector< IndexType > d_graph_row_offsets( graphAdjacencyMatrix.getRows() + 1 );
      thrust::device_vector< IndexType > d_graph_column_indices( graphAdjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< RealType > d_graph_values( graphAdjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< IndexType > d_graph_row_indices( graphAdjacencyMatrix.getNonzeroElementsCount() );
      thrust::device_vector< IndexType > d_graph_column_offsets( graphAdjacencyMatrix.getColumns() + 1 );
      thrust::copy( graph_row_offsets.begin(), graph_row_offsets.end(), d_graph_row_offsets.begin() );
      thrust::copy( graph_column_indices.begin(), graph_column_indices.end(), d_graph_column_indices.begin() );
      thrust::copy( graph_values.begin(), graph_values.end(), d_graph_values.begin() );

      auto graph = gunrock::graph::build::from_csr< gunrock::memory_space_t::device, gunrock::graph::view_t::csr >(
         graphAdjacencyMatrix.getRows(),              // rows
         graphAdjacencyMatrix.getColumns(),           // columns
         graphAdjacencyMatrix.getValues().getSize(),  // nonzeros
         d_graph_row_offsets.data().get(),            // row_offsets
         d_graph_column_indices.data().get(),         // column_indices
         d_graph_values.data().get(),                 // values
         d_graph_row_indices.data().get(),            // row_indices
         d_graph_column_offsets.data().get()          // column_offsets
      );

      GunrockBenchmark< Real, Index > gunrockBenchmark;
      Index start = 0;

      // Benchmarking breadth-first search of directed graph
      benchmark.setDatasetSize( digraphAdjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", std::string( "GPU" ) },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "BFS Gunrock dir" ) } } ) );
      std::vector< Index > bfsDistances( digraphAdjacencyMatrix.getRows() );
      gunrockBenchmark.breadthFirstSearch( benchmark, digraph, start, digraphAdjacencyMatrix.getRows(), bfsDistances );
      HostIndexVector gunrock_bfs_dist( bfsDistances );
      gunrock_bfs_dist.forAllElements(
         [] __cuda_callable__( Index i, Index & x )
         {
            x = x == std::numeric_limits< Index >::max() ? -1 : x;
         } );
      this->gunrockBfsDistancesDirected = gunrock_bfs_dist;

      if( this->boostBfsDistancesDirected != this->gunrockBfsDistancesDirected ) {
         std::cout << "BFS distances of directed graph from Boost and Gunrock are not equal!" << std::endl;
         std::cout << "Boost:   " << this->boostBfsDistancesDirected << std::endl;
         std::cout << "Gunrock: " << this->gunrockBfsDistancesDirected << std::endl;
         this->errors++;
      }

      // Benchmarking breadth-first search of undirected graph
      benchmark.setDatasetSize( graphAdjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", std::string( "GPU" ) },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "BFS Gunrock undir" ) } } ) );
      gunrockBenchmark.breadthFirstSearch( benchmark, graph, start, graphAdjacencyMatrix.getRows(), bfsDistances );
      gunrock_bfs_dist = bfsDistances;
      gunrock_bfs_dist.forAllElements(
         [] __cuda_callable__( Index i, Index & x )
         {
            x = x == std::numeric_limits< Index >::max() ? -1 : x;
         } );
      this->gunrockBfsDistancesUndirected = gunrock_bfs_dist;

      if( this->boostBfsDistancesUndirected != this->gunrockBfsDistancesUndirected ) {
         std::cout << "BFS distances of undirected graph from Boost and Gunrock are not equal!" << std::endl;
         std::cout << "Boost:   " << this->boostBfsDistancesUndirected << std::endl;
         std::cout << "Gunrock: " << this->gunrockBfsDistancesUndirected << std::endl;
         this->errors++;
      }

      // Benchmarking single-source shortest path of directed graph
      benchmark.setDatasetSize( digraphAdjacencyMatrix.getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", std::string( "GPU" ) },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "SSSP Gunrock dir" ) } } ) );
      std::vector< Real > ssspDistances( digraphAdjacencyMatrix.getRows() );
      gunrockBenchmark.singleSourceShortestPath( benchmark, digraph, start, digraphAdjacencyMatrix.getRows(), ssspDistances );
      HostRealVector gunrock_sssp_dist( ssspDistances );
      gunrock_sssp_dist.forAllElements(
         [] __cuda_callable__( Index i, Real & x )
         {
            x = x == std::numeric_limits< Real >::max() ? -1 : x;
         } );
      this->gunrockSSSPDistancesDirected = gunrock_sssp_dist;

      if( this->boostSSSPDistancesDirected != this->gunrockSSSPDistancesDirected ) {
         std::cout << "SSSP distances of directed graph from Boost and Gunrock are not equal!" << std::endl;
         std::cout << "Boost:   " << this->boostSSSPDistancesDirected << std::endl;
         std::cout << "Gunrock: " << this->gunrockSSSPDistancesDirected << std::endl;
         this->errors++;
      }

      // Benchmarking single-source shortest path of undirected graph
      benchmark.setDatasetSize( graphAdjacencyMatrix.getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataColumns(
         TNL::Benchmarks::Benchmark<>::MetadataColumns( { { "index type", TNL::getType< Index >() },
                                                          { "device", std::string( "GPU" ) },
                                                          { "format", "N/A" },
                                                          { "algorithm", std::string( "SSSP Gunrock undir" ) } } ) );
      gunrockBenchmark.singleSourceShortestPath( benchmark, graph, start, graphAdjacencyMatrix.getRows(), ssspDistances );
      gunrock_sssp_dist = ssspDistances;
      gunrock_sssp_dist.forAllElements(
         [] __cuda_callable__( Index i, Real & x )
         {
            x = x == std::numeric_limits< Real >::max() ? -1 : x;
         } );
      this->gunrockSSSPDistancesUndirected = gunrock_sssp_dist;

      if( this->boostSSSPDistancesUndirected != this->gunrockSSSPDistancesUndirected ) {
         std::cout << "SSSP distances of undirected graph from Boost and Gunrock are not equal!" << std::endl;
         std::cout << "Boost:   " << this->boostSSSPDistancesUndirected << std::endl;
         std::cout << "Gunrock: " << this->gunrockSSSPDistancesUndirected << std::endl;
         this->errors++;
      }
#endif
   }

   bool
   runBenchmark()
   {
      auto inputFile = parameters.getParameter< TNL::String >( "input-file" );
      const TNL::String logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const TNL::String outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const int loops = parameters.getParameter< int >( "loops" );
      const int verbose = parameters.getParameter< int >( "verbose" );

      auto mode = std::ios::out;
      if( outputMode == "append" )
         mode |= std::ios::app;
      std::ofstream logFile( logFileName.getString(), mode );
      TNL::Benchmarks::Benchmark<> benchmark( logFile, loops, verbose );

      // write global metadata into a separate file
      std::map< std::string, std::string > metadata = TNL::Benchmarks::getHardwareMetadata();
      TNL::Benchmarks::writeMapAsJson( metadata, logFileName, ".metadata.json" );

      this->errors = 0;

      TNL::String device = parameters.getParameter< TNL::String >( "device" );

      std::cout << "Graphs benchmark  with " << TNL::getType< Real >() << " precision and device: " << device << std::endl;

      HostDigraph digraph;
      std::cout << "Reading graph from file " << inputFile << std::endl;
      TNL::Graphs::GraphReader< HostDigraph >::readEdgeList( inputFile, digraph );

      HostMatrix symmetrizedAdjacencyMatrix = TNL::Matrices::getSymmetricPart< HostMatrix >( digraph.getAdjacencyMatrix() );
      HostGraph graph( symmetrizedAdjacencyMatrix );
      TNL::Graphs::GraphWriter< HostGraph >::writeEdgeList( inputFile + "-undirected.txt", graph );

      boostBenchmarks( digraph, graph, benchmark );
      gunrockBenchmarks( digraph, graph, benchmark );

      if( device == "sequential" || device == "all" )
         TNLBenchmarks< TNL::Devices::Sequential, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            digraph, graph, benchmark, "sequential", "CSRScalar" );
      if( device == "host" || device == "all" )
         TNLBenchmarks< TNL::Devices::Host, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            digraph, graph, benchmark, "host", "CSRScalar" );
#ifdef __CUDACC__
      if( device == "cuda" || device == "all" ) {
         TNLBenchmarks< TNL::Devices::Cuda, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            digraph, graph, benchmark, "cuda", "CSRScalar" );
         TNLBenchmarks< TNL::Devices::Cuda, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRVectorKernel >(
            digraph, graph, benchmark, "cuda", "CSRVector" );
         TNLBenchmarks< TNL::Devices::Cuda, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRLightKernel >(
            digraph, graph, benchmark, "cuda", "CSRLight" );
         TNLBenchmarks< TNL::Devices::Cuda, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRAdaptiveKernel >(
            digraph, graph, benchmark, "cuda", "CSRAdaptive" );
         TNLBenchmarks< TNL::Devices::Cuda, EllpackSegments, TNL::Algorithms::SegmentsReductionKernels::EllpackKernel >(
            digraph, graph, benchmark, "cuda", "Ellpack" );
         TNLBenchmarks< TNL::Devices::Cuda,
                        SlicedEllpackSegments,
                        TNL::Algorithms::SegmentsReductionKernels::SlicedEllpackKernel >(
            digraph, graph, benchmark, "cuda", "SlicedEllpack" );
         TNLBenchmarks< TNL::Devices::Cuda, BiEllpackSegments, TNL::Algorithms::SegmentsReductionKernels::BiEllpackKernel >(
            digraph, graph, benchmark, "cuda", "BiEllpack" );
      }
#endif
      if( ! errors )
         return true;
      return false;
   }

protected:
   const TNL::Config::ParameterContainer& parameters;

   // These vectors serve as a reference solution for comparison with TNL
   HostIndexVector boostBfsDistancesDirected, boostBfsDistancesUndirected;
   HostRealVector boostSSSPDistancesDirected, boostSSSPDistancesUndirected;

   HostIndexVector gunrockBfsDistancesDirected, gunrockBfsDistancesUndirected;
   HostRealVector gunrockSSSPDistancesDirected, gunrockSSSPDistancesUndirected;

   Real boostMSTTotalWeight;

   int errors;
};

}  // namespace TNL::Benchmarks::Graphs
