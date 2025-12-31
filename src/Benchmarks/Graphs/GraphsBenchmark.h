// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Graphs/GraphOperations.h>
#include <TNL/Graphs/Readers/EdgeListReader.h>
#include <TNL/Graphs/Readers/MtxReader.h>
#include <TNL/Graphs/Writers/EdgeListWriter.h>
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
#include <TNL/Algorithms/SegmentsReductionKernels/ChunkedEllpackKernel.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MatrixOperations.h>
#include "BoostGraph.h"
#include "GunrockBenchmark.h"
#include "LaunchConfigurationsSetup.h"

namespace TNL::Benchmarks::Graphs {

template< typename Real = double, typename Index = int >
struct GraphsBenchmark
{
   using RealType = Real;
   using IndexType = Index;
   using HostMatrix = TNL::Matrices::SparseMatrix< Real, TNL::Devices::Host, Index >;
   using HostGraph = TNL::Graphs::Graph< Real, TNL::Devices::Host, Index, TNL::Graphs::UndirectedGraph >;
   using HostDigraph = TNL::Graphs::Graph< Real, TNL::Devices::Host, Index, TNL::Graphs::DirectedGraph >;
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
   template< typename Device_, typename Index_, typename IndexAllocator_ >
   using ChunkedEllpackSegments = TNL::Algorithms::Segments::ChunkedEllpack< Device_, Index_, IndexAllocator_ >;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      config.addDelimiter( "Benchmark settings:" );
      config.addEntry< TNL::String >( "input-file", "Input file with the graph." );
      config.addEntry< TNL::String >( "log-file", "Log file name.", "tnl-benchmark-graphs.log" );
      config.addEntry< TNL::String >( "output-mode", "Mode for opening the log file.", "overwrite" );
      config.addEntryEnum( "append" );
      config.addEntryEnum( "overwrite" );
      config.addEntry< bool >( "with-bfs", "Run breadth-first search benchmark.", true );
      config.addEntry< bool >( "with-sssp", "Run single-source shortest paths benchmark.", true );
      config.addEntry< bool >( "with-mst", "Run minimum spanning tree benchmark.", true );

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

   GraphsBenchmark( const TNL::Config::ParameterContainer& parameters_ )
   : parameters( parameters_ )
   {}

   void
   boostBenchmarks( const HostDigraph& digraph,
                    const HostGraph& graph,
                    IndexType smallestNode,
                    IndexType largestNode,
                    TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_BOOST
      BoostGraph< Index, Real, TNL::Graphs::DirectedGraph > boostDigraph( digraph );
      BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph > boostGraph( graph );
      benchmark.setMetadataElement( { "solver", "Boost" } );

      if( this->withBfs ) {
         // Benchmarking breadth-first search of directed graph
         benchmark.setMetadataElement( { "problem", "BFS dir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         std::vector< Index > boostBfsDistances( digraph.getVertexCount() );
         auto bfs_boost_dir = [ & ]() mutable
         {
            boostDigraph.breadthFirstSearch( largestNode, boostBfsDistances );
         };
         benchmark.time< TNL::Devices::Sequential >( "sequential", bfs_boost_dir );
         HostIndexVector boost_bfs_dist( boostBfsDistances );
         boost_bfs_dist.forAllElements(
            [] __cuda_callable__( Index i, Index & x )
            {
               x = x == std::numeric_limits< Index >::max() ? -1 : x;
            } );
         this->boostBfsDistancesDirected.setSize( boost_bfs_dist.getSize() );
         for( Index i = 0; i < boost_bfs_dist.getSize(); i++ )
            if( boost_bfs_dist[ i ] == 0 && i != largestNode )
               this->boostBfsDistancesDirected[ i ] = -1;
            else
               this->boostBfsDistancesDirected[ i ] = boost_bfs_dist[ i ];

         // Benchmarking breadth-first search of undirected graph
         benchmark.setMetadataElement( { "problem", "BFS undir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         auto bfs_boost_undir = [ & ]() mutable
         {
            boostGraph.breadthFirstSearch( largestNode, boostBfsDistances );
         };
         benchmark.time< TNL::Devices::Sequential >( "sequential", bfs_boost_undir );
         boost_bfs_dist = boostBfsDistances;
         boost_bfs_dist.forAllElements(
            [] __cuda_callable__( Index i, Index & x )
            {
               x = x == std::numeric_limits< Index >::max() ? -1 : x;
            } );
         this->boostBfsDistancesUndirected.setSize( boost_bfs_dist.getSize() );
         for( Index i = 0; i < boost_bfs_dist.getSize(); i++ )
            if( boost_bfs_dist[ i ] == 0 && i != largestNode )
               this->boostBfsDistancesUndirected[ i ] = -1;
            else
               this->boostBfsDistancesUndirected[ i ] = boost_bfs_dist[ i ];
      }

      if( this->withSssp ) {
         // Benchmarking single-source shortest paths of directed graph
         benchmark.setMetadataElement( { "problem", "SSSP dir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         std::vector< Real > boostSSSPDistances( digraph.getVertexCount() );
         auto sssp_boost_dir = [ & ]() mutable
         {
            boostDigraph.singleSourceShortestPath( largestNode, boostSSSPDistances );
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
         benchmark.setMetadataElement( { "problem", "SSSP undir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         auto sssp_boost_undir = [ & ]() mutable
         {
            boostGraph.singleSourceShortestPath( largestNode, boostSSSPDistances );
         };
         benchmark.time< TNL::Devices::Sequential >( "sequential", sssp_boost_undir );
         boost_sssp_dist = boostSSSPDistances;
         boost_sssp_dist.forAllElements(
            [] __cuda_callable__( Index i, Real & x )
            {
               x = x == std::numeric_limits< Real >::max() ? -1 : x;
            } );
         this->boostSSSPDistancesUndirected = boost_sssp_dist;
      }

      if( this->withMst ) {
         // Benchmarking minimum spanning tree
         benchmark.setMetadataElement( { "problem", "MST undir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         using BoostEdge = typename BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph >::Edge;
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
         auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
         boostGraph.exportMst( boostMstEdges, filename + "-boost-mst.txt" );
      }
#endif  // HAVE_BOOST
   }

   void
   gunrockBenchmarks( const HostDigraph& hostDigraph,
                      const HostGraph& hostGraph,
                      IndexType smallestNode,
                      IndexType largestNode,
                      TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_GUNROCK
      auto filename = this->parameters.getParameter< TNL::String >( "input-file" );
      std::vector< IndexType > digraph_row_offsets, graph_row_offsets;
      std::vector< IndexType > digraph_column_indices, graph_column_indices;
      std::vector< RealType > digraph_values, graph_values;

      const auto& digraphAdjacencyMatrix = hostDigraph.getAdjacencyMatrix();
      TNL::Algorithms::copy( digraph_row_offsets, digraphAdjacencyMatrix.getSegments().getOffsets() );
      TNL::Algorithms::copy( digraph_column_indices, digraphAdjacencyMatrix.getColumnIndexes() );
      TNL::Algorithms::copy( digraph_values, digraphAdjacencyMatrix.getValues() );
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
      TNL::Algorithms::copy( graph_row_offsets, graphAdjacencyMatrix.getSegments().getOffsets() );
      TNL::Algorithms::copy( graph_column_indices, graphAdjacencyMatrix.getColumnIndexes() );
      TNL::Algorithms::copy( graph_values, graphAdjacencyMatrix.getValues() );
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
      benchmark.setMetadataElement( { "solver", "Gunrock" } );

      if( this->withBfs ) {
         // Benchmarking breadth-first search of directed graph
         benchmark.setDatasetSize( digraphAdjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS dir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );

         std::vector< Index > bfsDistances( digraphAdjacencyMatrix.getRows() );
         gunrockBenchmark.breadthFirstSearch( benchmark, digraph, largestNode, digraphAdjacencyMatrix.getRows(), bfsDistances );
         HostIndexVector gunrock_bfs_dist( bfsDistances );
         gunrock_bfs_dist.forAllElements(
            [] __cuda_callable__( Index i, Index & x )
            {
               x = x == std::numeric_limits< Index >::max() ? -1 : x;
            } );
         this->gunrockBfsDistancesDirected = gunrock_bfs_dist;

   #ifdef HAVE_BOOST
         if( this->boostBfsDistancesDirected != this->gunrockBfsDistancesDirected ) {
            std::cout << "BFS distances of directed graph from Boost and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
   #endif

         // Benchmarking breadth-first search of undirected graph
         benchmark.setDatasetSize( graphAdjacencyMatrix.getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS undir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         gunrockBenchmark.breadthFirstSearch( benchmark, graph, largestNode, graphAdjacencyMatrix.getRows(), bfsDistances );
         gunrock_bfs_dist = bfsDistances;
         gunrock_bfs_dist.forAllElements(
            [] __cuda_callable__( Index i, Index & x )
            {
               x = x == std::numeric_limits< Index >::max() ? -1 : x;
            } );
         this->gunrockBfsDistancesUndirected = gunrock_bfs_dist;

   #ifdef HAVE_BOOST
         if( this->boostBfsDistancesUndirected != this->gunrockBfsDistancesUndirected ) {
            std::cout << "BFS distances of undirected graph from Boost and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
   #endif
      }

      if( this->withSssp ) {
         // Benchmarking single-source shortest path of directed graph
         benchmark.setDatasetSize( digraphAdjacencyMatrix.getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
         benchmark.setMetadataElement( { "problem", "SSSP dir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         std::vector< Real > ssspDistances( digraphAdjacencyMatrix.getRows() );
         gunrockBenchmark.singleSourceShortestPath(
            benchmark, digraph, largestNode, digraphAdjacencyMatrix.getRows(), ssspDistances );
         HostRealVector gunrock_sssp_dist( ssspDistances );
         gunrock_sssp_dist.forAllElements(
            [] __cuda_callable__( Index i, Real & x )
            {
               x = x == std::numeric_limits< Real >::max() ? -1 : x;
            } );
         this->gunrockSSSPDistancesDirected = gunrock_sssp_dist;

   #ifdef HAVE_BOOST
         if( this->boostSSSPDistancesDirected != this->gunrockSSSPDistancesDirected ) {
            std::cout << "SSSP distances of directed graph from Boost and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
   #endif

         // Benchmarking single-source shortest path of undirected graph
         benchmark.setDatasetSize( graphAdjacencyMatrix.getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
         benchmark.setMetadataElement( { "problem", "SSSP undir" } );
         benchmark.setMetadataElement( { "format", "N/A" } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         gunrockBenchmark.singleSourceShortestPath(
            benchmark, graph, largestNode, graphAdjacencyMatrix.getRows(), ssspDistances );
         gunrock_sssp_dist = ssspDistances;
         gunrock_sssp_dist.forAllElements(
            [] __cuda_callable__( Index i, Real & x )
            {
               x = x == std::numeric_limits< Real >::max() ? -1 : x;
            } );
         this->gunrockSSSPDistancesUndirected = gunrock_sssp_dist;

   #ifdef HAVE_BOOST
         if( this->boostSSSPDistancesUndirected != this->gunrockSSSPDistancesUndirected ) {
            std::cout << "SSSP distances of undirected graph from Boost and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
   #endif
      }
#endif  // HAVE_GUNROCK
   }

   template< typename Device,
             template< typename Device_, typename Index_, typename IndexAllocator_ > class Segments,
             template< typename Index_, typename Device_ > class SegmentsKernel >
   void
   TNLBenchmarks( const HostDigraph& hostDigraph,
                  const HostGraph& hostGraph,
                  IndexType smallestNode,
                  IndexType largestNode,
                  TNL::Benchmarks::Benchmark<>& benchmark,
                  const TNL::String& device,
                  const TNL::String& segments )
   {
      using Matrix = TNL::Matrices::SparseMatrix< Real, Device, Index, TNL::Matrices::GeneralMatrix, Segments >;
      using Graph = TNL::Graphs::Graph< Real, Device, Index, TNL::Graphs::UndirectedGraph >;
      using Digraph = TNL::Graphs::Graph< Real, Device, Index, TNL::Graphs::DirectedGraph >;
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;
      using RealVector = TNL::Containers::Vector< Real, Device, Index >;
      using SegmentsType = typename Matrix::SegmentsType;
      //using KernelType = SegmentsKernel< Index, Device >;

      // TODO: Find a way how to use various reduction kernels for segments in the algorithms.

      Digraph digraph( hostDigraph );
      Graph graph( hostGraph );
      benchmark.setMetadataElement( { "solver", "TNL" } );

      if( this->withBfs ) {
         // Benchmarking breadth-first search with directed graph
         IndexVector bfsDistances( digraph.getVertexCount() );
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS dir" } );
         benchmark.setMetadataElement( { "format", segments } );

         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );
            auto bfs_tnl_dir = [ & ]() mutable
            {
               TNL::Graphs::breadthFirstSearch( digraph, largestNode, bfsDistances, launchConfig );
            };
            benchmark.time< Device >( device, bfs_tnl_dir );
#ifdef HAVE_BOOST
            if( bfsDistances != this->boostBfsDistancesDirected ) {
               std::cout << "BFS distances of directed graph from Boost and TNL are not equal!" << std::endl;
               this->errors++;
               //for( Index i = 0; i < digraph.getVertexCount(); i++ )
               //   if( bfsDistances.getElement( i ) != this->boostBfsDistancesDirected[ i ] )
               //      std::cerr << "i = " << i << " TNL -> " << bfsDistances.getElement( i ) << " Boost -> "
               //                << this->boostBfsDistancesDirected[ i ] << std::endl;
            }
#endif
#ifdef HAVE_GUNROCK
            if( bfsDistances != this->gunrockBfsDistancesDirected ) {
               std::cout << "BFS distances of directed graph from TNL and Gunrock are not equal!" << std::endl;
               this->errors++;
            }
#endif
         }
         // Benchmarking breadth-first search with undirected graph
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS undir" } );
         benchmark.setMetadataElement( { "format", segments } );

         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );

            auto bfs_tnl_undir = [ & ]() mutable
            {
               TNL::Graphs::breadthFirstSearch( graph, largestNode, bfsDistances, launchConfig );
            };
            benchmark.time< Device >( device, bfs_tnl_undir );
#ifdef HAVE_BOOST
            if( bfsDistances != this->boostBfsDistancesUndirected ) {
               std::cout << "BFS distances of undirected graph from Boost and TNL are not equal!" << std::endl;
               this->errors++;
               //for( Index i = 0; i < digraph.getVertexCount(); i++ )
               //   if( bfsDistances.getElement( i ) != this->boostBfsDistancesUndirected[ i ] )
               //      std::cerr << "i = " << i << " TNL -> " << bfsDistances.getElement( i ) << " Boost -> "
               //                << this->boostBfsDistancesUndirected[ i ] << std::endl;
            }
#endif
#ifdef HAVE_GUNROCK
            if( bfsDistances != this->gunrockBfsDistancesUndirected ) {
               std::cout << "BFS distances of undirected graph from TNL and Gunrock are not equal!" << std::endl;
               this->errors++;
            }
#endif
         }
      }

      if( this->withSssp ) {
         // Benchmarking single-source shortest paths with directed graph
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount()
                                   * ( sizeof( Index ) + sizeof( Real ) ) );
         benchmark.setMetadataElement( { "problem", "SSSP dir" } );
         benchmark.setMetadataElement( { "format", segments } );

         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );

            RealVector ssspDistances( digraph.getVertexCount(), 0 );
            auto sssp_tnl_dir = [ & ]() mutable
            {
               TNL::Graphs::singleSourceShortestPath( digraph, largestNode, ssspDistances, launchConfig );
            };
            if( min( digraph.getAdjacencyMatrix().getValues() ) < 0 ) {
               std::cout << "ERROR: Negative weights in the graph! Skipping SSSP benchmark." << std::endl;
               this->errors++;
            }
            else
               benchmark.time< Device >( device, sssp_tnl_dir );

#ifdef HAVE_BOOST
            if( ssspDistances != this->boostSSSPDistancesDirected ) {
               std::cout << "SSSP distances of directed graph from Boost and TNL are not equal!" << std::endl;
               this->errors++;
            }
#endif
#ifdef HAVE_GUNROCK
            if( ssspDistances != this->gunrockSSSPDistancesDirected ) {
               std::cout << "SSSP distances of directed graph from TNL and Gunrock are not equal!" << std::endl;
               this->errors++;
            }
#endif
         }

         // Benchmarking single-source shortest paths with undirected graph
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount()
                                   * ( sizeof( Index ) + sizeof( Real ) ) );
         benchmark.setMetadataElement( { "problem", "SSSP undir" } );
         benchmark.setMetadataElement( { "format", segments } );

         for( auto [ launchConfig, tag ] : LaunchConfigurationsSetup< SegmentsType >::create() ) {
            benchmark.setMetadataElement( { "threads mapping", tag } );

            RealVector ssspDistances( digraph.getVertexCount(), 0 );
            auto sssp_tnl_undir = [ & ]() mutable
            {
               TNL::Graphs::singleSourceShortestPath( graph, largestNode, ssspDistances, launchConfig );
            };
            benchmark.time< Device >( device, sssp_tnl_undir );
#ifdef HAVE_BOOST
            if( ssspDistances != this->boostSSSPDistancesUndirected ) {
               std::cout << "SSSP distances of undirected graph from Boost and TNL are not equal!" << std::endl;
               this->errors++;
            }
#endif
#ifdef HAVE_GUNROCK
            if( ssspDistances != this->gunrockSSSPDistancesUndirected ) {
               std::cout << "SSSP distances of undirected graph from TNL and Gunrock are not equal!" << std::endl;
               this->errors++;
            }
#endif
         }
      }

      if( this->withMst ) {
         // Benchmarking minimum spanning tree
         Graph mstGraph;
         IndexVector roots;
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount()
                                   * ( sizeof( Index ) + sizeof( Real ) ) );
         benchmark.setMetadataElement( { "problem", "MST undir" } );
         benchmark.setMetadataElement( { "format", segments } );

         auto mst_tnl = [ & ]() mutable
         {
            TNL::Graphs::minimumSpanningTree( graph, mstGraph, roots );
         };
         benchmark.time< Device >( device, mst_tnl );
         auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
         TNL::Graphs::Writers::EdgeListWriter< Graph >::write( filename + "-tnl-mst.txt", mstGraph );
         if( ! TNL::Graphs::isForest( mstGraph ) ) {
            std::cout << "ERROR: TNL MST is not a forest!" << std::endl;
            this->errors++;
         }
#ifdef HAVE_BOOST
         Real mstTotalWeight = TNL::Graphs::getTotalWeight( mstGraph );
         if( mstTotalWeight != boostMSTTotalWeight ) {
            std::cout << "ERROR: Total weights of boost MST and TNL MST do not match!" << std::endl;
            std::cout << "Boost MST total weight: " << boostMSTTotalWeight << std::endl;
            std::cout << "TNL MST total weight: " << mstTotalWeight << std::endl;
            this->errors++;
         }
#endif
      }
   }

   bool
   runBenchmark()
   {
      auto inputFile = parameters.getParameter< TNL::String >( "input-file" );
      const auto logFileName = parameters.getParameter< TNL::String >( "log-file" );
      const auto outputMode = parameters.getParameter< TNL::String >( "output-mode" );
      const int loops = parameters.getParameter< int >( "loops" );
      const int verbose = parameters.getParameter< int >( "verbose" );
      this->withBfs = parameters.getParameter< bool >( "with-bfs" );
      this->withSssp = parameters.getParameter< bool >( "with-sssp" );
      this->withMst = parameters.getParameter< bool >( "with-mst" );

      size_t dotPosition = inputFile.find_last_of( '.' );
      std::string inputFileExtension = "";
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

      std::cout << "Graphs benchmark  with " << TNL::getType< Real >() << " precision and device: " << device << '\n';

      HostDigraph digraph;
      std::cout << "Reading graph from file " << inputFile << std::endl;
      if( inputFileExtension == "mtx" )
         TNL::Graphs::Readers::MtxReader< HostDigraph >::read( inputFile, digraph );
      else
         TNL::Graphs::Readers::EdgeListReader< HostDigraph >::read( inputFile, digraph );
      // Make all weights positive because of benchmarking SSSP
      digraph.getAdjacencyMatrix().getValues() = abs( digraph.getAdjacencyMatrix().getValues() );

      auto symmetrizedAdjacencyMatrix = TNL::Matrices::getSymmetricPart< HostMatrix >( digraph.getAdjacencyMatrix() );
      HostGraph graph( symmetrizedAdjacencyMatrix );
      //TNL::Graphs::Writers::EdgeListWriter< HostGraph >::write( inputFile + "-undirected.txt", graph );
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

      boostBenchmarks( digraph, graph, smallest, largest, benchmark );
      gunrockBenchmarks( digraph, graph, smallest, largest, benchmark );

      if( device == "sequential" || device == "all" )
         TNLBenchmarks< TNL::Devices::Sequential, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            digraph, graph, smallest, largest, benchmark, "sequential", "CSR" );
      if( device == "host" || device == "all" )
         TNLBenchmarks< TNL::Devices::Host, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            digraph, graph, smallest, largest, benchmark, "host", "CSR" );
#ifdef __CUDACC__
      if( device == "cuda" || device == "all" ) {
         TNLBenchmarks< TNL::Devices::Cuda, CSRSegments, TNL::Algorithms::SegmentsReductionKernels::CSRScalarKernel >(
            digraph, graph, smallest, largest, benchmark, "cuda", "CSR" );
         TNLBenchmarks< TNL::Devices::Cuda, EllpackSegments, TNL::Algorithms::SegmentsReductionKernels::EllpackKernel >(
            digraph, graph, smallest, largest, benchmark, "cuda", "Ellpack" );
         TNLBenchmarks< TNL::Devices::Cuda,
                        SlicedEllpackSegments,
                        TNL::Algorithms::SegmentsReductionKernels::SlicedEllpackKernel >(
            digraph, graph, smallest, largest, benchmark, "cuda", "SlicedEllpack" );
         TNLBenchmarks< TNL::Devices::Cuda, BiEllpackSegments, TNL::Algorithms::SegmentsReductionKernels::BiEllpackKernel >(
            digraph, graph, smallest, largest, benchmark, "cuda", "BiEllpack" );
         TNLBenchmarks< TNL::Devices::Cuda,
                        ChunkedEllpackSegments,
                        TNL::Algorithms::SegmentsReductionKernels::ChunkedEllpackKernel >(
            digraph, graph, smallest, largest, benchmark, "cuda", "ChunkedEllpack" );
      }
#endif
      if( errors == 0 )
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

   bool withBfs, withSssp, withMst;
};

}  // namespace TNL::Benchmarks::Graphs
