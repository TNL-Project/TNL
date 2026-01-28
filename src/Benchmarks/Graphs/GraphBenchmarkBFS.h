// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBenchmarkBase.h"
#include "BoostGraph.h"
#include "GunrockBenchmark.h"
#include <TNL/Graphs/Algorithms/breadthFirstSearch.h>
#include "SemiringBFS.h"

namespace TNL::Benchmarks::Graphs {

template< typename Real = double, typename Index = int >
struct GraphBenchmarkBFS : public GraphBenchmarkBase< Real, Index, GraphBenchmarkBFS< Real, Index > >
{
   using Base = GraphBenchmarkBase< Real, Index, GraphBenchmarkBFS< Real, Index > >;
   using typename Base::HostDigraph;
   using typename Base::HostGraph;
   using typename Base::HostIndexVector;
   using typename Base::HostRealVector;
   using typename Base::IndexType;
   using typename Base::RealType;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      Base::configSetup( config );
      config.addEntry< bool >( "with-semirings", "Run semiring-based BFS and SSSP benchmarks.", true );
      config.addEntry< bool >( "with-gunrock", "Run Gunrock benchmarks.", true );
      config.addEntry< bool >( "with-boost", "Run Boost benchmarks.", true );
   }

   GraphBenchmarkBFS( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {
      withBoost = this->parameters.template getParameter< bool >( "with-boost" );
      withGunrock = this->parameters.template getParameter< bool >( "with-gunrock" );
      withSemirings = this->parameters.template getParameter< bool >( "with-semirings" );
   }

   void
   runOtherBenchmarks( const HostDigraph& digraph,
                       const HostGraph& graph,
                       IndexType smallestNode,
                       IndexType largestNode,
                       TNL::Benchmarks::Benchmark<>& benchmark )
   {
      if( withBoost )
         runBoostBFS( digraph, graph, largestNode, benchmark );
      if( withGunrock )
         runGunrockBFS( digraph, graph, largestNode, benchmark );
   }

   void
   runBoostBFS( const HostDigraph& digraph,
                const HostGraph& graph,
                IndexType largestNode,
                TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_BOOST
      BoostGraph< Index, Real, TNL::Graphs::DirectedGraph > boostDigraph( digraph );
      BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph > boostGraph( graph );
      benchmark.setMetadataElement( { "solver", "Boost" } );

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

      // Convert and normalize distances
      this->boostBfsDistancesDirected = HostIndexVector( boostBfsDistances );
      this->boostBfsDistancesDirected.forAllElements(
         [ largestNode ] __cuda_callable__( Index i, Index & x )
         {
            if( x == std::numeric_limits< Index >::max() || ( x == 0 && i != largestNode ) )
               x = -1;
         } );

      // Benchmarking breadth-first search of undirected graph
      benchmark.setMetadataElement( { "problem", "BFS undir" } );

      auto bfs_boost_undir = [ & ]() mutable
      {
         boostGraph.breadthFirstSearch( largestNode, boostBfsDistances );
      };
      benchmark.time< TNL::Devices::Sequential >( "sequential", bfs_boost_undir );

      // Convert and normalize distances
      this->boostBfsDistancesUndirected = HostIndexVector( boostBfsDistances );
      this->boostBfsDistancesUndirected.forAllElements(
         [ largestNode ] __cuda_callable__( Index i, Index & x )
         {
            if( x == std::numeric_limits< Index >::max() || ( x == 0 && i != largestNode ) )
               x = -1;
         } );
#endif  // HAVE_BOOST
   }

   void
   runGunrockBFS( const HostDigraph& digraph,
                  const HostGraph& graph,
                  IndexType largestNode,
                  TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_GUNROCK
      // Convert TNL graphs to Gunrock format
      auto gunrockDigraph = GunrockBenchmark< Real, Index >::convertToGunrockGraph( digraph );
      auto gunrockGraph = GunrockBenchmark< Real, Index >::convertToGunrockGraph( graph );

      GunrockBenchmark< Real, Index > gunrockBenchmark;
      benchmark.setMetadataElement( { "solver", "Gunrock" } );

      // Benchmarking breadth-first search of directed graph
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS dir" } );
      benchmark.setMetadataElement( { "format", "N/A" } );

      std::vector< Index > bfsDistances( digraph.getVertexCount() );
      benchmark.setCatchExceptions( false );
      gunrockBenchmark.breadthFirstSearch( benchmark, gunrockDigraph, largestNode, digraph.getVertexCount(), bfsDistances );

      // Convert and normalize distances
      this->gunrockBfsDistancesDirected = bfsDistances;
      this->gunrockBfsDistancesDirected.forAllElements(
         [] __cuda_callable__( Index i, Index & x )
         {
            if( x == std::numeric_limits< Index >::max() )
               x = -1;
         } );

   #ifdef HAVE_BOOST
      if( withBoost && this->boostBfsDistancesDirected != this->gunrockBfsDistancesDirected ) {
         std::cout << "BFS distances of directed graph from Boost and Gunrock are not equal!" << std::endl;
         this->errors++;
      }
   #endif

      // Benchmarking breadth-first search of undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS undir" } );
      benchmark.setMetadataElement( { "threads mapping", "" } );

      try {
         gunrockBenchmark.breadthFirstSearch( benchmark, gunrockGraph, largestNode, graph.getVertexCount(), bfsDistances );
      }
      catch( const std::exception& e ) {
         std::cerr << "Gunrock BFS on undirected graph failed: " << e.what() << std::endl;
         this->errors++;
         return;
      }

      // Convert and normalize distances
      this->gunrockBfsDistancesUndirected = HostIndexVector( bfsDistances );
      this->gunrockBfsDistancesUndirected.forAllElements(
         [] __cuda_callable__( Index i, Index & x )
         {
            if( x == std::numeric_limits< Index >::max() )
               x = -1;
         } );

   #ifdef HAVE_BOOST
      if( withBoost && this->boostBfsDistancesUndirected != this->gunrockBfsDistancesUndirected ) {
         std::cout << "BFS distances of undirected graph from Boost and Gunrock are not equal!" << std::endl;
         this->errors++;
      }
   #endif
#endif  // HAVE_GUNROCK
   }

   template< typename Digraph, typename Graph >
   void
   runTNLAlgorithm( Digraph& digraph,
                    Graph& graph,
                    IndexType smallestNode,
                    IndexType largestNode,
                    TNL::Benchmarks::Benchmark<>& benchmark,
                    const TNL::String& device,
                    const TNL::String& segments )
   {
      using Device = typename std::remove_reference_t< decltype( digraph ) >::DeviceType;
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;

      // Benchmarking breadth-first search with directed graph
      IndexVector bfsDistances( digraph.getVertexCount() );
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS dir" } );
      benchmark.setMetadataElement( { "format", segments } );

      for( const auto& launchEntry :
           Algorithms::Segments::traversingLaunchConfigurations( digraph.getAdjacencyMatrix().getSegments() ) )
      {
         const auto& launchConfig = launchEntry.first;
         const auto& tag = launchEntry.second;

         benchmark.setMetadataElement( { "threads mapping", tag } );
         auto bfs_tnl_dir = [ &, launchConfig ]() mutable
         {
            TNL::Graphs::Algorithms::breadthFirstSearch( digraph, largestNode, bfsDistances, launchConfig );
         };
         benchmark.time< Device >( device, bfs_tnl_dir );

#ifdef HAVE_BOOST
         if( withBoost && bfsDistances != this->boostBfsDistancesDirected ) {
            std::cout << "BFS distances of directed graph from Boost and TNL are not equal!" << '\n';
            this->errors++;
         }
#endif
#ifdef HAVE_GUNROCK
         if( withGunrock && bfsDistances != this->gunrockBfsDistancesDirected ) {
            std::cout << "BFS distances of directed graph from TNL and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
#endif
      }

      // Benchmarking breadth-first search with undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS undir" } );
      benchmark.setMetadataElement( { "format", segments } );

      for( const auto& launchEntry :
           Algorithms::Segments::traversingLaunchConfigurations( graph.getAdjacencyMatrix().getSegments() ) )
      {
         const auto& launchConfig = launchEntry.first;
         const auto& tag = launchEntry.second;

         benchmark.setMetadataElement( std::make_pair( "threads mapping", tag ) );

         auto bfs_tnl_undir = [ &, launchConfig ]() mutable
         {
            TNL::Graphs::Algorithms::breadthFirstSearch( graph, largestNode, bfsDistances, launchConfig );
         };
         benchmark.time< Device >( device, bfs_tnl_undir );

#ifdef HAVE_BOOST
         if( withBoost && bfsDistances != this->boostBfsDistancesUndirected ) {
            std::cout << "BFS distances of undirected graph from Boost and TNL are not equal!" << '\n';
            this->errors++;
         }
#endif
#ifdef HAVE_GUNROCK
         if( withGunrock && bfsDistances != this->gunrockBfsDistancesUndirected ) {
            std::cout << "BFS distances of undirected graph from TNL and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
#endif
      }

      if( withSemirings && ! std::is_same_v< Device, TNL::Devices::Sequential > ) {
         // Benchmarking semiring-based BFS with directed graph
         IndexVector semiringBfsDistances( digraph.getVertexCount() );
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "Semiring BFS dir" } );
         benchmark.setMetadataElement( { "format", segments } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         auto semiring_bfs_dir = [ & ]() mutable
         {
            semiringBFS( digraph, largestNode, semiringBfsDistances );
         };
         benchmark.time< Device >( device, semiring_bfs_dir );
#ifdef HAVE_BOOST
         if( withBoost && semiringBfsDistances != this->boostBfsDistancesUndirected ) {
            std::cout << "BFS distances of undirected graph from Boost and TNL are not equal!" << '\n';
            this->errors++;
         }
#endif

         // Benchmarking semiring-based BFS with undirected graph
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "Semiring BFS undir" } );
         benchmark.setMetadataElement( { "format", segments } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         auto semiring_bfs_undir = [ & ]() mutable
         {
            semiringBFS( graph, largestNode, semiringBfsDistances );
         };
         benchmark.time< Device >( device, semiring_bfs_undir );
#ifdef HAVE_BOOST
         if( withBoost && semiringBfsDistances != this->boostBfsDistancesUndirected ) {
            std::cout << "BFS distances of undirected graph from Boost and TNL are not equal!" << '\n';
            this->errors++;
         }
#endif
      }
   }

protected:
   // Reference solutions for comparison
   HostIndexVector boostBfsDistancesDirected, boostBfsDistancesUndirected;
   HostIndexVector gunrockBfsDistancesDirected, gunrockBfsDistancesUndirected;
   bool withBoost;
   bool withGunrock;
   bool withSemirings;
};

}  // namespace TNL::Benchmarks::Graphs
