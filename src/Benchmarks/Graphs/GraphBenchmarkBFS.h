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
      config.addDelimiter( "BFS benchmark settings:" );
      config.addEntry< bool >( "with-semirings", "Run semiring-based BFS and SSSP benchmarks.", true );
      config.addEntry< bool >( "with-gunrock", "Run Gunrock benchmarks.", true );
      config.addEntry< bool >( "with-boost", "Run Boost benchmarks.", true );
      config.addEntry< double >(
         "bitmap-threshold", "Frontier fraction below which top-down bitmap mode is used (0 = disabled).", 0.0 );
      config.addEntry< double >(
         "bottomup-threshold", "Frontier fraction above which bottom-up mode is used (0 = disabled, undirected only).", 0.0 );
      config.addEntry< bool >( "with-predecessors", "Benchmark BFS with predecessor tracking.", false );
      config.addEntry< bool >( "with-visitor", "Benchmark BFS with a visitor callback.", false );
      config.addEntry< bool >( "deterministic", "Use deterministic predecessor selection (smallest source per layer).", false );
   }

   GraphBenchmarkBFS( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {
      withBoost = this->parameters.template getParameter< bool >( "with-boost" );
      withGunrock = this->parameters.template getParameter< bool >( "with-gunrock" );
      withSemirings = this->parameters.template getParameter< bool >( "with-semirings" );
      bitmapThreshold = this->parameters.template getParameter< double >( "bitmap-threshold" );
      bottomUpThreshold = this->parameters.template getParameter< double >( "bottomup-threshold" );
      withPredecessors = this->parameters.template getParameter< bool >( "with-predecessors" );
      withVisitor = this->parameters.template getParameter< bool >( "with-visitor" );
      deterministic = this->parameters.template getParameter< bool >( "deterministic" );
   }

   void
   runOtherBenchmarks(
      const HostDigraph& digraph,
      const HostGraph& graph,
      IndexType smallestNode,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
      if( withBoost )
         runBoostBFS( digraph, graph, largestNode, benchmark );
      if( withGunrock )
         runGunrockBFS( digraph, graph, largestNode, benchmark );
   }

   void
   runBoostBFS(
      const HostDigraph& digraph,
      const HostGraph& graph,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_BOOST
      BoostGraph< Index, Real, TNL::Graphs::DirectedGraph > boostDigraph( digraph );
      BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph > boostGraph( graph );
      benchmark.setMetadataElement( { "solver", "Boost" } );

      // Benchmarking breadth-first search of directed graph
      benchmark.setMetadataElement( { "problem", "BFS dir" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "" } );

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
   runGunrockBFS(
      const HostDigraph& digraph,
      const HostGraph& graph,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_GUNROCK
      // Convert TNL graphs to Gunrock format
      auto gunrockDigraphHolder = GunrockBenchmark< Real, Index >::convertToGunrockGraph( digraph );
      auto gunrockGraphHolder = GunrockBenchmark< Real, Index >::convertToGunrockGraph( graph );

      GunrockBenchmark< Real, Index > gunrockBenchmark;
      benchmark.setMetadataElement( { "solver", "Gunrock" } );

      // Benchmarking breadth-first search of directed graph
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS dir" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );

      std::vector< Index > bfsDistances( digraph.getVertexCount() );
      benchmark.setCatchExceptions( false );
      gunrockBenchmark.breadthFirstSearch(
         benchmark, gunrockDigraphHolder.graph, largestNode, digraph.getVertexCount(), bfsDistances );

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
         std::cout << "BFS distances of directed graph from Boost and Gunrock are not equal!\n";
         this->errors++;
      }
   #endif

      // Benchmarking breadth-first search of undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS undir" } );
      benchmark.setMetadataElement( { "launch cfg.", "" } );

      try {
         gunrockBenchmark.breadthFirstSearch(
            benchmark, gunrockGraphHolder.graph, largestNode, graph.getVertexCount(), bfsDistances );
      }
      catch( const std::exception& e ) {
         std::cerr << "Gunrock BFS on undirected graph failed: " << e.what() << '\n';
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
         std::cout << "BFS distances of undirected graph from Boost and Gunrock are not equal!\n";
         this->errors++;
      }
   #endif
#endif  // HAVE_GUNROCK
   }

   struct NoOpVisitor
   {
      __cuda_callable__
      void
      operator()( IndexType, IndexType ) const
      {}
   };

   template< typename Digraph, typename Graph >
   void
   runTNLAlgorithm(
      Digraph& digraph,
      Graph& graph,
      IndexType smallestNode,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark,
      const TNL::String& device,
      const TNL::String& segments )
   {
      using Device = typename std::remove_reference_t< decltype( digraph ) >::DeviceType;
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;

      const auto runBFS =
         [ & ](
            auto& g, auto& dist, auto& pred, const auto& launchCfg, const char* modeTag, double bitmapThr, double bottomUpThr )
      {
         benchmark.setMetadataElement( { "mode", modeTag } );
         auto bfs_lambda = [ &, launchCfg, bitmapThr, bottomUpThr ]() mutable
         {
            NoOpVisitor visitor;
            if( withVisitor && withPredecessors )
               TNL::Graphs::Algorithms::breadthFirstSearchWithVisitorAndPredecessors(
                  g, largestNode, visitor, dist, pred, deterministic, bitmapThr, bottomUpThr, launchCfg );
            else if( withVisitor )
               TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
                  g, largestNode, visitor, dist, bitmapThr, bottomUpThr, launchCfg );
            else if( withPredecessors )
               TNL::Graphs::Algorithms::breadthFirstSearchWithPredecessors(
                  g, largestNode, dist, pred, deterministic, bitmapThr, bottomUpThr, launchCfg );
            else
               TNL::Graphs::Algorithms::breadthFirstSearch( g, largestNode, dist, bitmapThr, bottomUpThr, launchCfg );
         };
         benchmark.time< Device >( device, bfs_lambda );
      };

      // Benchmarking BFS with directed graph
      {
         IndexVector bfsDistances( digraph.getVertexCount() );
         IndexVector bfsPredecessors( digraph.getVertexCount() );
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS dir" } );
         benchmark.setMetadataElement( { "kernel", segments } );

         for( const auto& launchEntry :
              Algorithms::Segments::traversingLaunchConfigurations( digraph.getAdjacencyMatrix().getSegments() ) )
         {
            const auto& launchConfig = launchEntry.first;
            const auto& tag = launchEntry.second;
            benchmark.setMetadataElement( { "launch cfg.", tag } );

            runBFS( digraph, bfsDistances, bfsPredecessors, launchConfig, "top-down compact", 0.0, 0.0 );

#ifdef HAVE_BOOST
            if( withBoost && bfsDistances != this->boostBfsDistancesDirected ) {
               std::cout << "BFS distances of directed graph from Boost and TNL are not equal!\n";
               this->errors++;
            }
#endif
#ifdef HAVE_GUNROCK
            if( withGunrock && bfsDistances != this->gunrockBfsDistancesDirected ) {
               std::cout << "BFS distances of directed graph from TNL and Gunrock are not equal!\n";
               this->errors++;
            }
#endif

            if( bitmapThreshold > 0.0 )
               runBFS( digraph, bfsDistances, bfsPredecessors, launchConfig, "top-down bitmap", bitmapThreshold, 0.0 );
         }
      }

      // Benchmarking BFS with undirected graph
      {
         IndexVector bfsDistances( graph.getVertexCount() );
         IndexVector bfsPredecessors( graph.getVertexCount() );
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS undir" } );
         benchmark.setMetadataElement( { "kernel", segments } );

         for( const auto& launchEntry :
              Algorithms::Segments::traversingLaunchConfigurations( graph.getAdjacencyMatrix().getSegments() ) )
         {
            const auto& launchConfig = launchEntry.first;
            const auto& tag = launchEntry.second;
            benchmark.setMetadataElement( { "launch cfg.", tag } );

            runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "top-down compact", 0.0, 0.0 );

#ifdef HAVE_BOOST
            if( withBoost && bfsDistances != this->boostBfsDistancesUndirected ) {
               std::cout << "BFS distances of undirected graph from Boost and TNL are not equal!\n";
               this->errors++;
            }
#endif
#ifdef HAVE_GUNROCK
            if( withGunrock && bfsDistances != this->gunrockBfsDistancesUndirected ) {
               std::cout << "BFS distances of undirected graph from TNL and Gunrock are not equal!\n";
               this->errors++;
            }
#endif

            if( bitmapThreshold > 0.0 )
               runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "top-down bitmap", bitmapThreshold, 0.0 );

            if( bottomUpThreshold > 0.0 )
               runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "bottom-up", 0.0, bottomUpThreshold );

            if( bitmapThreshold > 0.0 && bottomUpThreshold > 0.0 )
               runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "hybrid", bitmapThreshold, bottomUpThreshold );
         }
      }

      if( withSemirings && ! std::is_same_v< Device, TNL::Devices::Sequential > ) {
         // Benchmarking semiring-based BFS with directed graph
         IndexVector semiringBfsDistances( digraph.getVertexCount() );
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "Semiring BFS dir" } );
         benchmark.setMetadataElement( { "kernel", segments } );
         benchmark.setMetadataElement( { "mode", "N/A" } );
         benchmark.setMetadataElement( { "launch cfg.", "" } );

         auto semiring_bfs_dir = [ & ]() mutable
         {
            semiringBFS( digraph, largestNode, semiringBfsDistances );
         };
         benchmark.time< Device >( device, semiring_bfs_dir );
#ifdef HAVE_BOOST
         if( withBoost && semiringBfsDistances != this->boostBfsDistancesDirected ) {
            std::cout << "BFS distances of directed graph from Boost and TNL are not equal!\n";
            this->errors++;
         }
#endif

         // Benchmarking semiring-based BFS with undirected graph
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "Semiring BFS undir" } );
         benchmark.setMetadataElement( { "kernel", segments } );
         benchmark.setMetadataElement( { "launch cfg.", "" } );

         auto semiring_bfs_undir = [ & ]() mutable
         {
            semiringBFS( graph, largestNode, semiringBfsDistances );
         };
         benchmark.time< Device >( device, semiring_bfs_undir );
#ifdef HAVE_BOOST
         if( withBoost && semiringBfsDistances != this->boostBfsDistancesUndirected ) {
            std::cout << "BFS distances of undirected graph from Boost and TNL are not equal!\n";
            this->errors++;
         }
#endif
      }
   }

protected:
   HostIndexVector boostBfsDistancesDirected, boostBfsDistancesUndirected;
   HostIndexVector gunrockBfsDistancesDirected, gunrockBfsDistancesUndirected;
   bool withBoost;
   bool withGunrock;
   bool withSemirings;
   double bitmapThreshold;
   double bottomUpThreshold;
   bool withPredecessors;
   bool withVisitor;
   bool deterministic;
};

}  // namespace TNL::Benchmarks::Graphs
