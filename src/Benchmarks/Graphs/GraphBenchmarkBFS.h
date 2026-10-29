// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <tuple>
#include <vector>

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
      config.addEntry< bool >( "with-expand-contract", "Benchmark the top-down expand-contract mode.", true );
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
      withExpandContract = this->parameters.template getParameter< bool >( "with-expand-contract" );
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
      benchmark.time< TNL::Devices::Sequential >( "Boost", bfs_boost_dir );

      // Convert and normalize distances
      this->boostBfsDistancesDirected = HostIndexVector( boostBfsDistances );
      this->boostBfsDistancesDirected.forAllElements(
         [ largestNode ] __cuda_callable__( Index i, Index& x )
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
      benchmark.time< TNL::Devices::Sequential >( "Boost", bfs_boost_undir );

      // Convert and normalize distances
      this->boostBfsDistancesUndirected = HostIndexVector( boostBfsDistances );
      this->boostBfsDistancesUndirected.forAllElements(
         [ largestNode ] __cuda_callable__( Index i, Index& x )
         {
            if( x == std::numeric_limits< Index >::max() || ( x == 0 && i != largestNode ) )
               x = -1;
         } );
#endif  // HAVE_BOOST
   }

   // Gunrock's load-balancing strategies, each run without (Gunrock's default) and with the filter operator.
   static std::vector< std::tuple< typename GunrockBenchmark< Real, Index >::LoadBalance, std::string, bool > >
   gunrockConfigurations()
   {
      std::vector< std::tuple< typename GunrockBenchmark< Real, Index >::LoadBalance, std::string, bool > > configurations;
      for( const auto& [ loadBalance, tag ] : GunrockBenchmark< Real, Index >::loadBalanceConfigurations() ) {
         configurations.emplace_back( loadBalance, tag, false );
         configurations.emplace_back( loadBalance, tag + " filter", true );
      }
      return configurations;
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
      // Propagate exceptions (e.g. an unsupported load-balance strategy on
      // this platform) to the per-iteration try/catch below instead of
      // having them silently swallowed inside benchmark.time().
      benchmark.setCatchExceptions( false );

      // Benchmarking breadth-first search of directed graph
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS dir" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );

      std::vector< Index > bfsDistances( digraph.getVertexCount() );
      for( const auto& [ loadBalance, tag, withFilter ] : gunrockConfigurations() ) {
         benchmark.setMetadataElement( { "launch cfg.", tag } );

         try {
            gunrockBenchmark.breadthFirstSearch(
               benchmark,
               gunrockDigraphHolder.graph,
               largestNode,
               digraph.getVertexCount(),
               bfsDistances,
               loadBalance,
               withFilter );
         }
         catch( const std::exception& e ) {
            std::cerr << "Gunrock BFS on directed graph with load balance '" << tag << "' failed: " << e.what() << '\n';
            continue;
         }

         // Convert and normalize distances
         HostIndexVector gunrockDistances( bfsDistances );
         gunrockDistances.forAllElements(
            [] __cuda_callable__( Index i, Index& x )
            {
               if( x == std::numeric_limits< Index >::max() )
                  x = -1;
            } );

         // The block-mapped strategy is Gunrock's own default (used by its own
         // benchmarks and by our TNL-vs-Gunrock cross-checks below).
         if( loadBalance == GunrockBenchmark< Real, Index >::LoadBalance::BlockMapped && ! withFilter )
            this->gunrockBfsDistancesDirected = gunrockDistances;

   #ifdef HAVE_BOOST
         if( withBoost && this->boostBfsDistancesDirected != gunrockDistances ) {
            std::cout << "BFS distances of directed graph from Boost and Gunrock (" << tag << ") are not equal!\n";
            this->errors++;
         }
   #endif
      }

      // Benchmarking breadth-first search of undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", "BFS undir" } );

      for( const auto& [ loadBalance, tag, withFilter ] : gunrockConfigurations() ) {
         benchmark.setMetadataElement( { "launch cfg.", tag } );

         try {
            gunrockBenchmark.breadthFirstSearch(
               benchmark,
               gunrockGraphHolder.graph,
               largestNode,
               graph.getVertexCount(),
               bfsDistances,
               loadBalance,
               withFilter );
         }
         catch( const std::exception& e ) {
            std::cerr << "Gunrock BFS on undirected graph with load balance '" << tag << "' failed: " << e.what() << '\n';
            continue;
         }

         // Convert and normalize distances
         HostIndexVector gunrockDistances( bfsDistances );
         gunrockDistances.forAllElements(
            [] __cuda_callable__( Index i, Index& x )
            {
               if( x == std::numeric_limits< Index >::max() )
                  x = -1;
            } );

         if( loadBalance == GunrockBenchmark< Real, Index >::LoadBalance::BlockMapped && ! withFilter )
            this->gunrockBfsDistancesUndirected = gunrockDistances;

   #ifdef HAVE_BOOST
         if( withBoost && this->boostBfsDistancesUndirected != gunrockDistances ) {
            std::cout << "BFS distances of undirected graph from Boost and Gunrock (" << tag << ") are not equal!\n";
            this->errors++;
         }
   #endif
      }
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
      const TNL::String& segments )
   {
      using Device = typename std::remove_reference_t< decltype( digraph ) >::DeviceType;
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;

      using TNL::Graphs::Algorithms::BFSTopDownMode;
      const auto runBFS = [ & ](
                             auto& g,
                             auto& dist,
                             auto& pred,
                             const auto& launchCfg,
                             const char* modeTag,
                             double bitmapThreshold,
                             double bottomUpThreshold,
                             BFSTopDownMode topDownMode = BFSTopDownMode::Compact )
      {
         benchmark.setMetadataElement( { "mode", modeTag } );
         auto bfs_lambda = [ &, launchCfg, bitmapThreshold, bottomUpThreshold, topDownMode ]() mutable
         {
            NoOpVisitor visitor;
            if( withVisitor && withPredecessors )
               TNL::Graphs::Algorithms::breadthFirstSearchWithVisitorAndPredecessors(
                  g,
                  largestNode,
                  visitor,
                  dist,
                  pred,
                  deterministic,
                  bitmapThreshold,
                  bottomUpThreshold,
                  topDownMode,
                  launchCfg );
            else if( withVisitor )
               TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
                  g, largestNode, visitor, dist, bitmapThreshold, bottomUpThreshold, topDownMode, launchCfg );
            else if( withPredecessors )
               TNL::Graphs::Algorithms::breadthFirstSearchWithPredecessors(
                  g, largestNode, dist, pred, deterministic, bitmapThreshold, bottomUpThreshold, topDownMode, launchCfg );
            else
               TNL::Graphs::Algorithms::breadthFirstSearch(
                  g, largestNode, dist, bitmapThreshold, bottomUpThreshold, topDownMode, launchCfg );
         };
         benchmark.time< Device >( "TNL", bfs_lambda );
      };

      // The other traversal modes must give the same distances as the top-down compact mode,
      // which is compared with Boost and Gunrock.
      const auto checkMode =
         [ & ]( const IndexVector& dist, const IndexVector& compactDist, const char* graphTag, const char* modeTag )
      {
         if( dist != compactDist ) {
            std::cout << "BFS distances of " << graphTag << " graph from TNL " << modeTag
                      << " and TNL top-down compact are not equal!\n";
            this->errors++;
         }
      };

      // Benchmarking BFS with directed graph
      {
         IndexVector bfsDistances( digraph.getVertexCount() );
         IndexVector bfsPredecessors( digraph.getVertexCount() );
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS dir" } );
         benchmark.setMetadataElement( { "kernel", segments } );

         for( const auto& [ launchConfig, tag ] :
              Algorithms::Segments::traversingLaunchConfigurations( digraph.getAdjacencyMatrix().getSegments() ) )
         {
            benchmark.setMetadataElement( { "launch cfg.", tag } );

            runBFS( digraph, bfsDistances, bfsPredecessors, launchConfig, "top-down compact", 0.0, 0.0 );
            const IndexVector compactDistances( bfsDistances );

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

            if( withExpandContract ) {
               runBFS(
                  digraph,
                  bfsDistances,
                  bfsPredecessors,
                  launchConfig,
                  "top-down expand-contract",
                  0.0,
                  0.0,
                  BFSTopDownMode::ExpandContract );
               checkMode( bfsDistances, compactDistances, "directed", "top-down expand-contract" );
            }

            if( bitmapThreshold > 0.0 ) {
               runBFS( digraph, bfsDistances, bfsPredecessors, launchConfig, "top-down bitmap", bitmapThreshold, 0.0 );
               checkMode( bfsDistances, compactDistances, "directed", "top-down bitmap" );
            }
         }
      }

      // Benchmarking BFS with undirected graph
      {
         IndexVector bfsDistances( graph.getVertexCount() );
         IndexVector bfsPredecessors( graph.getVertexCount() );
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
         benchmark.setMetadataElement( { "problem", "BFS undir" } );
         benchmark.setMetadataElement( { "kernel", segments } );

         for( const auto& [ launchConfig, tag ] :
              Algorithms::Segments::traversingLaunchConfigurations( graph.getAdjacencyMatrix().getSegments() ) )
         {
            benchmark.setMetadataElement( { "launch cfg.", tag } );

            runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "top-down compact", 0.0, 0.0 );
            const IndexVector compactDistances( bfsDistances );

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

            if( withExpandContract ) {
               runBFS(
                  graph,
                  bfsDistances,
                  bfsPredecessors,
                  launchConfig,
                  "top-down expand-contract",
                  0.0,
                  0.0,
                  BFSTopDownMode::ExpandContract );
               checkMode( bfsDistances, compactDistances, "undirected", "top-down expand-contract" );
            }

            if( bitmapThreshold > 0.0 ) {
               runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "top-down bitmap", bitmapThreshold, 0.0 );
               checkMode( bfsDistances, compactDistances, "undirected", "top-down bitmap" );
            }

            if( bottomUpThreshold > 0.0 ) {
               runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "bottom-up", 0.0, bottomUpThreshold );
               checkMode( bfsDistances, compactDistances, "undirected", "bottom-up" );
            }

            if( bitmapThreshold > 0.0 && bottomUpThreshold > 0.0 ) {
               runBFS( graph, bfsDistances, bfsPredecessors, launchConfig, "hybrid", bitmapThreshold, bottomUpThreshold );
               checkMode( bfsDistances, compactDistances, "undirected", "hybrid" );
            }
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
         benchmark.time< Device >( "TNL", semiring_bfs_dir );
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
         benchmark.time< Device >( "TNL", semiring_bfs_undir );
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
   bool withExpandContract;
   bool withPredecessors;
   bool withVisitor;
   bool deterministic;
};

}  // namespace TNL::Benchmarks::Graphs
