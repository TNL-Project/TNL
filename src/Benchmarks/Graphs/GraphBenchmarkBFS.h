// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <climits>
#include <cstdint>
#include <exception>
#include <string>
#include <tuple>
#include <vector>

#include "GraphBenchmarkBase.h"
#include "BoostGraph.h"
#include "GunrockBenchmark.h"
#ifdef HAVE_GAPBS
   #include "GapbsKernels.h"
#endif
#include <TNL/Graphs/Algorithms/breadthFirstSearch.h>
#include "SemiringBFS.h"

namespace TNL::Benchmarks::Graphs {

/**
 * \brief BFS benchmark following the GAP Benchmark Suite.
 *
 * The BFS is run from 64 source vertices picked like in the GAP reference code, each trial is timed separately and
 * includes the allocation of the outputs. The BFS kernel of GAP computes the parents, so the predecessors are
 * computed by default (`--with-predecessors`). The output of every trial is verified.
 */
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

   static constexpr bool isSingleSource = true;
   static constexpr int defaultTrials = 64;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      Base::configSetup( config );
      config.addDelimiter( "BFS benchmark settings:" );
      config.addEntry< bool >( "with-semirings", "Run semiring-based BFS and SSSP benchmarks.", true );
      config.addEntry< bool >( "with-gunrock", "Run Gunrock benchmarks.", true );
      config.addEntry< bool >( "with-boost", "Run Boost benchmarks.", true );
      config.addEntry< bool >( "with-gapbs", "Run the GAP reference code (gapbs) benchmarks.", true );
      config.addEntry< double >(
         "bitmap-threshold", "Frontier fraction below which top-down bitmap mode is used (0 = disabled).", 0.0 );
      config.addEntry< double >(
         "bottomup-threshold", "Frontier fraction above which bottom-up mode is used (0 = disabled, undirected only).", 0.0 );
      config.addEntry< bool >( "with-expand-contract", "Benchmark the top-down expand-contract mode.", true );
      config.addEntry< bool >(
         "with-predecessors", "Benchmark BFS with predecessor tracking, which is the BFS kernel of the GAP benchmark.", true );
      config.addEntry< bool >( "with-visitor", "Benchmark BFS with a visitor callback.", false );
      config.addEntry< bool >( "deterministic", "Use deterministic predecessor selection (smallest source per layer).", false );
   }

   GraphBenchmarkBFS( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {
      withBoost = this->parameters.template getParameter< bool >( "with-boost" );
      withGunrock = this->parameters.template getParameter< bool >( "with-gunrock" );
      withGapbs = this->parameters.template getParameter< bool >( "with-gapbs" );
      withSemirings = this->parameters.template getParameter< bool >( "with-semirings" );
      bitmapThreshold = this->parameters.template getParameter< double >( "bitmap-threshold" );
      bottomUpThreshold = this->parameters.template getParameter< double >( "bottomup-threshold" );
      withExpandContract = this->parameters.template getParameter< bool >( "with-expand-contract" );
      withPredecessors = this->parameters.template getParameter< bool >( "with-predecessors" );
      withVisitor = this->parameters.template getParameter< bool >( "with-visitor" );
      deterministic = this->parameters.template getParameter< bool >( "deterministic" );
      maxBoostEdges = this->parameters.template getParameter< int >( "max-boost-edges" );
   }

   static std::string
   problemName( bool directed )
   {
      return directed ? "BFS dir" : "BFS undir";
   }

   void
   runOtherBenchmarks(
      const HostDigraph& digraph,
      const HostGraph& graph,
      IndexType smallestNode,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
      if( withGapbs ) {
         if( this->runDirected )
            runGapbsBFS( this->directedInputFile, digraph, true, benchmark );
         if( this->runUndirected )
            runGapbsBFS( this->undirectedInputFile, graph, false, benchmark );
      }
      if( withBoost ) {
         if( this->runDirected )
            runBoostBFS( digraph, true, benchmark );
         if( this->runUndirected )
            runBoostBFS( graph, false, benchmark );
      }
      if( withGunrock ) {
         if( this->runDirected )
            runGunrockBFS( digraph, true, benchmark );
         if( this->runUndirected )
            runGunrockBFS( graph, false, benchmark );
      }
   }

   template< typename HostGraphType >
   void
   setOtherSolverMetadata(
      TNL::Benchmarks::Benchmark& benchmark,
      const HostGraphType& hostGraph,
      bool directed,
      const std::string& solver,
      const std::string& mode )
   {
      benchmark.setDatasetSize( hostGraph.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "solver", solver } );
      benchmark.setMetadataElement( { "problem", problemName( directed ) } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "" } );
      benchmark.setMetadataElement( { "mode", mode } );
   }

   template< typename HostGraphType >
   void
   runGapbsBFS( const std::string& file, const HostGraphType& hostGraph, bool directed, TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_GAPBS
      if( ! this->gapInput ) {
         std::cout << "Skipping gapbs, it needs an .sg input file of the GAP benchmark.\n";
         return;
      }
      try {
         Gapbs::BFS gapbs( file );
         setOtherSolverMetadata( benchmark, hostGraph, directed, "gapbs", "direction-optimizing" );
         std::vector< std::int32_t > parents;
         const auto& matrix = hostGraph.getAdjacencyMatrix();
         auto reset = [ & ]()
         {
            gapbs.release();
         };
         auto compute = [ & ]( Index source )
         {
            gapbs.run( source );
         };
         auto check = [ & ]( Index source )
         {
            gapbs.getParents( parents );
            return checkOutputSize(
               parents.size(),
               hostGraph,
               [ & ]()
               {
                  return verifyBFSParents( matrix, source, parents.data() );
               } );
         };
         this->template timeSingleSourceTrials< TNL::Devices::Host >(
            benchmark, "gapbs", directed, "gapbs " + problemName( directed ), reset, compute, check );
      }
      catch( const std::exception& e ) {
         std::cerr << "gapbs BFS failed: " << e.what() << '\n';
         this->errors++;
      }
#endif
   }

   template< typename HostGraphType >
   void
   runBoostBFS( const HostGraphType& hostGraph, bool directed, TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_BOOST
      if( hostGraph.getAdjacencyMatrix().getNonzeroElementsCount() > maxBoostEdges ) {
         std::cout << "Skipping Boost, the graph has more than " << maxBoostEdges << " edges (see --max-boost-edges).\n";
         return;
      }
      BoostGraph< Index, Real, typename HostGraphType::GraphOrientation > boostGraph( hostGraph );
      setOtherSolverMetadata( benchmark, hostGraph, directed, "Boost", "N/A" );

      std::vector< Index > distances;
      std::vector< Index > predecessors;
      const auto& matrix = hostGraph.getAdjacencyMatrix();
      auto reset = [ & ]()
      {
         std::vector< Index >().swap( distances );
         std::vector< Index >().swap( predecessors );
      };
      // Boost computes the predecessors always, it does not have a BFS variant without them.
      auto compute = [ & ]( Index source )
      {
         boostGraph.breadthFirstSearch( source, distances, predecessors );
      };
      auto check = [ & ]( Index source )
      {
         return checkOutputSize(
            distances.size(),
            hostGraph,
            [ & ]()
            {
               return verifyBFS( matrix, source, distances.data(), predecessors.data() );
            } );
      };
      this->template timeSingleSourceTrials< TNL::Devices::Sequential >(
         benchmark, "Boost", directed, "Boost " + problemName( directed ), reset, compute, check );
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

   template< typename HostGraphType >
   void
   runGunrockBFS( const HostGraphType& hostGraph, bool directed, TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_GUNROCK
      using Gunrock = GunrockBenchmark< Real, Index >;
      using VertexType = typename Gunrock::VertexType;
      const Index n = hostGraph.getVertexCount();
      if( n > INT_MAX ) {
         std::cout << "Skipping Gunrock, it uses 32-bit vertex indexes.\n";
         return;
      }
      try {
         // Convert TNL graph to Gunrock format
         auto gunrockGraphHolder = Gunrock::convertToGunrockGraph( hostGraph );
         setOtherSolverMetadata( benchmark, hostGraph, directed, "Gunrock", "N/A" );
         const auto& matrix = hostGraph.getAdjacencyMatrix();

         thrust::device_vector< VertexType > distances;
         thrust::device_vector< VertexType > predecessors;
         std::vector< VertexType > hostDistances;
         std::vector< VertexType > hostPredecessors;

         // Propagate exceptions (e.g. an unsupported load-balance strategy on
         // this platform) to the try/catch below instead of having them
         // silently swallowed inside benchmark.time().
         benchmark.setCatchExceptions( false );
         for( const auto& [ loadBalance, tag, withFilter ] : gunrockConfigurations() ) {
            benchmark.setMetadataElement( { "launch cfg.", tag } );
            auto reset = [ & ]()
            {
               Gunrock::release( distances );
               Gunrock::release( predecessors );
            };
            auto compute = [ &, loadBalance = loadBalance, withFilter = withFilter ]( Index source )
            {
               Gunrock::breadthFirstSearch(
                  gunrockGraphHolder.graph, source, n, distances, predecessors, loadBalance, withFilter );
            };
            auto check = [ & ]( Index source )
            {
               hostDistances.resize( distances.size() );
               hostPredecessors.resize( predecessors.size() );
               thrust::copy( distances.begin(), distances.end(), hostDistances.begin() );
               thrust::copy( predecessors.begin(), predecessors.end(), hostPredecessors.begin() );
               for( auto& d : hostDistances )
                  if( d == std::numeric_limits< VertexType >::max() )
                     d = -1;
               return checkOutputSize(
                  hostDistances.size(),
                  hostGraph,
                  [ & ]()
                  {
                     return verifyBFS< std::decay_t< decltype( matrix ) >, VertexType >(
                        matrix, source, hostDistances.data(), hostPredecessors.data() );
                  } );
            };
            try {
               this->template timeSingleSourceTrials< TNL::Devices::GPU >(
                  benchmark, "Gunrock", directed, "Gunrock " + problemName( directed ) + " " + tag, reset, compute, check );
            }
            catch( const std::exception& e ) {
               std::cerr << "Gunrock " << problemName( directed ) << " with load balance '" << tag << "' failed: " << e.what()
                         << '\n';
            }
         }
         benchmark.setCatchExceptions( true );
      }
      catch( const std::bad_alloc& e ) {
         std::cerr << "Skipping Gunrock, the graph does not fit in the memory: " << e.what() << '\n';
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

   // Returns a violation if a solver did not produce an output of the right size (e.g. because it threw an exception),
   // otherwise the result of the verification.
   template< typename HostGraphType, typename Verify >
   static GraphVerificationResult
   checkOutputSize( std::size_t size, const HostGraphType& hostGraph, Verify&& verify )
   {
      if( size != static_cast< std::size_t >( hostGraph.getVertexCount() ) ) {
         GraphVerificationResult result;
         result.violations = 1;
         result.message = "the output has a wrong size";
         return result;
      }
      return verify();
   }

   struct TNLMode
   {
      const char* tag;
      double bitmapThreshold;
      double bottomUpThreshold;
      TNL::Graphs::Algorithms::BFSTopDownMode topDownMode;
   };

   [[nodiscard]] std::vector< TNLMode >
   getTNLModes( bool directed ) const
   {
      using TNL::Graphs::Algorithms::BFSTopDownMode;
      std::vector< TNLMode > modes{ { "top-down compact", 0.0, 0.0, BFSTopDownMode::Compact } };
      if( withExpandContract )
         modes.push_back( { "top-down expand-contract", 0.0, 0.0, BFSTopDownMode::ExpandContract } );
      if( bitmapThreshold > 0.0 )
         modes.push_back( { "top-down bitmap", bitmapThreshold, 0.0, BFSTopDownMode::Compact } );
      if( ! directed && bottomUpThreshold > 0.0 )
         modes.push_back( { "bottom-up", 0.0, bottomUpThreshold, BFSTopDownMode::Compact } );
      if( ! directed && bitmapThreshold > 0.0 && bottomUpThreshold > 0.0 )
         modes.push_back( { "hybrid", bitmapThreshold, bottomUpThreshold, BFSTopDownMode::Compact } );
      return modes;
   }

   template< typename DeviceGraph, typename HostGraphType >
   void
   runTNLBFS(
      DeviceGraph& g,
      const HostGraphType& hostGraph,
      bool directed,
      TNL::Benchmarks::Benchmark& benchmark,
      const TNL::String& segments )
   {
      using Device = typename DeviceGraph::DeviceType;
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;

      IndexVector distances;
      IndexVector predecessors;
      HostIndexVector hostDistances;
      HostIndexVector hostPredecessors;
      const auto& matrix = hostGraph.getAdjacencyMatrix();
      benchmark.setDatasetSize( g.getAdjacencyMatrix().getNonzeroElementsCount() * sizeof( Index ) );
      benchmark.setMetadataElement( { "problem", problemName( directed ) } );
      benchmark.setMetadataElement( { "kernel", segments } );

      for( const auto& [ launchConfig, tag ] :
           Algorithms::Segments::traversingLaunchConfigurations( g.getAdjacencyMatrix().getSegments() ) )
      {
         benchmark.setMetadataElement( { "launch cfg.", tag } );
         for( const TNLMode& mode : getTNLModes( directed ) ) {
            benchmark.setMetadataElement( { "mode", mode.tag } );
            auto reset = [ & ]()
            {
               // the BFS allocates the outputs, so the allocation is measured
               distances.reset();
               predecessors.reset();
            };
            auto compute = [ &, launchConfig = launchConfig ]( Index source )
            {
               NoOpVisitor visitor;
               if( withVisitor && withPredecessors )
                  TNL::Graphs::Algorithms::breadthFirstSearchWithVisitorAndPredecessors(
                     g,
                     source,
                     visitor,
                     distances,
                     predecessors,
                     deterministic,
                     mode.bitmapThreshold,
                     mode.bottomUpThreshold,
                     mode.topDownMode,
                     launchConfig );
               else if( withVisitor )
                  TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor(
                     g,
                     source,
                     visitor,
                     distances,
                     mode.bitmapThreshold,
                     mode.bottomUpThreshold,
                     mode.topDownMode,
                     launchConfig );
               else if( withPredecessors )
                  TNL::Graphs::Algorithms::breadthFirstSearchWithPredecessors(
                     g,
                     source,
                     distances,
                     predecessors,
                     deterministic,
                     mode.bitmapThreshold,
                     mode.bottomUpThreshold,
                     mode.topDownMode,
                     launchConfig );
               else
                  TNL::Graphs::Algorithms::breadthFirstSearch(
                     g, source, distances, mode.bitmapThreshold, mode.bottomUpThreshold, mode.topDownMode, launchConfig );
            };
            auto check = [ & ]( Index source )
            {
               hostDistances = distances;
               hostPredecessors = predecessors;
               return checkOutputSize(
                  hostDistances.getSize(),
                  hostGraph,
                  [ & ]()
                  {
                     return verifyBFS(
                        matrix, source, hostDistances.getData(), withPredecessors ? hostPredecessors.getData() : nullptr );
                  } );
            };
            this->template timeSingleSourceTrials< Device >(
               benchmark,
               "TNL",
               directed,
               "TNL " + problemName( directed ) + " " + std::string( segments ) + " " + tag + " " + mode.tag,
               reset,
               compute,
               check );
         }
      }

      if( withSemirings && ! std::is_same_v< Device, TNL::Devices::Sequential > ) {
         // The semiring-based BFS computes only the distances.
         benchmark.setMetadataElement( { "problem", "Semiring " + problemName( directed ) } );
         benchmark.setMetadataElement( { "mode", "N/A" } );
         benchmark.setMetadataElement( { "launch cfg.", "" } );
         auto reset = [ & ]()
         {
            distances.reset();
         };
         auto compute = [ & ]( Index source )
         {
            semiringBFS( g, source, distances );
         };
         auto check = [ & ]( Index source )
         {
            hostDistances = distances;
            return checkOutputSize(
               hostDistances.getSize(),
               hostGraph,
               [ & ]()
               {
                  return verifyBFS( matrix, source, hostDistances.getData() );
               } );
         };
         this->template timeSingleSourceTrials< Device >(
            benchmark,
            "TNL",
            directed,
            "TNL Semiring " + problemName( directed ) + " " + std::string( segments ),
            reset,
            compute,
            check );
      }
   }

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
      if( this->runDirected )
         runTNLBFS( digraph, *this->hostDigraphPointer, true, benchmark, segments );
      if( this->runUndirected )
         runTNLBFS( graph, *this->hostGraphPointer, false, benchmark, segments );
   }

protected:
   bool withBoost;
   bool withGunrock;
   bool withGapbs;
   bool withSemirings;
   double bitmapThreshold;
   double bottomUpThreshold;
   bool withExpandContract;
   bool withPredecessors;
   bool withVisitor;
   bool deterministic;
   std::int64_t maxBoostEdges;
};

}  // namespace TNL::Benchmarks::Graphs
