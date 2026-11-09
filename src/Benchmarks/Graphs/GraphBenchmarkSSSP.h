// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <climits>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include "GraphBenchmarkBase.h"
#include "BoostGraph.h"
#include "GunrockBenchmark.h"
#ifdef HAVE_GAPBS
   #include "GapbsKernels.h"
#endif
#include <TNL/Graphs/Algorithms/singleSourceShortestPath.h>
#include "SemiringSSSP.h"

namespace TNL::Benchmarks::Graphs {

/**
 * \brief SSSP benchmark following the GAP Benchmark Suite.
 *
 * The SSSP is run from 64 source vertices picked like in the GAP reference code, each trial is timed separately and
 * includes the allocation of the outputs. The weights of the GAP graphs are integers (1-255 for the graphs without
 * weights); the distances are verified for every trial by checking their optimality conditions.
 */
template< typename Real = double, typename Index = int >
class GraphBenchmarkSSSP : public GraphBenchmarkBase< Real, Index, GraphBenchmarkSSSP< Real, Index > >
{
public:
   using Base = GraphBenchmarkBase< Real, Index, GraphBenchmarkSSSP< Real, Index > >;
   using typename Base::HostDigraph;
   using typename Base::HostGraph;
   using typename Base::HostIndexVector;
   using typename Base::HostRealVector;
   using typename Base::IndexType;
   using typename Base::RealType;

   // Dijkstra (used here) requires non-negative edge weights; GraphBenchmarkBase
   // reads this flag to decide whether to make all weights positive after
   // loading the input graph.
   static constexpr bool needsNonNegativeWeights = true;
   static constexpr bool isSingleSource = true;
   static constexpr int defaultTrials = 64;

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      Base::configSetup( config );
      config.addDelimiter( "SSSP benchmark settings:" );
      config.addEntry< bool >( "with-semirings", "Run semiring-based BFS and SSSP benchmarks.", true );
      config.addEntry< bool >( "with-gunrock", "Run Gunrock benchmarks.", true );
      config.addEntry< bool >( "with-boost", "Run Boost benchmarks.", true );
      config.addEntry< bool >( "with-gapbs", "Run the GAP reference code (gapbs) benchmarks.", true );
      config.addEntry< int >(
         "gapbs-delta",
         "The delta parameter of the delta-stepping SSSP of gapbs, which GAP allows to set for each graph. "
         "0 means the value used by the GAP benchmark Makefile: 50000 for the road graph and 2 for the other graphs.",
         0 );
      config.addEntry< double >(
         "bitmap-threshold", "Frontier fraction below which top-down bitmap mode is used (0 = disabled).", 0.0 );
   }

   GraphBenchmarkSSSP( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters )
   {
      withBoost = this->parameters.template getParameter< bool >( "with-boost" );
      withGunrock = this->parameters.template getParameter< bool >( "with-gunrock" );
      withGapbs = this->parameters.template getParameter< bool >( "with-gapbs" );
      withSemirings = this->parameters.template getParameter< bool >( "with-semirings" );
      bitmapThreshold = this->parameters.template getParameter< double >( "bitmap-threshold" );
      maxBoostEdges = this->parameters.template getParameter< int >( "max-boost-edges" );
      gapbsDelta = this->parameters.template getParameter< int >( "gapbs-delta" );
   }

   static std::string
   problemName( bool directed )
   {
      return directed ? "SSSP dir" : "SSSP undir";
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
            runGapbsSSSP( this->directedInputFile, digraph, true, benchmark );
         if( this->runUndirected )
            runGapbsSSSP( this->undirectedInputFile, graph, false, benchmark );
      }
      if( withBoost ) {
         if( this->runDirected )
            runBoostSSSP( digraph, true, benchmark );
         if( this->runUndirected )
            runBoostSSSP( graph, false, benchmark );
      }
      if( withGunrock ) {
         if( this->runDirected )
            runGunrockSSSP( digraph, true, benchmark );
         if( this->runUndirected )
            runGunrockSSSP( graph, false, benchmark );
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
      benchmark.setDatasetSize(
         hostGraph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "solver", solver } );
      benchmark.setMetadataElement( { "problem", problemName( directed ) } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "" } );
      benchmark.setMetadataElement( { "mode", mode } );
   }

   // Returns a violation if a solver did not produce an output of the right size (e.g. because it threw an exception),
   // otherwise the verification of the distances. The distances that are not smaller than `unreached` are unreached.
   template< typename HostGraphType, typename Distance >
   static GraphVerificationResult
   checkDistances( const HostGraphType& hostGraph, Index source, const std::vector< Distance >& distances, Distance unreached )
   {
      if( distances.size() != static_cast< std::size_t >( hostGraph.getVertexCount() ) ) {
         GraphVerificationResult result;
         result.violations = 1;
         result.message = "the output has a wrong size";
         return result;
      }
      // The verification is done in the arithmetic of the benchmarked solver.
      std::vector< Distance > normalized( distances );
      for( auto& d : normalized )
         if( d >= unreached )
            d = -1;
      using ValueMatrix = TNL::Matrices::SparseMatrix< Distance, TNL::Devices::Host, Index >;
      if constexpr( std::is_same_v< ValueMatrix, std::decay_t< decltype( hostGraph.getAdjacencyMatrix() ) > > )
         return verifySSSP( hostGraph.getAdjacencyMatrix(), source, normalized.data() );
      else {
         // integer distances of gapbs: verify them in double, which represents the integer sums exactly
         std::vector< double > converted( normalized.begin(), normalized.end() );
         return verifySSSP( hostGraph.getAdjacencyMatrix(), source, converted.data() );
      }
   }

   [[nodiscard]] std::int32_t
   getGapbsDelta() const
   {
      if( gapbsDelta > 0 )
         return gapbsDelta;
      // the values of benchmark/bench.mk of gapbs
      const std::string file = this->parameters.template getParameter< TNL::String >( "input-file" );
      const std::string name = file.substr( file.find_last_of( '/' ) + 1 );
      return name.rfind( "road", 0 ) == 0 ? 50000 : 2;
   }

   template< typename HostGraphType >
   void
   runGapbsSSSP( const std::string& file, const HostGraphType& hostGraph, bool directed, TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_GAPBS
      if( ! this->gapInput ) {
         std::cout << "Skipping gapbs, it needs a .wsg input file of the GAP benchmark.\n";
         return;
      }
      try {
         Gapbs::SSSP gapbs( file );
         const std::int32_t delta = getGapbsDelta();
         std::cout << "gapbs SSSP uses delta = " << delta << '\n';
         setOtherSolverMetadata( benchmark, hostGraph, directed, "gapbs", "delta-stepping " + std::to_string( delta ) );
         std::vector< std::int64_t > distances;
         auto reset = [ & ]()
         {
            gapbs.release();
         };
         auto compute = [ & ]( Index source )
         {
            gapbs.run( source, delta );
         };
         auto check = [ & ]( Index source )
         {
            gapbs.getDistances( distances );
            return checkDistances( hostGraph, source, distances, std::numeric_limits< std::int64_t >::max() );
         };
         this->template timeSingleSourceTrials< TNL::Devices::Host >(
            benchmark, "gapbs", directed, "gapbs " + problemName( directed ), reset, compute, check );
      }
      catch( const std::exception& e ) {
         std::cerr << "gapbs SSSP failed: " << e.what() << '\n';
         this->errors++;
      }
#endif
   }

   template< typename HostGraphType >
   void
   runBoostSSSP( const HostGraphType& hostGraph, bool directed, TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_BOOST
      if( hostGraph.getAdjacencyMatrix().getNonzeroElementsCount() > maxBoostEdges ) {
         std::cout << "Skipping Boost, the graph has more than " << maxBoostEdges << " edges (see --max-boost-edges).\n";
         return;
      }
      BoostGraph< Index, Real, typename HostGraphType::GraphOrientation > boostGraph( hostGraph );
      setOtherSolverMetadata( benchmark, hostGraph, directed, "Boost", "N/A" );
      std::vector< Real > distances;
      auto reset = [ & ]()
      {
         std::vector< Real >().swap( distances );
      };
      auto compute = [ & ]( Index source )
      {
         boostGraph.singleSourceShortestPath( source, distances );
      };
      auto check = [ & ]( Index source )
      {
         // Boost marks the unreached vertices by the largest value
         return checkDistances( hostGraph, source, distances, std::numeric_limits< Real >::max() );
      };
      this->template timeSingleSourceTrials< TNL::Devices::Sequential >(
         benchmark, "Boost", directed, "Boost " + problemName( directed ), reset, compute, check );
#endif  // HAVE_BOOST
   }

   template< typename HostGraphType >
   void
   runGunrockSSSP( const HostGraphType& hostGraph, bool directed, TNL::Benchmarks::Benchmark& benchmark )
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
         auto gunrockGraphHolder = Gunrock::convertToGunrockGraph( hostGraph );
         setOtherSolverMetadata( benchmark, hostGraph, directed, "Gunrock", "N/A" );
         thrust::device_vector< Real > distances;
         thrust::device_vector< VertexType > predecessors;
         std::vector< Real > hostDistances;

         // Propagate exceptions (e.g. an unsupported load-balance strategy on
         // this platform) to the try/catch below instead of having them
         // silently swallowed inside benchmark.time().
         benchmark.setCatchExceptions( false );
         for( const auto& [ loadBalance, tag ] : Gunrock::loadBalanceConfigurations() ) {
            benchmark.setMetadataElement( { "launch cfg.", tag } );
            auto reset = [ & ]()
            {
               Gunrock::release( distances );
               Gunrock::release( predecessors );
            };
            auto compute = [ &, loadBalance = loadBalance ]( Index source )
            {
               Gunrock::singleSourceShortestPath( gunrockGraphHolder.graph, source, n, distances, predecessors, loadBalance );
            };
            auto check = [ & ]( Index source )
            {
               hostDistances.resize( distances.size() );
               thrust::copy( distances.begin(), distances.end(), hostDistances.begin() );
               return checkDistances( hostGraph, source, hostDistances, std::numeric_limits< Real >::max() );
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

   template< typename DeviceGraph, typename HostGraphType >
   void
   runTNLSSSP(
      DeviceGraph& g,
      const HostGraphType& hostGraph,
      bool directed,
      TNL::Benchmarks::Benchmark& benchmark,
      const TNL::String& segments )
   {
      using Device = typename DeviceGraph::DeviceType;
      using RealVector = TNL::Containers::Vector< Real, Device, Index >;

      if( min( hostGraph.getAdjacencyMatrix().getValues() ) < 0 ) {
         std::cout << "ERROR: Negative weights in the graph! Skipping SSSP benchmark.\n";
         this->errors++;
         return;
      }

      RealVector distances;
      HostRealVector hostDistances;
      std::vector< Real > hostDistancesStd;
      benchmark.setDatasetSize( g.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "problem", problemName( directed ) } );
      benchmark.setMetadataElement( { "kernel", segments } );

      auto reset = [ & ]()
      {
         // the SSSP allocates the output, so the allocation is measured
         distances.reset();
      };
      auto check = [ & ]( Index source )
      {
         hostDistances = distances;
         hostDistancesStd.assign( hostDistances.getData(), hostDistances.getData() + hostDistances.getSize() );
         return checkDistances( hostGraph, source, hostDistancesStd, std::numeric_limits< Real >::max() );
      };

      for( const auto& [ launchConfig, tag ] :
           Algorithms::Segments::traversingLaunchConfigurations( g.getAdjacencyMatrix().getSegments() ) )
      {
         benchmark.setMetadataElement( { "launch cfg.", tag } );
         std::vector< std::pair< const char*, double > > modes{ { "top-down compact", 0.0 } };
         if( bitmapThreshold > 0.0 )
            modes.emplace_back( "top-down bitmap", bitmapThreshold );
         for( const auto& [ modeTag, threshold ] : modes ) {
            benchmark.setMetadataElement( { "mode", modeTag } );
            auto compute = [ &, launchConfig = launchConfig, threshold = threshold ]( Index source )
            {
               TNL::Graphs::Algorithms::singleSourceShortestPath( g, source, distances, threshold, launchConfig );
            };
            this->template timeSingleSourceTrials< Device >(
               benchmark,
               "TNL",
               directed,
               "TNL " + problemName( directed ) + " " + std::string( segments ) + " " + tag + " " + modeTag,
               reset,
               compute,
               check );
         }
      }

      if( withSemirings && ! std::is_same_v< Device, TNL::Devices::Sequential > ) {
         benchmark.setMetadataElement( { "problem", "Semiring " + problemName( directed ) } );
         benchmark.setMetadataElement( { "launch cfg.", "Default" } );
         benchmark.setMetadataElement( { "mode", "N/A" } );
         auto compute = [ & ]( Index source )
         {
            distances.setSize( g.getVertexCount() );
            distances = std::numeric_limits< Real >::max();
            distances.setElement( source, 0 );
            semiringSSSP( g, source, distances );
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
         runTNLSSSP( digraph, *this->hostDigraphPointer, true, benchmark, segments );
      if( this->runUndirected )
         runTNLSSSP( graph, *this->hostGraphPointer, false, benchmark, segments );
   }

protected:
   bool withBoost;
   bool withGunrock;
   bool withGapbs;
   bool withSemirings;
   double bitmapThreshold;
   std::int64_t maxBoostEdges;
   int gapbsDelta;
};

}  // namespace TNL::Benchmarks::Graphs
