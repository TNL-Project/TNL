// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBenchmarkBase.h"
#include "BoostGraph.h"
#include "GunrockBenchmark.h"
#include <TNL/Graphs/Algorithms/singleSourceShortestPath.h>
#include "SemiringSSSP.h"

namespace TNL::Benchmarks::Graphs {

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

   static void
   configSetup( TNL::Config::ConfigDescription& config )
   {
      Base::configSetup( config );
      config.addEntry< bool >( "with-semirings", "Run semiring-based BFS and SSSP benchmarks.", true );
      config.addEntry< bool >( "with-gunrock", "Run Gunrock benchmarks.", true );
      config.addEntry< bool >( "with-boost", "Run Boost benchmarks.", true );
   }

   GraphBenchmarkSSSP( const TNL::Config::ParameterContainer& parameters )
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
         runBoostBenchmarks( digraph, graph, smallestNode, largestNode, benchmark );
      if( withGunrock )
         runGunrockBenchmarks( digraph, graph, smallestNode, largestNode, benchmark );
   }

   void
   runBoostBenchmarks( const HostDigraph& digraph,
                       const HostGraph& graph,
                       IndexType smallestNode,
                       IndexType largestNode,
                       TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_BOOST
      BoostGraph< Index, Real, TNL::Graphs::DirectedGraph > boostDigraph( digraph );
      BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph > boostGraph( graph );
      benchmark.setMetadataElement( { "solver", "Boost" } );

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
#endif  // HAVE_BOOST
   }

   void
   runGunrockBenchmarks( const HostDigraph& digraph,
                         const HostGraph& graph,
                         IndexType smallestNode,
                         IndexType largestNode,
                         TNL::Benchmarks::Benchmark<>& benchmark )
   {
#ifdef HAVE_GUNROCK
      auto gunrockDigraph = GunrockBenchmark< Real, Index >::convertToGunrockGraph( digraph );
      auto gunrockGraph = GunrockBenchmark< Real, Index >::convertToGunrockGraph( graph );

      GunrockBenchmark< Real, Index > gunrockBenchmark;
      benchmark.setMetadataElement( { "solver", "Gunrock" } );

      // Benchmarking single-source shortest path of directed graph
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "problem", "SSSP dir" } );
      benchmark.setMetadataElement( { "format", "N/A" } );
      benchmark.setMetadataElement( { "threads mapping", "" } );

      std::vector< Real > ssspDistances( digraph.getVertexCount() );
      gunrockBenchmark.singleSourceShortestPath(
         benchmark, gunrockDigraph, largestNode, digraph.getVertexCount(), ssspDistances );
      HostRealVector gunrock_sssp_dist( ssspDistances );
      gunrock_sssp_dist.forAllElements(
         [] __cuda_callable__( Index i, Real & x )
         {
            x = x == std::numeric_limits< Real >::max() ? -1 : x;
         } );
      this->gunrockSSSPDistancesDirected = gunrock_sssp_dist;

   #ifdef HAVE_BOOST
      if( withBoost && this->boostSSSPDistancesDirected != this->gunrockSSSPDistancesDirected ) {
         std::cout << "SSSP distances of directed graph from Boost and Gunrock are not equal!" << std::endl;
         this->errors++;
      }
   #endif

      // Benchmarking single-source shortest path of undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "problem", "SSSP undir" } );
      benchmark.setMetadataElement( { "format", "N/A" } );
      benchmark.setMetadataElement( { "threads mapping", "" } );

      gunrockBenchmark.singleSourceShortestPath( benchmark, gunrockGraph, largestNode, graph.getVertexCount(), ssspDistances );
      gunrock_sssp_dist = ssspDistances;
      gunrock_sssp_dist.forAllElements(
         [] __cuda_callable__( Index i, Real & x )
         {
            x = x == std::numeric_limits< Real >::max() ? -1 : x;
         } );
      this->gunrockSSSPDistancesUndirected = gunrock_sssp_dist;

   #ifdef HAVE_BOOST
      if( withBoost && this->boostSSSPDistancesUndirected != this->gunrockSSSPDistancesUndirected ) {
         std::cout << "SSSP distances of undirected graph from Boost and Gunrock are not equal!" << std::endl;
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
      using RealVector = TNL::Containers::Vector< Real, Device, Index >;

      // Benchmarking single-source shortest paths with directed graph
      benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "problem", "SSSP dir" } );
      benchmark.setMetadataElement( { "format", segments } );

      for( const auto& launchEntry :
           Algorithms::Segments::traversingLaunchConfigurations( digraph.getAdjacencyMatrix().getSegments() ) )
      {
         const auto& launchConfig = launchEntry.first;
         const auto& tag = launchEntry.second;

         benchmark.setMetadataElement( { "threads mapping", tag } );

         RealVector ssspDistances( digraph.getVertexCount(), 0 );
         auto sssp_tnl_dir = [ &, launchConfig ]() mutable
         {
            TNL::Graphs::Algorithms::singleSourceShortestPath( digraph, largestNode, ssspDistances, launchConfig );
         };
         if( min( digraph.getAdjacencyMatrix().getValues() ) < 0 ) {
            std::cout << "ERROR: Negative weights in the graph! Skipping SSSP benchmark." << '\n';
            this->errors++;
         }
         else
            benchmark.time< Device >( device, sssp_tnl_dir );

#ifdef HAVE_BOOST
         if( withBoost && ssspDistances != this->boostSSSPDistancesDirected ) {
            std::cout << "SSSP distances of directed graph from Boost and TNL are not equal!" << '\n';
            this->errors++;
         }
#endif
#ifdef HAVE_GUNROCK
         if( withGunrock && ssspDistances != this->gunrockSSSPDistancesDirected ) {
            std::cout << "SSSP distances of directed graph from TNL and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
#endif
      }

      // Benchmarking single-source shortest paths with undirected graph
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "problem", "SSSP undir" } );
      benchmark.setMetadataElement( { "format", segments } );

      for( const auto& launchEntry :
           Algorithms::Segments::traversingLaunchConfigurations( graph.getAdjacencyMatrix().getSegments() ) )
      {
         const auto& launchConfig = launchEntry.first;
         const auto& tag = launchEntry.second;

         benchmark.setMetadataElement( { "threads mapping", tag } );

         RealVector ssspDistances( digraph.getVertexCount(), 0 );
         auto sssp_tnl_undir = [ &, launchConfig ]() mutable
         {
            TNL::Graphs::Algorithms::singleSourceShortestPath( graph, largestNode, ssspDistances, launchConfig );
         };
         benchmark.time< Device >( device, sssp_tnl_undir );

#ifdef HAVE_BOOST
         if( withBoost && ssspDistances != this->boostSSSPDistancesUndirected ) {
            std::cout << "SSSP distances of undirected graph from Boost and TNL are not equal!" << '\n';
            this->errors++;
         }
#endif
#ifdef HAVE_GUNROCK
         if( withGunrock && ssspDistances != this->gunrockSSSPDistancesUndirected ) {
            std::cout << "SSSP distances of undirected graph from TNL and Gunrock are not equal!" << std::endl;
            this->errors++;
         }
#endif
      }

      if( this->withSemirings && ! std::is_same_v< Device, TNL::Devices::Sequential > ) {
         // Benchmarking semiring-based SSSP with directed graph
         RealVector semiringSsspDistances( digraph.getVertexCount() );
         semiringSsspDistances = std::numeric_limits< Real >::max();
         semiringSsspDistances.setElement( largestNode, 0 );
         benchmark.setDatasetSize( digraph.getAdjacencyMatrix().getNonzeroElementsCount()
                                   * ( sizeof( Index ) + sizeof( Real ) ) );
         benchmark.setMetadataElement( { "problem", "Semiring SSSP dir" } );
         benchmark.setMetadataElement( { "format", segments } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         auto semiring_sssp_dir = [ & ]() mutable
         {
            semiringSSSP( digraph, largestNode, semiringSsspDistances );
         };
         if( min( digraph.getAdjacencyMatrix().getValues() ) < 0 ) {
            std::cout << "ERROR: Negative weights in the graph! Skipping semiring SSSP benchmark." << '\n';
            this->errors++;
         }
         else
            benchmark.time< Device >( device, semiring_sssp_dir );

         // Benchmarking semiring-based SSSP with undirected graph
         semiringSsspDistances = std::numeric_limits< Real >::max();
         semiringSsspDistances.setElement( largestNode, 0 );
         benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount()
                                   * ( sizeof( Index ) + sizeof( Real ) ) );
         benchmark.setMetadataElement( { "problem", "Semiring SSSP undir" } );
         benchmark.setMetadataElement( { "format", segments } );
         benchmark.setMetadataElement( { "threads mapping", "" } );

         auto semiring_sssp_undir = [ & ]() mutable
         {
            semiringSSSP( graph, largestNode, semiringSsspDistances );
         };
         benchmark.time< Device >( device, semiring_sssp_undir );
      }
   }

protected:
   // Reference solutions for comparison
   HostRealVector boostSSSPDistancesDirected, boostSSSPDistancesUndirected;
   HostRealVector gunrockSSSPDistancesDirected, gunrockSSSPDistancesUndirected;
   bool withBoost;
   bool withGunrock;
   bool withSemirings;
};

}  // namespace TNL::Benchmarks::Graphs
