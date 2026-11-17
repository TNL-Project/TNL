// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBenchmarkBase.h"
#include "BoostGraph.h"
#include <TNL/Graphs/Algorithms/experimental/minimumSpanningTree.h>
#include <TNL/Graphs/Algorithms/trees.h>

namespace TNL::Benchmarks::Graphs {

template< typename Real = double, typename Index = int >
class GraphBenchmarkMST : public GraphBenchmarkBase< Real, Index, GraphBenchmarkMST< Real, Index > >
{
public:
   using Base = GraphBenchmarkBase< Real, Index, GraphBenchmarkMST< Real, Index > >;
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
      config.addEntry< bool >( "with-boost", "Run Boost benchmarks.", true );
   }

   GraphBenchmarkMST( const TNL::Config::ParameterContainer& parameters, GraphBenchmarkProgress& progress )
   : Base( parameters, progress ),
     boostMSTTotalWeight( 0.0 )
   {
      withBoost = this->parameters.template getParameter< bool >( "with-boost" );
   }

   void
   runOtherBenchmarks(
      const HostDigraph& digraph,
      const HostGraph& graph,
      IndexType smallestNode,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
      // Boost is CPU-only, so with several MPI ranks splitting the GPU kernel sweep between
      // them, only rank 0 bothers with it (confirmed with the user).
      if( withBoost && this->rank == 0 )
         runBoostBenchmarks( digraph, graph, smallestNode, largestNode, benchmark );
      runGunrockBenchmarks( digraph, graph, smallestNode, largestNode, benchmark );
   }

   void
   runBoostBenchmarks(
      const HostDigraph& digraph,
      const HostGraph& graph,
      IndexType smallestNode,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
#ifdef HAVE_BOOST
      BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph > boostGraph( graph );
      benchmark.setMetadataElement( { "solver", "Boost" } );

      // Benchmarking minimum spanning tree
      benchmark.setMetadataElement( { "problem", "MST undir" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );
      benchmark.setMetadataElement( { "launch cfg.", "" } );

      using BoostEdge = typename BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph >::Edge;
      std::vector< BoostEdge > boostMstEdges;
      auto mst_boost = [ & ]() mutable
      {
         boostGraph.minimumSpanningTree( boostMstEdges );
      };
      // boostMstEdges/boostMSTTotalWeight are needed below as the reference for every TNL MST
      // comparison in this process run, even one resumed past this cell - so they must always be
      // (re)computed; only the timed measurement + log row is skipped if already done.
      const std::string precision = TNL::getType< Real >();
      if( this->progress.isDone(
             TNL::getType< TNL::Devices::Sequential >(), precision, "undirected", "Boost", "N/A", "N/A", "N/A" ) )
         mst_boost();
      else {
         benchmark.time< TNL::Devices::Sequential >( "Boost", mst_boost );
         this->progress.markDone(
            TNL::getType< TNL::Devices::Sequential >(), precision, "undirected", "Boost", "N/A", "N/A", "N/A" );
      }
      this->boostMSTTotalWeight = 0.0;
      for( auto& edge : boostMstEdges ) {
         Real weight = boost::get( boost::edge_weight, boostGraph.getGraph(), edge );
         this->boostMSTTotalWeight += weight;
      }
      //std::cout << "Boost MST total weight: " << boostMSTTotalWeight << '\n';
      auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
      boostGraph.exportMst( boostMstEdges, filename + "-boost-mst.txt" );

#endif  // HAVE_BOOST
   }

   void
   runGunrockBenchmarks(
      const HostDigraph& hostDigraph,
      const HostGraph& hostGraph,
      IndexType smallestNode,
      IndexType largestNode,
      TNL::Benchmarks::Benchmark& benchmark )
   {
      // Gunrock doesn't have MST implementation, so this is empty
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
      using Device = typename std::remove_reference_t< decltype( digraph ) >::DeviceType;
      using IndexVector = TNL::Containers::Vector< Index, Device, Index >;

      // This whole cell (measurement, output file, verification) is skipped as one unit on
      // resume - the verification was already done truthfully in the run that completed it.
      const std::string precision = TNL::getType< Real >();
      if( this->progress.isDone( TNL::getType< Device >(), precision, "undirected", "TNL", segments, "N/A", "N/A" ) )
         return;

      // Benchmarking minimum spanning tree
      Graph mstGraph;
      IndexVector roots;
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "problem", "MST undir" } );
      benchmark.setMetadataElement( { "kernel", "N/A" } );

      auto mst_tnl = [ & ]() mutable
      {
         TNL::Graphs::Algorithms::experimental::minimumSpanningTree( graph, mstGraph, roots );
      };
      benchmark.time< Device >( "TNL", mst_tnl );
      this->progress.markDone( TNL::getType< Device >(), precision, "undirected", "TNL", segments, "N/A", "N/A" );

      auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
      TNL::Graphs::Writers::EdgeListWriter< Graph >::write( filename + "-tnl-mst.txt", mstGraph );

      if( ! TNL::Graphs::Algorithms::isForest( mstGraph ) ) {
         std::cout << "ERROR: TNL MST is not a forest!\n";
         this->errors++;
      }

#ifdef HAVE_BOOST
      Real mstTotalWeight = TNL::Graphs::getTotalWeight( mstGraph );
      if( withBoost && mstTotalWeight != boostMSTTotalWeight ) {
         std::cout << "ERROR: Total weights of boost MST and TNL MST do not match!\n";
         std::cout << "Boost MST total weight: " << boostMSTTotalWeight << '\n';
         std::cout << "TNL MST total weight: " << mstTotalWeight << '\n';
         this->errors++;
      }
#endif
   }

protected:
   bool withBoost;
   // Reference solution for comparison
   Real boostMSTTotalWeight;
};

}  // namespace TNL::Benchmarks::Graphs
