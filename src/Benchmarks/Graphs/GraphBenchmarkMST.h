// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBenchmarkBase.h"
#include "BoostGraph.h"
#include <TNL/Graphs/Algorithms/minimumSpanningTree.h>
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

   GraphBenchmarkMST( const TNL::Config::ParameterContainer& parameters )
   : Base( parameters ),
     boostMSTTotalWeight( 0.0 )
   {
      withBoost = this->parameters.template getParameter< bool >( "with-boost" );
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
      BoostGraph< Index, Real, TNL::Graphs::UndirectedGraph > boostGraph( graph );
      benchmark.setMetadataElement( { "solver", "Boost" } );

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
      if( this->verbose > 0 )
         std::cout << "Boost MST total weight: " << boostMSTTotalWeight << std::endl;
      auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
      boostGraph.exportMst( boostMstEdges, filename + "-boost-mst.txt" );

#endif  // HAVE_BOOST
   }

   void
   runGunrockBenchmarks( const HostDigraph& hostDigraph,
                         const HostGraph& hostGraph,
                         IndexType smallestNode,
                         IndexType largestNode,
                         TNL::Benchmarks::Benchmark<>& benchmark )
   {
      // Gunrock doesn't have MST implementation, so this is empty
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

      // Benchmarking minimum spanning tree
      Graph mstGraph;
      IndexVector roots;
      benchmark.setDatasetSize( graph.getAdjacencyMatrix().getNonzeroElementsCount() * ( sizeof( Index ) + sizeof( Real ) ) );
      benchmark.setMetadataElement( { "problem", "MST undir" } );
      benchmark.setMetadataElement( { "format", segments } );

      auto mst_tnl = [ & ]() mutable
      {
         TNL::Graphs::Algorithms::minimumSpanningTree( graph, mstGraph, roots );
      };
      benchmark.time< Device >( device, mst_tnl );

      auto filename = this->parameters.template getParameter< TNL::String >( "input-file" );
      TNL::Graphs::Writers::EdgeListWriter< Graph >::write( filename + "-tnl-mst.txt", mstGraph );

      if( ! TNL::Graphs::Algorithms::isForest( mstGraph ) ) {
         std::cout << "ERROR: TNL MST is not a forest!" << '\n';
         this->errors++;
      }

#ifdef HAVE_BOOST
      Real mstTotalWeight = TNL::Graphs::getTotalWeight( mstGraph );
      if( withBoost && mstTotalWeight != boostMSTTotalWeight ) {
         std::cout << "ERROR: Total weights of boost MST and TNL MST do not match!" << '\n';
         std::cout << "Boost MST total weight: " << boostMSTTotalWeight << std::endl;
         std::cout << "TNL MST total weight: " << mstTotalWeight << std::endl;
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
