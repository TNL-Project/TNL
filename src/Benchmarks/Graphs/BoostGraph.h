// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <TNL/Graphs/Graph.h>

#ifdef HAVE_BOOST

   #include <boost/graph/adjacency_list.hpp>
   #include <boost/graph/breadth_first_search.hpp>
   #include <boost/graph/dijkstra_shortest_paths.hpp>
   #include <boost/graph/kruskal_min_spanning_tree.hpp>
   #include <boost/graph/graph_utility.hpp>

namespace TNL::Benchmarks::Graphs {

template< typename Value = double, typename GraphType = TNL::Graphs::DirectedGraph >
struct BoostAdjacencyList
{
   using type = boost::adjacency_list< boost::vecS,
                                       boost::vecS,
                                       boost::directedS,
                                       boost::no_property,
                                       boost::property< boost::edge_weight_t, Value > >;
};

template< typename Value >
struct BoostAdjacencyList< Value, TNL::Graphs::UndirectedGraph >
{
   using type = boost::adjacency_list< boost::vecS,
                                       boost::vecS,
                                       boost::undirectedS,
                                       boost::no_property,
                                       boost::property< boost::edge_weight_t, Value > >;
};

template< typename Index = int, typename Real = double, typename GraphType = TNL::Graphs::DirectedGraph >
struct BoostGraph
{
   using IndexType = Index;
   using RealType = Real;
   using AdjacencyList = typename BoostAdjacencyList< Real, GraphType >::type;
   using Vertex = typename boost::graph_traits< AdjacencyList >::vertex_descriptor;
   using Edge = typename boost::graph_traits< AdjacencyList >::edge_descriptor;

   static constexpr bool
   isDirected()
   {
      return std::is_same_v< GraphType, TNL::Graphs::DirectedGraph >;
   }
   static constexpr bool
   isUndirected()
   {
      return std::is_same_v< GraphType, TNL::Graphs::UndirectedGraph >;
   }

   BoostGraph() = default;

   template< typename TNLGraph >
   BoostGraph( const TNLGraph& graph )
   {
      static_assert( std::is_same_v< typename TNLGraph::GraphOrientation, GraphType >, "Graph types must match." );

      for( Index rowIdx = 0; rowIdx < graph.getVertexCount(); rowIdx++ ) {
         const auto row = graph.getAdjacencyMatrix().getRow( rowIdx );
         for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
            auto value = row.getValue( localIdx );
            if( value != 0.0 ) {
               add_edge( rowIdx, row.getColumnIndex( localIdx ), value, this->graph );
            }
         }
      }
   }

   void
   breadthFirstSearch( Index start, std::vector< Index >& distances )
   {
      distances.resize( boost::num_vertices( graph ) );
      auto recorder = record_distances( distances.data(), boost::on_tree_edge{} );
      auto visitor = boost::make_bfs_visitor( recorder );
      boost::breadth_first_search( graph, boost::vertex( start, graph ), boost::visitor( visitor ) );
   }

   void
   singleSourceShortestPath( Index start, std::vector< Real >& distances )
   {
      distances.resize( boost::num_vertices( graph ) );
      boost::dijkstra_shortest_paths(
         graph,
         start,
         boost::predecessor_map( boost::dummy_property_map() )
            .distance_map( boost::make_iterator_property_map( distances.begin(), get( boost::vertex_index, graph ) ) ) );
   }

   void
   minimumSpanningTree( std::vector< Edge >& spanning_tree )
   {
      boost::kruskal_minimum_spanning_tree( graph, std::back_inserter( spanning_tree ) );
   }

   [[nodiscard]] const AdjacencyList&
   getGraph() const
   {
      return graph;
   }

   void
   exportMst( const std::vector< Edge >& mst, const TNL::String& filename )
   {
      // Open file to write MST
      std::ofstream file( filename.getString() );

      // Write the edges of the Minimum Spanning Tree to file
      for( auto& edge : mst ) {
         int u = boost::source( edge, graph );
         int v = boost::target( edge, graph );
         int weight = boost::get( boost::edge_weight, graph, edge );
         file << u << " " << v << " " << weight << '\n';
      }

      // Close the file
      file.close();
   }

protected:
   AdjacencyList graph;
};

}  // namespace TNL::Benchmarks::Graphs

#endif
