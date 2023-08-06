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

// Custom visitor to update the distance map
struct bfs_distance_visitor : public boost::default_bfs_visitor
{
   bfs_distance_visitor( std::vector< int >& distances ) : distances_( distances ) {}

   template< typename Edge, typename Graph >
   void
   tree_edge( Edge e, const Graph& g ) const
   {
      typename boost::graph_traits< Graph >::vertex_descriptor u = source( e, g ), v = target( e, g );
      distances_[ v ] = distances_[ u ] + 1;
   }

   std::vector< int >& distances_;
};

template< typename Value = double, TNL::Graphs::GraphTypes GraphType = TNL::Graphs::GraphTypes::Directed >
struct BoostAdjacencyList
{
   using type = boost::adjacency_list< boost::vecS,
                                       boost::vecS,
                                       boost::directedS,
                                       boost::no_property,
                                       boost::property< boost::edge_weight_t, Value > >;
};

template< typename Value >
struct BoostAdjacencyList< Value, TNL::Graphs::GraphTypes::Undirected >
{
   using type = boost::adjacency_list< boost::vecS,
                                       boost::vecS,
                                       boost::undirectedS,
                                       boost::no_property,
                                       boost::property< boost::edge_weight_t, Value > >;
};

template< typename Index = int, typename Real = double, TNL::Graphs::GraphTypes GraphType = TNL::Graphs::GraphTypes::Directed >
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
      return ( GraphType == TNL::Graphs::GraphTypes::Directed );
   }
   static constexpr bool
   isUndirected()
   {
      return ( GraphType == TNL::Graphs::GraphTypes::Undirected );
   }

   BoostGraph() {}

   template< typename TNLGraph >
   BoostGraph( const TNLGraph& graph )
   {
      static_assert( TNLGraph::getGraphType() == GraphType, "Graph types must match." );

      for( Index rowIdx = 0; rowIdx < graph.getNodeCount(); rowIdx++ ) {
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
      // Define a distance map to store distances from the source vertex
      distances.resize( boost::num_vertices( graph ) );
      std::fill( distances.begin(), distances.end(), -1 );

      // Initialize the distance map for the source vertex
      distances[ start ] = 0;

      bfs_distance_visitor distance_visitor( distances );
      boost::breadth_first_search( graph, boost::vertex( 0, graph ), boost::visitor( distance_visitor ) );
   }

   void
   singleSourceShortestPath( Index start, std::vector< Real >& distances )
   {
      // Define a distance map to store distances from the source vertex
      distances.resize( boost::num_vertices( graph ) );

      // Compute the shortest paths from the source vertex (vertex 0) using Dijkstra's algorithm
      Vertex source_vertex = 0;
      boost::dijkstra_shortest_paths(
         graph,
         source_vertex,
         boost::predecessor_map( boost::dummy_property_map() )
            .distance_map( boost::make_iterator_property_map( distances.begin(), get( boost::vertex_index, graph ) ) ) );
   }

   void
   minimumSpanningTree( std::vector< Edge >& spanning_tree )
   {
      boost::kruskal_minimum_spanning_tree( graph, std::back_inserter( spanning_tree ) );
   }

   const AdjacencyList&
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
         file << u << " " << v << " " << weight << std::endl;
      }

      // Close the file
      file.close();
   }

protected:
   AdjacencyList graph;
};

}  // namespace TNL::Benchmarks::Graphs

#endif
