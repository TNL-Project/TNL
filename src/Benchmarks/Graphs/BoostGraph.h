#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/breadth_first_search.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <boost/graph/graph_utility.hpp>

using namespace std;
using namespace boost;

// Custom visitor to update the distance map
struct bfs_distance_visitor : public boost::default_bfs_visitor {
  bfs_distance_visitor(std::vector<int>& distances) : distances_(distances) {}

  template <typename Edge, typename Graph>
  void tree_edge(Edge e, const Graph& g) const {
    typename graph_traits<Graph>::vertex_descriptor
        u = source(e, g), v = target(e, g);
    distances_[v] = distances_[u] + 1;
  }

  std::vector<int>& distances_;
};


template< typename Index = int,
          typename Real = double >
struct BoostGraph
{
   using IndexType = Index;
   using RealType = Real;
   using AdjacencyList = boost::adjacency_list< boost::vecS,
                                                boost::vecS,
                                                boost::directedS,
                                                boost::no_property,
                                                boost::property<boost::edge_weight_t, Real> >;
   using Vertex = typename boost::graph_traits< AdjacencyList >::vertex_descriptor;
   using Edge = typename boost::graph_traits< AdjacencyList >::edge_descriptor;

   BoostGraph(){}

   template< typename TNLDigraph >
   BoostGraph( const TNLDigraph& digraph )
   {
      for( Index rowIdx = 0; rowIdx < digraph.getNodeCount(); rowIdx++ )
      {
         const auto row = digraph.getAdjacencyMatrix().getRow( rowIdx );
         for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ )
         {
            if( row.getValue( localIdx ) != 0.0 )
               add_edge( rowIdx, row.getColumnIndex( localIdx ), 1.0, graph );
         }
      }
   }

   void breadthFirstSearch( Index start, std::vector< Index >& distances )
   {
      // Define a distance map to store distances from the source vertex
      distances.resize( boost::num_vertices(graph) );
      std::fill( distances.begin(), distances.end(), -1 );

      // Initialize the distance map for the source vertex
      distances[0] = 0;

      ::bfs_distance_visitor distance_visitor( distances );
      boost::breadth_first_search(graph, boost::vertex(0, graph), boost::visitor( distance_visitor ) );
   }

   void singleSourceShortestPath( Index start, std::vector< Real >& distances )
   {
      // Define a distance map to store distances from the source vertex
      distances.resize( boost::num_vertices(graph) );

      // Compute the shortest paths from the source vertex (vertex 0) using Dijkstra's algorithm
      Vertex source_vertex = 0;
      boost::dijkstra_shortest_paths( graph, source_vertex,
                                      boost::predecessor_map(boost::dummy_property_map())
                                          .distance_map(boost::make_iterator_property_map(
                                           distances.begin(), get(boost::vertex_index, graph))));

      // Print the distances from the source vertex
      //for (size_t i = 0; i < distances.size(); ++i) {
      //   std::cout << "Distance from vertex " << start << " to vertex " << i
      //              << ": " << distances[i] << std::endl;
      //}
   }

   void minimumSpanningTree( std::vector< Edge >& spanning_tree )
   {
      //boost::kruskal_minimum_spanning_tree(graph, std::back_inserter(spanning_tree));
   }
protected:
   AdjacencyList graph;
};
