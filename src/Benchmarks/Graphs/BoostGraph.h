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
   using EdgeWeightProperty = boost::property<boost::edge_weight_t, Real > ;
   using AdjacencyList = boost::adjacency_list< boost::vecS, boost::vecS, boost::directedS, boost::no_property, EdgeWeightProperty >;
   using IndexType = Index;

   BoostGraph(){}

   template< typename TNLMatrix >
   BoostGraph( const TNLMatrix& m )
   {
      TNL_ASSERT_TRUE( m.getRows() == m.getColumns(), "Adjacency matrix must be square matrix." );
      for( Index rowIdx = 0; rowIdx < m.getRows(); rowIdx++ )
      {
         const auto row = m.getRow( rowIdx );
         for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ )
         {
            if( row.getValue( localIdx ) != 0.0 )
            {
               add_edge( rowIdx, row.getColumnIndex( localIdx ), graph );
               //std::cout << "Adding edge: " << rowIdx << " -> " << row.getColumnIndex( localIdx ) << std::endl;
               //std::cout << rowIdx << " " << row.getColumnIndex( localIdx ) << std::endl;
            }
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
      //boost::Vertex source_vertex = 0;
      boost::dijkstra_shortest_paths( graph, start,
                                      boost::predecessor_map(boost::dummy_property_map())
                                          .distance_map(boost::make_iterator_property_map(
                                           distances.begin(), get(boost::vertex_index, graph))));

      // Print the distances from the source vertex
      for (size_t i = 0; i < distances.size(); ++i) {
         std::cout << "Distance from vertex " << start << " to vertex " << i
                     << ": " << distances[i] << std::endl;
      }
   }

protected:
   AdjacencyList graph;
};
