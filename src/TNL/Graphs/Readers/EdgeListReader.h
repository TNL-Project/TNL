// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>
#include <string>

namespace TNL::Graphs::Readers {

/**
 * \brief Reader for graphs in edge list format.
 *
 * This reader reads a graph from a file in edge list format, where each line
 * represents an edge between two vertices, optionally with a weight. Lines starting
 * with '#' are treated as comments and ignored.
 *
 * Example of weighted edge list format:
 *
 * ```
 * # This is a comment
 * 0 1 2.5
 * 1 2 1.0
 * 2 0 3.0
 * ```
 *
 * Example of unweighted edge list format:
 *
 * ```
 * # This is a comment
 * 0 1
 * 1 2
 * 2 0
 * ```
 *
 * \tparam Graph The type of the graph to be read.
 */
template< typename Graph >
struct EdgeListReader
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   /**
    * \brief Reads a graph from an edge list file.
    *
    * \param file_name is the name of the file containing the edge list.
    * \param graph is the graph object to be populated.
    */
   static void
   read( const std::string& file_name, Graph& graph )
   {
      using Edge = std::pair< IndexType, IndexType >;
      std::ifstream file( file_name );
      IndexType vertices( 0 );
      std::map< Edge, ValueType > edges;

      std::string line;
      while( getline( file, line ) ) {
         if( line.empty() ) {
            continue;
         }

         if( line[ 0 ] == '#' ) {
            continue;
         }

         std::istringstream ss( line );
         int from_node;
         int to_node;
         ss >> from_node >> to_node;
         vertices = std::max( vertices, std::max( from_node, to_node ) );
         ValueType weight = 1.0;
         if( ! ss.eof() ) {
            ss >> weight;
         }
         edges.emplace( Edge( from_node, to_node ), weight );
      }
      vertices++;  // vertices are numbered from 0
      graph.setVertexCount( vertices );
      graph.setEdges( edges );
   }
};

}  // namespace TNL::Graphs::Readers
