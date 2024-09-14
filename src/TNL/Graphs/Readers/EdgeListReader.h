// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>
#include <string>

namespace TNL::Graphs::Readers {

template< typename Graph >
struct EdgeListReader
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   static void
   read( const std::string& file_name, Graph& graph )
   {
      using Edge = std::pair< IndexType, IndexType >;
      std::ifstream file( file_name );
      IndexType nodes( 0 );
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
         nodes = std::max( nodes, std::max( from_node, to_node ) );
         ValueType weight = 1.0;
         if( ! ss.eof() ) {
            ss >> weight;
         }
         edges.emplace( Edge( from_node, to_node ), weight );
      }
      nodes++;  // nodes are numbered from 0
      graph.setNodeCount( nodes );
      graph.setEdges( edges );
   }
};

}  // namespace TNL::Graphs::Readers
