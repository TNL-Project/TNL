// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <tuple>
#include <string>

#include <TNL/Containers/Vector.h>

namespace TNL {
namespace Algorithms {
namespace Graphs {

template< typename Matrix >
struct GraphReader
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   static void readEdgeList( const std::string& file_name, Matrix& matrix )
   {
      using Edge = std::pair< IndexType, IndexType >;
      std::ifstream file(file_name);
      IndexType nodes( 0 );
      std::map< Edge, RealType > edges;

      std::string line;
      while( getline(file, line) ) {
         if( line.empty() ) {
            continue;
         }

         if (line[0] == '#') {
            continue;
         }

         std::istringstream ss(line);
         int from_node, to_node;
         ss >> from_node >> to_node;
         nodes = std::max( nodes, std::max( from_node, to_node ) );
         edges.emplace( Edge( from_node, to_node ), 1.0 );
      }
      nodes++; // nodes are numbered from 0
      matrix.setDimensions( nodes, nodes );
      matrix.setElements( edges );
   }
};

} // namespace Graphs
} // namespace Algorithms
} // namespace TNL
