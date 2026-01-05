// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>
#include <string>

#include <TNL/Matrices/MatrixReader.h>

namespace TNL::Graphs::Readers {

template< typename Graph >
struct MtxReader
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using MatrixType = typename Graph::AdjacencyMatrixType;

   static void
   read( const std::string& file_name, Graph& graph )
   {
      MatrixType adjacencyMatrix;
      Matrices::MatrixReader< MatrixType >::readMtx( file_name, adjacencyMatrix );
      if( adjacencyMatrix.getRows() != adjacencyMatrix.getColumns() )
         throw std::runtime_error( "Error in Graph MtxReader: adjacency matrix is not square!" );
      graph.setAdjacencyMatrix( std::move( adjacencyMatrix ) );
   }
};

}  // namespace TNL::Graphs::Readers
