// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <fstream>
#include <sstream>
#include <string>

#include <TNL/Matrices/MatrixReader.h>

namespace TNL::Graphs::Readers {

/**
 * \brief Reader for graphs in Matrix Market format.
 *
 * This reader uses \ref TNL::Matrices::MatrixReader to read the adjacency matrix of the graph
 * from a file in Matrix Market format (.mtx) and sets it as the adjacency matrix of the provided graph.
 *
 * \tparam Graph The type of the graph to be read.
 */
template< typename Graph >
struct MtxReader
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using MatrixType = typename Graph::AdjacencyMatrixType;

   /**
    * \brief Reads a graph from a Matrix Market file.
    *
    * \param file_name is the name of the file containing the adjacency matrix in Matrix Market format.
    * \param graph is the graph object to be populated.
    */
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
