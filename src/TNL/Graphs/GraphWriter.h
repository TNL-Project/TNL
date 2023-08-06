// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixBase.h>

namespace TNL::Graphs {

template< typename Graph >
struct GraphWriter
{
   using GraphType = Graph;
   using MatrixType = typename GraphType::MatrixType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using HostMatrixType = typename MatrixType::template Self< ValueType, Devices::Host >;

   static void
   writeEdgeList( std::ostream& str, const Graph& graph )
   {
      if constexpr( ! std::is_same_v< DeviceType, Devices::Host > && ! std::is_same_v< DeviceType, Devices::Host > ) {
         HostMatrixType hostMatrix;
         hostMatrix = graph.getAdjacencyMatrix();
         writeEdgeList( str, hostMatrix );
      }
      else
         writeEdgeList( str, graph.getAdjacencyMatrix() );
   }

   static void
   writeEdgeList( const TNL::String& fileName, const Graph& graph )
   {
      std::ofstream file( fileName.getString() );
      writeEdgeList( file, graph );
   }

protected:
   template< typename Matrix >
   static void
   writeEdgeList( std::ostream& str, const Matrix& matrix )
   {
      for( IndexType rowIdx = 0; rowIdx < matrix.getRows(); rowIdx++ ) {
         auto row = matrix.getRow( rowIdx );
         for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
            auto columnIdx = row.getColumnIndex( localIdx );
            if( columnIdx != Matrices::paddingIndex< IndexType > ) {
               str << rowIdx << " " << columnIdx << " " << row.getValue( localIdx ) << std::endl;
            }
         }
      }
   }
};

}  // namespace TNL::Graphs
