// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphOperations.h"

namespace TNL::Graphs {

template< typename Graph >
[[nodiscard]] typename Graph::ValueType
getTotalWeight( const Graph& graph )
{
   using ValueType = typename Graph::ValueType;
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;

   auto values_view = graph.getAdjacencyMatrix().getValues().getConstView();
   auto column_indexes_view = graph.getAdjacencyMatrix().getColumnIndexes().getConstView();
   ValueType w = Algorithms::reduce< DeviceType >(
      0,
      values_view.getSize(),
      [ = ] __cuda_callable__( IndexType i )
      {
         if( column_indexes_view[ i ] != Matrices::paddingIndex< IndexType > )
            return values_view[ i ];
         return (ValueType) 0;
      },
      TNL::Plus{} );
   if constexpr( Graph::isUndirected() && ! Graph::AdjacencyMatrixType::isSymmetric() )
      return 0.5 * w;
   return w;
}

}  // namespace TNL::Graphs
