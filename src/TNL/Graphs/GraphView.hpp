// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphView.h"

namespace TNL::Graphs {

template< typename AdjacencyMatrixView, typename GraphType_ >
__cuda_callable__
GraphView< AdjacencyMatrixView, GraphType_ >::GraphView( AdjacencyMatrixView& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( adjacencyMatrixView );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
__cuda_callable__
GraphView< AdjacencyMatrixView, GraphType_ >::GraphView( AdjacencyMatrixView&& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( std::forward< AdjacencyMatrixView >( adjacencyMatrixView ) );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
void __cuda_callable__
GraphView< AdjacencyMatrixView, GraphType_ >::bind( AdjacencyMatrixView& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( adjacencyMatrixView );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
void __cuda_callable__
GraphView< AdjacencyMatrixView, GraphType_ >::bind( AdjacencyMatrixView&& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( std::forward< AdjacencyMatrixView >( adjacencyMatrixView ) );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
void __cuda_callable__
GraphView< AdjacencyMatrixView, GraphType_ >::bind( GraphView& graphView )
{
   this->adjacencyMatrixView.bind( graphView.adjacencyMatrixView );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
void __cuda_callable__
GraphView< AdjacencyMatrixView, GraphType_ >::bind( GraphView&& graphView )
{
   this->adjacencyMatrixView.bind( std::forward< GraphView >( graphView ).adjacencyMatrixView );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphView< AdjacencyMatrixView, GraphType_ >::getView() -> ViewType
{
   return GraphView( this->adjacencyMatrixView );
}

template< typename AdjacencyMatrixView, typename GraphType_ >
[[nodiscard]] __cuda_callable__
auto
GraphView< AdjacencyMatrixView, GraphType_ >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->adjacencyMatrixView );
}

}  // namespace TNL::Graphs

#include "GraphView.hpp"
