// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphView.h"

namespace TNL::Graphs {

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
__cuda_callable__
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::GraphView( AdjacencyMatrixView& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( adjacencyMatrixView );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
__cuda_callable__
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::GraphView( AdjacencyMatrixView&& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( std::forward< AdjacencyMatrixView >( adjacencyMatrixView ) );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
void __cuda_callable__
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::bind( AdjacencyMatrixView& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( adjacencyMatrixView );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
void __cuda_callable__
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::bind( AdjacencyMatrixView&& adjacencyMatrixView )
{
   this->adjacencyMatrixView.bind( std::forward< AdjacencyMatrixView >( adjacencyMatrixView ) );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
void __cuda_callable__
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::bind( GraphView& graphView )
{
   this->adjacencyMatrixView.bind( graphView.adjacencyMatrixView );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
void __cuda_callable__
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::bind( GraphView&& graphView )
{
   this->adjacencyMatrixView.bind( std::forward< GraphView >( graphView ).adjacencyMatrixView );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::getView() -> ViewType
{
   return GraphView( this->adjacencyMatrixView );
}

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
[[nodiscard]] __cuda_callable__
auto
GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >::getConstView() const -> ConstViewType
{
   return ConstViewType( this->adjacencyMatrixView.getConstView() );
}

}  // namespace TNL::Graphs

#include "GraphView.hpp"
