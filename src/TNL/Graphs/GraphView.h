// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBase.h"

namespace TNL::Graphs {

template< typename AdjacencyMatrixView_, typename GraphType_ >
struct GraphView : public GraphBase< AdjacencyMatrixView_, GraphType_ >
{
   static_assert( Matrices::is_matrix_view_v< AdjacencyMatrixView_ > );

   using Base = GraphBase< AdjacencyMatrixView_, GraphType_ >;

   //! \brief Type of the adjacency matrix view.
   using AdjacencyMatrixView = AdjacencyMatrixView_;

   //! \brief Type of constant view of the adjacency matrix.
   using ConstAdjacencyMatrixView = decltype( std::declval< const AdjacencyMatrixView& >().getConstView() );

   //! \brief Type for indexing of the graph nodes.
   using IndexType = typename AdjacencyMatrixView::IndexType;

   //! \brief Type of device where the graph will be operating.
   using DeviceType = typename AdjacencyMatrixView::DeviceType;

   //! \brief Type for weights of the graph edges.
   using ValueType = typename AdjacencyMatrixView::RealType;

   //! \brief Type of the graph - directed or undirected.
   using GraphType = GraphType_;

   //! \brief Type of view of the graph.
   using ViewType = GraphView< AdjacencyMatrixView, GraphType_ >;

   //! \brief Type of constant view of the graph.
   using ConstViewType = GraphView< ConstAdjacencyMatrixView, GraphType_ >;

   //! \brief Type of the graph nodes view.
   using NodeView = GraphNodeView< AdjacencyMatrixView, GraphType_ >;

   //! \brief Type of constant graph nodes view.
   using ConstNodeView = typename NodeView::ConstNodeView;

   template< typename Matrix_ = AdjacencyMatrixView, typename GraphType__ = GraphType_ >
   using Self = Graph< Matrix_, GraphType__ >;

   using Base::isDirected;
   using Base::isUndirected;

   //! \brief Default constructor.
   __cuda_callable__
   GraphView() = default;

   //! \brief Constructor with sparse matrix view as an adjacency matrix.
   __cuda_callable__
   GraphView( AdjacencyMatrixView& adjacencyMatrixView );

   //! \brief Constructor with sparse matrix view as an adjacency matrix.
   __cuda_callable__
   GraphView( AdjacencyMatrixView&& adjacencyMatrixView );

   //!  \brief Method for rebinding (reinitialization) using another sparse matrix view as an adjacency matrix.
   void __cuda_callable__
   bind( AdjacencyMatrixView& adjacencyMatrixView );

   //! \brief Method for rebinding (reinitialization) using another sparse matrix view as an adjacency matrix.
   __cuda_callable__
   void
   bind( AdjacencyMatrixView&& adjacencyMatrixView );

   //! \brief Method for rebinding (reinitialization) using another graph view.
   void __cuda_callable__
   bind( GraphView& graphView );

   //! \brief Method for rebinding (reinitialization) using another graph view.
   void __cuda_callable__
   bind( GraphView&& graphView );

   /**
    * \brief Returns a modifiable view of the sparse matrix.
    *
    * \return sparse matrix view.
    */
   [[nodiscard]] __cuda_callable__
   ViewType
   getView();

   /**
    * \brief Returns a non-modifiable view of the sparse matrix.
    *
    * \return sparse matrix view.
    */
   [[nodiscard]] __cuda_callable__
   ConstViewType
   getConstView() const;
};

}  // namespace TNL::Graphs

#include "GraphView.hpp"
