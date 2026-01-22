// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "GraphBase.h"

namespace TNL::Graphs {

/**
 * \brief View type for Graph class.
 *
 * GraphView provides a lightweight, non-owning view of a Graph object. It can be used to access
 * graph data without copying, making it efficient for passing graphs to functions or kernels.
 *
 */
template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
struct GraphView
: public GraphBase< Value,
                    Device,
                    Index,
                    Orientation,
                    std::conditional_t< std::is_const_v< Value >, typename AdjacencyMatrix::ConstMatrixType, AdjacencyMatrix > >
{
   //! \brief Type of the adjacency matrix.
   using AdjacencyMatrixType =
      std::conditional_t< std::is_const_v< Value >, typename AdjacencyMatrix::ConstMatrixType, AdjacencyMatrix >;

   //! \brief Type of the adjacency matrix view.
   using AdjacencyMatrixView = typename AdjacencyMatrixType::ViewType;

   //! \brief Type of constant view of the adjacency matrix.
   using ConstAdjacencyMatrixView = typename AdjacencyMatrixType::ConstViewType;

   //! \brief Type for indexing of the graph nodes.
   using IndexType = Index;

   //! \brief Type of device where the graph will be operating.
   using DeviceType = Device;

   //! \brief Type for weights of the graph edges.
   using ValueType = Value;

   //! \brief Type of the graph - directed or undirected.
   using GraphType = Orientation;

   //! \brief Type of view of the graph.
   using ViewType = GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >;

   //! \brief Type of constant view of the graph.
   using ConstViewType =
      GraphView< std::add_const_t< Value >, Device, Index, Orientation, typename AdjacencyMatrixType::ConstMatrixType >;

   //! \brief Type of the graph nodes view.
   using VertexView = GraphVertexView< AdjacencyMatrixView, Orientation >;

   //! \brief Type of constant graph nodes view.
   using ConstVertexView = typename VertexView::ConstVertexView;

   template< typename Value_ = Value,
             typename Device_ = Device,
             typename Index_ = Index,
             typename Orientation_ = Orientation,
             typename AdjacencyMatrix_ = AdjacencyMatrix >
   using Self = GraphView< Value_, Device_, Index_, Orientation_, AdjacencyMatrix_ >;

   using Base = GraphBase< Value, Device, Index, Orientation, AdjacencyMatrixType >;
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
