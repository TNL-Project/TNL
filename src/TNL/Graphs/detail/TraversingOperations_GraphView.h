// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/traverse.h>
#include "../GraphView.h"
#include "TraversingOperations.h"

namespace TNL::Graphs::detail {

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
struct TraversingOperations< GraphView< Value, Device, Index, Orientation, AdjacencyMatrix > >
{
   using GraphViewType = GraphView< Value, Device, Index, Orientation, AdjacencyMatrix >;
   using ConstGraphViewType = typename std::remove_cv_t<
      typename std::remove_reference_t< decltype( std::declval< GraphViewType >().getConstView() ) > >;
   using ValueType = typename GraphViewType::ValueType;
   using DeviceType = typename GraphViewType::DeviceType;
   using IndexType = typename GraphViewType::IndexType;
   using VertexView = typename GraphViewType::VertexView;
   using ConstVertexView = typename ConstGraphViewType::ConstVertexView;
   using AdjacencyMatrixType = AdjacencyMatrix;
   using RowViewType = typename AdjacencyMatrix::RowView;
   using ConstRowViewType = typename AdjacencyMatrixType::ConstRowView;

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forEdges( GraphViewType& graph,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::forElements( graph.getAdjacencyMatrixView(), begin, end, function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forEdges( const ConstGraphViewType& graph,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::forElements( graph.getAdjacencyMatrixView(), begin, end, function, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forEdges( GraphViewType& graph,
             const Array& rowIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::forElements( graph.getAdjacencyMatrixView(), rowIndexes.getConstView( begin, end ), function, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forEdges( const ConstGraphViewType& graph,
             const Array& rowIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::forElements( graph.getAdjacencyMatrixView(), rowIndexes.getConstView( begin, end ), function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forEdgesIf( GraphViewType& graph,
               IndexBegin begin,
               IndexEnd end,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::forElementsIf( graph.getAdjacencyMatrixView(), begin, end, condition, function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forEdgesIf( const ConstGraphViewType& graph,
               IndexBegin begin,
               IndexEnd end,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::forElementsIf( graph.getAdjacencyMatrixView(), begin, end, condition, function, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forVertices( GraphViewType& graph,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( RowViewType & rowView ) mutable
      {
         VertexView vertexView( rowView );
         function( vertexView );
      };
      Matrices::forRows( graph.getAdjacencyMatrixView(), begin, end, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forVertices( const ConstGraphViewType& graph,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( const RowViewType& rowView ) mutable
      {
         function( ConstVertexView( rowView ) );
      };
      Matrices::forRows( graph.getAdjacencyMatrixView(), begin, end, f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forVertices( GraphViewType& graph,
                const Array& rowIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( RowViewType & rowView ) mutable
      {
         VertexView vertexView( rowView );
         function( vertexView );
      };
      Matrices::forRows( graph.getAdjacencyMatrixView(), rowIndexes.getConstView( begin, end ), f, launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Function >
   static void
   forVertices( const ConstGraphViewType& graph,
                const Array& rowIndexes,
                IndexBegin begin,
                IndexEnd end,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( const RowViewType& rowView ) mutable
      {
         function( ConstVertexView( rowView ) );
      };
      Matrices::forRows( graph.getAdjacencyMatrixView(), rowIndexes.getConstView( begin, end ), f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename VertexCondition, typename Function >
   static void
   forVerticesIf( GraphViewType& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  VertexCondition&& rowCondition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( RowViewType & rowView ) mutable
      {
         VertexView vertexView( rowView );
         function( vertexView );
      };
      Matrices::forRowsIf( graph.getAdjacencyMatrixView(), begin, end, rowCondition, f, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename VertexCondition, typename Function >
   static void
   forVerticesIf( const ConstGraphViewType& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  VertexCondition&& rowCondition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      auto f = [ = ] __cuda_callable__( const RowViewType& rowView ) mutable
      {
         function( ConstVertexView( rowView ) );
      };
      Matrices::forRowsIf( graph.getAdjacencyMatrixView(), begin, end, rowCondition, f, launchConfig );
   }
};

}  //namespace TNL::Graphs::detail
