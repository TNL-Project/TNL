// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/reduce.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Matrices/reduce.h>
#include "ReductionOperations.h"

namespace TNL::Graphs::detail {

template< typename Value, typename Device, typename Index, typename Orientation, typename AdjacencyMatrix >
struct ReductionOperations< GraphView< Value, Device, Index, Orientation, AdjacencyMatrix > >
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

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVertices( GraphViewType& graph,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRows( graph.getAdjacencyMatrixView(), begin, end, fetch, reduction, store, identity, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVertices( const ConstGraphViewType& graph,
                   IndexBegin begin,
                   IndexEnd end,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRows( graph.getAdjacencyMatrixView(), begin, end, fetch, reduction, store, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVertices( GraphViewType& graph,
                   const Array& vertexIndices,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRows( graph.getAdjacencyMatrixView(), vertexIndices, fetch, reduction, store, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVertices( const ConstGraphViewType& graph,
                   const Array& vertexIndices,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRows( graph.getAdjacencyMatrixView(), vertexIndices, fetch, reduction, store, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store >
   static void
   reduceVertices( GraphViewType& graph,
                   const Array& vertexIndices,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      using FetchValue = decltype( fetch( IndexType(), IndexType(), ValueType() ) );
      const FetchValue identity = reduction.template getIdentity< FetchValue >();
      reduceVertices( graph,
                      vertexIndices,
                      std::forward< Fetch >( fetch ),
                      reduction,
                      std::forward< Store >( store ),
                      identity,
                      launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store >
   static void
   reduceVertices( const ConstGraphViewType& graph,
                   const Array& vertexIndices,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      using FetchValue = decltype( fetch( IndexType(), IndexType(), ValueType() ) );
      const FetchValue identity = reduction.template getIdentity< FetchValue >();
      reduceVertices( graph,
                      vertexIndices,
                      std::forward< Fetch >( fetch ),
                      reduction,
                      std::forward< Store >( store ),
                      identity,
                      launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceVerticesIf( GraphViewType& graph,
                     IndexBegin begin,
                     IndexEnd end,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     const FetchValue& identity,
                     Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      return Matrices::reduceRowsIf( graph.getAdjacencyMatrixView(),
                                     begin,
                                     end,
                                     std::forward< Condition >( condition ),
                                     fetch,
                                     reduction,
                                     store,
                                     identity,
                                     launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceVerticesIf( const ConstGraphViewType& graph,
                     IndexBegin begin,
                     IndexEnd end,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     const FetchValue& identity,
                     Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      return Matrices::reduceRowsIf( graph.getAdjacencyMatrixView(),
                                     begin,
                                     end,
                                     std::forward< Condition >( condition ),
                                     fetch,
                                     reduction,
                                     store,
                                     identity,
                                     launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVerticesWithArgument( GraphViewType& graph,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               const FetchValue& identity,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRowsWithArgument(
         graph.getAdjacencyMatrixView(), begin, end, fetch, reduction, store, identity, launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVerticesWithArgument( const ConstGraphViewType& graph,
                               IndexBegin begin,
                               IndexEnd end,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               const FetchValue& identity,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRowsWithArgument(
         graph.getAdjacencyMatrixView(), begin, end, fetch, reduction, store, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVerticesWithArgument( GraphViewType& graph,
                               const Array& vertexIndices,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               const FetchValue& identity,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRowsWithArgument(
         graph.getAdjacencyMatrixView(), vertexIndices, fetch, reduction, store, identity, launchConfig );
   }

   template< typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue >
   static void
   reduceVerticesWithArgument( const ConstGraphViewType& graph,
                               const Array& vertexIndices,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               const FetchValue& identity,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      Matrices::reduceRowsWithArgument(
         graph.getAdjacencyMatrixView(), vertexIndices, fetch, reduction, store, identity, launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceVerticesWithArgumentIf( GraphViewType& graph,
                                 IndexBegin begin,
                                 IndexEnd end,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 Store&& store,
                                 const FetchValue& identity,
                                 Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      return Matrices::reduceRowsWithArgumentIf( graph.getAdjacencyMatrixView(),
                                                 begin,
                                                 end,
                                                 std::forward< Condition >( condition ),
                                                 fetch,
                                                 reduction,
                                                 store,
                                                 identity,
                                                 launchConfig );
   }

   template< typename IndexBegin,
             typename IndexEnd,
             typename Condition,
             typename Fetch,
             typename Reduction,
             typename Store,
             typename FetchValue >
   static IndexType
   reduceVerticesWithArgumentIf( const ConstGraphViewType& graph,
                                 IndexBegin begin,
                                 IndexEnd end,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 Store&& store,
                                 const FetchValue& identity,
                                 Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      return Matrices::reduceRowsWithArgumentIf( graph.getAdjacencyMatrixView(),
                                                 begin,
                                                 end,
                                                 std::forward< Condition >( condition ),
                                                 fetch,
                                                 reduction,
                                                 store,
                                                 identity,
                                                 launchConfig );
   }
};
}  //namespace TNL::Graphs::detail
