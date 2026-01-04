// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "reduce.h"
#include "detail/ReductionOperations.h"

namespace TNL::Graphs {
template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVertices( Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceVertices( graph,
                   0,
                   graph.getVertexCount(),
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   identity,
                   launchConfig );
}

template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVertices( const Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   reduceVertices( graph,
                   (IndexType) 0,
                   graph.getVertexCount(),
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   identity,
                   launchConfig );
}

template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVertices( Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVertices( graph,
                   (IndexType) 0,
                   graph.getVertexCount(),
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   Reduction::template getIdentity< FetchValue >(),
                   launchConfig );
}

template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVertices( const Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Graph::IndexType;
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVertices( graph,
                   (IndexType) 0,
                   graph.getVertexCount(),
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   Reduction::template getIdentity< FetchValue >(),
                   launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceVertices( Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::ReductionOperations< typename Graph::ViewType >::reduceVertices( graph_view,
                                                                            begin,
                                                                            end,
                                                                            std::forward< Fetch >( fetch ),
                                                                            reduction,
                                                                            std::forward< Store >( store ),
                                                                            identity,
                                                                            launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceVertices( const Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Graph::ConstViewType >::reduceVertices( graph.getConstView(),
                                                                                 begin,
                                                                                 end,
                                                                                 std::forward< Fetch >( fetch ),
                                                                                 reduction,
                                                                                 std::forward< Store >( store ),
                                                                                 identity,
                                                                                 launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceVertices( Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVertices( graph,
                   begin,
                   end,
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   Reduction::template getIdentity< FetchValue >(),
                   launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceVertices( const Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVertices( graph,
                   begin,
                   end,
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   Reduction::template getIdentity< FetchValue >(),
                   launchConfig );
}

template< typename Graph, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceVertices( Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::ReductionOperations< typename Graph::ViewType >::reduceVertices( graph_view,
                                                                            vertexIndexes,
                                                                            std::forward< Fetch >( fetch ),
                                                                            reduction,
                                                                            std::forward< Store >( store ),
                                                                            identity,
                                                                            launchConfig );
}

template< typename Graph, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceVertices( const Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Graph::ConstViewType >::reduceVertices( graph.getConstView(),
                                                                                 vertexIndexes,
                                                                                 std::forward< Fetch >( fetch ),
                                                                                 reduction,
                                                                                 std::forward< Store >( store ),
                                                                                 identity,
                                                                                 launchConfig );
}

template< typename Graph, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceVertices( Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVertices( graph,
                   vertexIndexes,
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   Reduction::template getIdentity< FetchValue >(),
                   launchConfig );
}

template< typename Graph, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceVertices( const Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVertices( graph.getConstView(),
                   vertexIndexes,
                   0,
                   vertexIndexes.getSize(),
                   std::forward< Fetch >( fetch ),
                   reduction,
                   std::forward< Store >( store ),
                   Reduction::template getIdentity< FetchValue >(),
                   launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Graph::IndexType
reduceAllVerticesIf( Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     const FetchValue& identity,
                     Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceVerticesIf( graph,
                            (decltype( graph.getVertexCount() )) 0,
                            graph.getVertexCount(),
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            reduction,
                            std::forward< Store >( store ),
                            identity,
                            launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Graph::IndexType
reduceAllVerticesIf( const Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     const FetchValue& identity,
                     Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceVerticesIf( graph,
                            (decltype( graph.getVertexCount() )) 0,
                            graph.getVertexCount(),
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            reduction,
                            std::forward< Store >( store ),
                            identity,
                            launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesIf( Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesIf( graph,
                            (decltype( graph.getVertexCount() )) 0,
                            graph.getVertexCount(),
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            reduction,
                            std::forward< Store >( store ),
                            Reduction::template getIdentity< FetchValue >(),
                            launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesIf( const Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesIf( graph,
                            (decltype( graph.getVertexCount() )) 0,
                            graph.getVertexCount(),
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            reduction,
                            std::forward< Store >( store ),
                            Reduction::template getIdentity< FetchValue >(),
                            launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Graph::IndexType
reduceVerticesIf( Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  const FetchValue& identity,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   return detail::ReductionOperations< typename Graph::ViewType >::reduceVerticesIf( graph_view,
                                                                                     begin,
                                                                                     end,
                                                                                     std::forward< Condition >( condition ),
                                                                                     std::forward< Fetch >( fetch ),
                                                                                     reduction,
                                                                                     std::forward< Store >( store ),
                                                                                     identity,
                                                                                     launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Graph::IndexType
reduceVerticesIf( const Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  const FetchValue& identity,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return detail::ReductionOperations< typename Graph::ConstViewType >::reduceVerticesIf(
      graph.getConstView(),
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesIf( Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesIf( graph,
                            begin,
                            end,
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            reduction,
                            std::forward< Store >( store ),
                            Reduction::template getIdentity< FetchValue >(),
                            launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesIf( const Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesIf( graph,
                            begin,
                            end,
                            std::forward< Condition >( condition ),
                            std::forward< Fetch >( fetch ),
                            reduction,
                            std::forward< Store >( store ),
                            Reduction::template getIdentity< FetchValue >(),
                            launchConfig );
}

template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVerticesWithArgument( Graph& graph,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               const FetchValue& identity,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceVerticesWithArgument( graph,
                               0,
                               graph.getVertexCount(),
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               identity,
                               launchConfig );
}

template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVerticesWithArgument( const Graph& graph,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               const FetchValue& identity,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceVerticesWithArgument( graph,
                               0,
                               graph.getVertexCount(),
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               identity,
                               launchConfig );
}

template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVerticesWithArgument( Graph& graph,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVerticesWithArgument( graph,
                               0,
                               graph.getVertexCount(),
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               Reduction::template getIdentity< FetchValue >(),
                               launchConfig );
}

template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVerticesWithArgument( const Graph& graph,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               Store&& store,
                               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVerticesWithArgument( graph,
                               0,
                               graph.getVertexCount(),
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               Reduction::template getIdentity< FetchValue >(),
                               launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceVerticesWithArgument( Graph& graph,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            const FetchValue& identity,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::ReductionOperations< typename Graph::ViewType >::reduceVerticesWithArgument( graph_view,
                                                                                        begin,
                                                                                        end,
                                                                                        std::forward< Fetch >( fetch ),
                                                                                        reduction,
                                                                                        std::forward< Store >( store ),
                                                                                        identity,
                                                                                        launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceVerticesWithArgument( const Graph& graph,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            const FetchValue& identity,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Graph::ConstViewType >::reduceVerticesWithArgument( graph.getConstView(),
                                                                                             begin,
                                                                                             end,
                                                                                             std::forward< Fetch >( fetch ),
                                                                                             reduction,
                                                                                             std::forward< Store >( store ),
                                                                                             identity,
                                                                                             launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceVerticesWithArgument( Graph& graph,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVerticesWithArgument( graph,
                               begin,
                               end,
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               Reduction::template getIdentity< FetchValue >(),
                               launchConfig );
}

template< typename Graph, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceVerticesWithArgument( const Graph& graph,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVerticesWithArgument( graph.getConstView(),
                               begin,
                               end,
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               Reduction::template getIdentity< FetchValue >(),
                               launchConfig );
}

template< typename Graph, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceVerticesWithArgument( Graph& graph,
                            const Array& vertexIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            const FetchValue& identity,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   detail::ReductionOperations< typename Graph::ViewType >::reduceVerticesWithArgument( graph_view,
                                                                                        vertexIndexes,
                                                                                        std::forward< Fetch >( fetch ),
                                                                                        reduction,
                                                                                        std::forward< Store >( store ),
                                                                                        identity,
                                                                                        launchConfig );
}

template< typename Graph, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceVerticesWithArgument( const Graph& graph,
                            const Array& vertexIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            const FetchValue& identity,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Graph::ConstViewType >::reduceVerticesWithArgument( graph.getConstView(),
                                                                                             vertexIndexes,
                                                                                             std::forward< Fetch >( fetch ),
                                                                                             reduction,
                                                                                             std::forward< Store >( store ),
                                                                                             identity,
                                                                                             launchConfig );
}

template< typename Graph, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceVerticesWithArgument( Graph& graph,
                            const Array& vertexIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVerticesWithArgument( graph,
                               vertexIndexes,
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               Reduction::template getIdentity< FetchValue >(),
                               launchConfig );
}

template< typename Graph,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T >
void
reduceVerticesWithArgument( const Graph& graph,
                            const Array& vertexIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            Store&& store,
                            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   reduceVerticesWithArgument( graph.getConstView(),
                               vertexIndexes,
                               std::forward< Fetch >( fetch ),
                               reduction,
                               std::forward< Store >( store ),
                               Reduction::template getIdentity< FetchValue >(),
                               launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf( Graph& graph,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 Store&& store,
                                 const FetchValue& identity,
                                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceVerticesWithArgumentIf( graph,
                                        (decltype( graph.getVertexCount() )) 0,
                                        graph.getVertexCount(),
                                        std::forward< Condition >( condition ),
                                        std::forward< Fetch >( fetch ),
                                        reduction,
                                        std::forward< Store >( store ),
                                        identity,
                                        launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf( const Graph& graph,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 Store&& store,
                                 const FetchValue& identity,
                                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceVerticesWithArgumentIf( graph,
                                        (decltype( graph.getVertexCount() )) 0,
                                        graph.getVertexCount(),
                                        std::forward< Condition >( condition ),
                                        std::forward< Fetch >( fetch ),
                                        reduction,
                                        std::forward< Store >( store ),
                                        identity,
                                        launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf( Graph& graph,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 Store&& store,
                                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesWithArgumentIf( graph,
                                        (decltype( graph.getVertexCount() )) 0,
                                        graph.getVertexCount(),
                                        std::forward< Condition >( condition ),
                                        std::forward< Fetch >( fetch ),
                                        reduction,
                                        std::forward< Store >( store ),
                                        Reduction::template getIdentity< FetchValue >(),
                                        launchConfig );
}

template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf( const Graph& graph,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 Store&& store,
                                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesWithArgumentIf( graph,
                                        (decltype( graph.getVertexCount() )) 0,
                                        graph.getVertexCount(),
                                        std::forward< Condition >( condition ),
                                        std::forward< Fetch >( fetch ),
                                        reduction,
                                        std::forward< Store >( store ),
                                        Reduction::template getIdentity< FetchValue >(),
                                        launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Graph::IndexType
reduceVerticesWithArgumentIf( Graph& graph,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              Store&& store,
                              const FetchValue& identity,
                              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto graph_view = graph.getView();
   return detail::ReductionOperations< typename Graph::ViewType >::reduceVerticesWithArgumentIf(
      graph_view,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Graph::IndexType
reduceVerticesWithArgumentIf( const Graph& graph,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              Store&& store,
                              const FetchValue& identity,
                              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return detail::ReductionOperations< typename Graph::ConstViewType >::reduceVerticesWithArgumentIf(
      graph.getConstView(),
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesWithArgumentIf( Graph& graph,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              Store&& store,
                              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesWithArgumentIf( graph,
                                        begin,
                                        end,
                                        std::forward< Condition >( condition ),
                                        std::forward< Fetch >( fetch ),
                                        reduction,
                                        std::forward< Store >( store ),
                                        Reduction::template getIdentity< FetchValue >(),
                                        launchConfig );
}

template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesWithArgumentIf( const Graph& graph,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              Store&& store,
                              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( typename Graph::IndexType(), typename Graph::IndexType(), typename Graph::RealType() ) );
   return reduceVerticesWithArgumentIf( graph.getConstView(),
                                        begin,
                                        end,
                                        std::forward< Condition >( condition ),
                                        std::forward< Fetch >( fetch ),
                                        reduction,
                                        std::forward< Store >( store ),
                                        Reduction::template getIdentity< FetchValue >(),
                                        launchConfig );
}

}  //namespace TNL::Graphs
