// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "reduce.h"
#include "detail/ReductionOperations.h"

namespace TNL::Matrices {
template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRows( Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRows( matrix,
               0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               identity,
               launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRows( const Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   reduceRows( matrix,
               (IndexType) 0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               identity,
               launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRows( Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows( matrix,
               (IndexType) 0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               Reduction::template getIdentity< FetchValue >(),
               launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRows( const Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Store&& store,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows( matrix,
               (IndexType) 0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               Reduction::template getIdentity< FetchValue >(),
               launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceRows( Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows( matrix_view,
                                                                         begin,
                                                                         end,
                                                                         std::forward< Fetch >( fetch ),
                                                                         reduction,
                                                                         std::forward< Store >( store ),
                                                                         identity,
                                                                         launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceRows( const Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows( matrix.getConstView(),
                                                                              begin,
                                                                              end,
                                                                              std::forward< Fetch >( fetch ),
                                                                              reduction,
                                                                              std::forward< Store >( store ),
                                                                              identity,
                                                                              launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T >
void
reduceRows( Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows( matrix,
               begin,
               end,
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               Reduction::template getIdentity< FetchValue >(),
               launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T >
void
reduceRows( const Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows( matrix,
               begin,
               end,
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               Reduction::template getIdentity< FetchValue >(),
               launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRows( Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows( matrix_view,
                                                                         rowIndexes,
                                                                         std::forward< Fetch >( fetch ),
                                                                         reduction,
                                                                         std::forward< Store >( store ),
                                                                         identity,
                                                                         launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRows( const Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows( matrix.getConstView(),
                                                                              rowIndexes,
                                                                              std::forward< Fetch >( fetch ),
                                                                              reduction,
                                                                              std::forward< Store >( store ),
                                                                              identity,
                                                                              launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceRows( Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows( matrix,
               rowIndexes,
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               Reduction::template getIdentity< FetchValue >(),
               launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceRows( const Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Store&& store,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows( matrix.getConstView(),
               rowIndexes,
               0,
               rowIndexes.getSize(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Store >( store ),
               Reduction::template getIdentity< FetchValue >(),
               launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Matrix::IndexType
reduceAllRowsIf( Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Store&& store,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsIf( matrix,
                        (decltype( matrix.getRows() )) 0,
                        matrix.getRows(),
                        std::forward< Condition >( condition ),
                        std::forward< Fetch >( fetch ),
                        reduction,
                        std::forward< Store >( store ),
                        identity,
                        launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Matrix::IndexType
reduceAllRowsIf( const Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Store&& store,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsIf( matrix,
                        (decltype( matrix.getRows() )) 0,
                        matrix.getRows(),
                        std::forward< Condition >( condition ),
                        std::forward< Fetch >( fetch ),
                        reduction,
                        std::forward< Store >( store ),
                        identity,
                        launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Matrix::IndexType
reduceAllRowsIf( Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Store&& store,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf( matrix,
                        (decltype( matrix.getRows() )) 0,
                        matrix.getRows(),
                        std::forward< Condition >( condition ),
                        std::forward< Fetch >( fetch ),
                        reduction,
                        std::forward< Store >( store ),
                        Reduction::template getIdentity< FetchValue >(),
                        launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Matrix::IndexType
reduceAllRowsIf( const Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Store&& store,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf( matrix,
                        (decltype( matrix.getRows() )) 0,
                        matrix.getRows(),
                        std::forward< Condition >( condition ),
                        std::forward< Fetch >( fetch ),
                        reduction,
                        std::forward< Store >( store ),
                        Reduction::template getIdentity< FetchValue >(),
                        launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Matrix::IndexType
reduceRowsIf( Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Store&& store,
              const FetchValue& identity,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   return detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsIf( matrix_view,
                                                                                  begin,
                                                                                  end,
                                                                                  std::forward< Condition >( condition ),
                                                                                  std::forward< Fetch >( fetch ),
                                                                                  reduction,
                                                                                  std::forward< Store >( store ),
                                                                                  identity,
                                                                                  launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Matrix::IndexType
reduceRowsIf( const Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Store&& store,
              const FetchValue& identity,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsIf( matrix.getConstView(),
                                                                                       begin,
                                                                                       end,
                                                                                       std::forward< Condition >( condition ),
                                                                                       std::forward< Fetch >( fetch ),
                                                                                       reduction,
                                                                                       std::forward< Store >( store ),
                                                                                       identity,
                                                                                       launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Matrix::IndexType
reduceRowsIf( Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Store&& store,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf( matrix,
                        begin,
                        end,
                        std::forward< Condition >( condition ),
                        std::forward< Fetch >( fetch ),
                        reduction,
                        std::forward< Store >( store ),
                        Reduction::template getIdentity< FetchValue >(),
                        launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Matrix::IndexType
reduceRowsIf( const Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Store&& store,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf( matrix,
                        begin,
                        end,
                        std::forward< Condition >( condition ),
                        std::forward< Fetch >( fetch ),
                        reduction,
                        std::forward< Store >( store ),
                        Reduction::template getIdentity< FetchValue >(),
                        launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRowsWithArgument( Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument( matrix,
                           0,
                           matrix.getRows(),
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           identity,
                           launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRowsWithArgument( const Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument( matrix,
                           0,
                           matrix.getRows(),
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           identity,
                           launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRowsWithArgument( Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument( matrix,
                           0,
                           matrix.getRows(),
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           Reduction::template getIdentity< FetchValue >(),
                           launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRowsWithArgument( const Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Store&& store,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument( matrix,
                           0,
                           matrix.getRows(),
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           Reduction::template getIdentity< FetchValue >(),
                           launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument( matrix_view,
                                                                                     begin,
                                                                                     end,
                                                                                     std::forward< Fetch >( fetch ),
                                                                                     reduction,
                                                                                     std::forward< Store >( store ),
                                                                                     identity,
                                                                                     launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument( matrix.getConstView(),
                                                                                          begin,
                                                                                          end,
                                                                                          std::forward< Fetch >( fetch ),
                                                                                          reduction,
                                                                                          std::forward< Store >( store ),
                                                                                          identity,
                                                                                          launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument( matrix,
                           begin,
                           end,
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           Reduction::template getIdentity< FetchValue >(),
                           launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument( matrix.getConstView(),
                           begin,
                           end,
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           Reduction::template getIdentity< FetchValue >(),
                           launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument( matrix_view,
                                                                                     rowIndexes,
                                                                                     std::forward< Fetch >( fetch ),
                                                                                     reduction,
                                                                                     std::forward< Store >( store ),
                                                                                     identity,
                                                                                     launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument( matrix.getConstView(),
                                                                                          rowIndexes,
                                                                                          std::forward< Fetch >( fetch ),
                                                                                          reduction,
                                                                                          std::forward< Store >( store ),
                                                                                          identity,
                                                                                          launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument( matrix,
                           rowIndexes,
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           Reduction::template getIdentity< FetchValue >(),
                           launchConfig );
}

template< typename Matrix,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Store&& store,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument( matrix.getConstView(),
                           rowIndexes,
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Store >( store ),
                           Reduction::template getIdentity< FetchValue >(),
                           launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Matrix::IndexType
reduceAllRowsWithArgumentIf( Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Store&& store,
                             const FetchValue& identity,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsWithArgumentIf( matrix,
                                    (decltype( matrix.getRows() )) 0,
                                    matrix.getRows(),
                                    std::forward< Condition >( condition ),
                                    std::forward< Fetch >( fetch ),
                                    reduction,
                                    std::forward< Store >( store ),
                                    identity,
                                    launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Matrix::IndexType
reduceAllRowsWithArgumentIf( const Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Store&& store,
                             const FetchValue& identity,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsWithArgumentIf( matrix,
                                    (decltype( matrix.getRows() )) 0,
                                    matrix.getRows(),
                                    std::forward< Condition >( condition ),
                                    std::forward< Fetch >( fetch ),
                                    reduction,
                                    std::forward< Store >( store ),
                                    identity,
                                    launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Matrix::IndexType
reduceAllRowsWithArgumentIf( Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Store&& store,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf( matrix,
                                    (decltype( matrix.getRows() )) 0,
                                    matrix.getRows(),
                                    std::forward< Condition >( condition ),
                                    std::forward< Fetch >( fetch ),
                                    reduction,
                                    std::forward< Store >( store ),
                                    Reduction::template getIdentity< FetchValue >(),
                                    launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Matrix::IndexType
reduceAllRowsWithArgumentIf( const Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Store&& store,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf( matrix,
                                    (decltype( matrix.getRows() )) 0,
                                    matrix.getRows(),
                                    std::forward< Condition >( condition ),
                                    std::forward< Fetch >( fetch ),
                                    reduction,
                                    std::forward< Store >( store ),
                                    Reduction::template getIdentity< FetchValue >(),
                                    launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Matrix::IndexType
reduceRowsWithArgumentIf( Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Store&& store,
                          const FetchValue& identity,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   return detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgumentIf(
      matrix_view,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Matrix::IndexType
reduceRowsWithArgumentIf( const Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Store&& store,
                          const FetchValue& identity,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgumentIf(
      matrix.getConstView(),
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Matrix::IndexType
reduceRowsWithArgumentIf( Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Store&& store,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf( matrix,
                                    begin,
                                    end,
                                    std::forward< Condition >( condition ),
                                    std::forward< Fetch >( fetch ),
                                    reduction,
                                    std::forward< Store >( store ),
                                    Reduction::template getIdentity< FetchValue >(),
                                    launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Matrix::IndexType
reduceRowsWithArgumentIf( const Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Store&& store,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf( matrix.getConstView(),
                                    begin,
                                    end,
                                    std::forward< Condition >( condition ),
                                    std::forward< Fetch >( fetch ),
                                    reduction,
                                    std::forward< Store >( store ),
                                    Reduction::template getIdentity< FetchValue >(),
                                    launchConfig );
}

}  // namespace TNL::Matrices
