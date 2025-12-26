// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "reduce.h"
#include "detail/ReductionOperations.h"

namespace TNL::Matrices {
template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRows( Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRows( matrix,
               0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Keep >( keep ),
               identity,
               launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRows( const Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               const FetchValue& identity,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   reduceRows( matrix,
               (IndexType) 0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Keep >( keep ),
               identity,
               launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRows( Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   reduceRows( matrix,
               (IndexType) 0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Keep >( keep ),
               launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRows( const Matrix& matrix,
               Fetch&& fetch,
               Reduction&& reduction,
               Keep&& keep,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   reduceRows( matrix,
               (IndexType) 0,
               matrix.getRows(),
               std::forward< Fetch >( fetch ),
               reduction,
               std::forward< Keep >( keep ),
               launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T >
void
reduceRows( Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows( matrix_view,
                                                                         begin,
                                                                         end,
                                                                         std::forward< Fetch >( fetch ),
                                                                         reduction,
                                                                         std::forward< Keep >( keep ),
                                                                         identity,
                                                                         launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T >
void
reduceRows( const Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows( matrix.getConstView(),
                                                                              begin,
                                                                              end,
                                                                              std::forward< Fetch >( fetch ),
                                                                              reduction,
                                                                              std::forward< Keep >( keep ),
                                                                              identity,
                                                                              launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename T >
void
reduceRows( Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows(
      matrix_view, begin, end, std::forward< Fetch >( fetch ), reduction, std::forward< Keep >( keep ), launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename T >
void
reduceRows( const Matrix& matrix,
            IndexBegin begin,
            IndexEnd end,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows( matrix.getConstView(),
                                                                              begin,
                                                                              end,
                                                                              std::forward< Fetch >( fetch ),
                                                                              reduction,
                                                                              std::forward< Keep >( keep ),
                                                                              launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue, typename T >
void
reduceRows( Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows( matrix_view,
                                                                         rowIndexes,
                                                                         std::forward< Fetch >( fetch ),
                                                                         reduction,
                                                                         std::forward< Keep >( keep ),
                                                                         identity,
                                                                         launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue, typename T >
void
reduceRows( const Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            const FetchValue& identity,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows( matrix.getConstView(),
                                                                              rowIndexes,
                                                                              std::forward< Fetch >( fetch ),
                                                                              reduction,
                                                                              std::forward< Keep >( keep ),
                                                                              identity,
                                                                              launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Keep, typename T >
void
reduceRows( Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows( matrix_view,
                                                                         rowIndexes,
                                                                         0,
                                                                         rowIndexes.getSize(),
                                                                         std::forward< Fetch >( fetch ),
                                                                         reduction,
                                                                         std::forward< Keep >( keep ),
                                                                         launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Keep, typename T >
void
reduceRows( const Matrix& matrix,
            const Array& rowIndexes,
            Fetch&& fetch,
            Reduction&& reduction,
            Keep&& keep,
            Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows( matrix.getConstView(),
                                                                              rowIndexes,
                                                                              0,
                                                                              rowIndexes.getSize(),
                                                                              std::forward< Fetch >( fetch ),
                                                                              reduction,
                                                                              std::forward< Keep >( keep ),
                                                                              launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsIf( Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIf( matrix,
                 (decltype( matrix.getRows() )) 0,
                 matrix.getRows(),
                 std::forward< Condition >( condition ),
                 std::forward< Fetch >( fetch ),
                 reduction,
                 std::forward< Keep >( keep ),
                 identity,
                 launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsIf( const Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 const FetchValue& identity,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIf( matrix,
                 (decltype( matrix.getRows() )) 0,
                 matrix.getRows(),
                 std::forward< Condition >( condition ),
                 std::forward< Fetch >( fetch ),
                 reduction,
                 std::forward< Keep >( keep ),
                 identity,
                 launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIf( Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIf( matrix,
                 (decltype( matrix.getRows() )) 0,
                 matrix.getRows(),
                 std::forward< Condition >( condition ),
                 std::forward< Fetch >( fetch ),
                 reduction,
                 std::forward< Keep >( keep ),
                 launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIf( const Matrix& matrix,
                 Condition&& condition,
                 Fetch&& fetch,
                 Reduction&& reduction,
                 Keep&& keep,
                 Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIf( matrix,
                 (decltype( matrix.getRows() )) 0,
                 matrix.getRows(),
                 std::forward< Condition >( condition ),
                 std::forward< Fetch >( fetch ),
                 reduction,
                 std::forward< Keep >( keep ),
                 launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue >
void
reduceRowsIf( Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              const FetchValue& identity,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsIf( matrix_view,
                                                                           begin,
                                                                           end,
                                                                           std::forward< Condition >( condition ),
                                                                           std::forward< Fetch >( fetch ),
                                                                           reduction,
                                                                           std::forward< Keep >( keep ),
                                                                           identity,
                                                                           launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue >
void
reduceRowsIf( const Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              const FetchValue& identity,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsIf( matrix.getConstView(),
                                                                                begin,
                                                                                end,
                                                                                std::forward< Condition >( condition ),
                                                                                std::forward< Fetch >( fetch ),
                                                                                reduction,
                                                                                std::forward< Keep >( keep ),
                                                                                identity,
                                                                                launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIf( Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsIf( matrix_view,
                                                                           begin,
                                                                           end,
                                                                           std::forward< Condition >( condition ),
                                                                           std::forward< Fetch >( fetch ),
                                                                           reduction,
                                                                           std::forward< Keep >( keep ),
                                                                           launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIf( const Matrix& matrix,
              IndexBegin begin,
              IndexEnd end,
              Condition&& condition,
              Fetch&& fetch,
              Reduction&& reduction,
              Keep&& keep,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsIf( matrix.getConstView(),
                                                                                begin,
                                                                                end,
                                                                                std::forward< Condition >( condition ),
                                                                                std::forward< Fetch >( fetch ),
                                                                                reduction,
                                                                                std::forward< Keep >( keep ),
                                                                                launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsWithArgument( Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument( matrix,
                           0,
                           matrix.getRows(),
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Keep >( keep ),
                           identity,
                           launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsWithArgument( const Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
                           const FetchValue& identity,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument( matrix,
                           0,
                           matrix.getRows(),
                           std::forward< Fetch >( fetch ),
                           reduction,
                           std::forward< Keep >( keep ),
                           identity,
                           launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsWithArgument( Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument(
      matrix, 0, matrix.getRows(), std::forward< Fetch >( fetch ), reduction, std::forward< Keep >( keep ), launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsWithArgument( const Matrix& matrix,
                           Fetch&& fetch,
                           Reduction&& reduction,
                           Keep&& keep,
                           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument(
      matrix, 0, matrix.getRows(), std::forward< Fetch >( fetch ), reduction, std::forward< Keep >( keep ), launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument( matrix_view,
                                                                                     begin,
                                                                                     end,
                                                                                     std::forward< Fetch >( fetch ),
                                                                                     reduction,
                                                                                     std::forward< Keep >( keep ),
                                                                                     identity,
                                                                                     launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue,
          typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument( matrix.getConstView(),
                                                                                          begin,
                                                                                          end,
                                                                                          std::forward< Fetch >( fetch ),
                                                                                          reduction,
                                                                                          std::forward< Keep >( keep ),
                                                                                          identity,
                                                                                          launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument(
      matrix_view, begin, end, std::forward< Fetch >( fetch ), reduction, std::forward< Keep >( keep ), launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename Keep, typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        IndexBegin begin,
                        IndexEnd end,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument( matrix.getConstView(),
                                                                                          begin,
                                                                                          end,
                                                                                          std::forward< Fetch >( fetch ),
                                                                                          reduction,
                                                                                          std::forward< Keep >( keep ),
                                                                                          launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue, typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument( matrix_view,
                                                                                     rowIndexes,
                                                                                     std::forward< Fetch >( fetch ),
                                                                                     reduction,
                                                                                     std::forward< Keep >( keep ),
                                                                                     identity,
                                                                                     launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Keep, typename FetchValue, typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        const FetchValue& identity,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument( matrix.getConstView(),
                                                                                          rowIndexes,
                                                                                          std::forward< Fetch >( fetch ),
                                                                                          reduction,
                                                                                          std::forward< Keep >( keep ),
                                                                                          identity,
                                                                                          launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Keep, typename T >
void
reduceRowsWithArgument( Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( matrix.getRow( 0 ).getRowIndex(), 0, typename Matrix::RealType() ) );
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument(
      matrix_view,
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Keep >( keep ),
      Reduction::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename T >
void
reduceRowsWithArgument( const Matrix& matrix,
                        const Array& rowIndexes,
                        Fetch&& fetch,
                        Reduction&& reduction,
                        Keep&& keep,
                        Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue = decltype( fetch( matrix.getRow( 0 ).getRowIndex(), 0, typename Matrix::RealType() ) );
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument(
      matrix.getConstView(),
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Keep >( keep ),
      Reduction::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsIfWithArgument( Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
                             const FetchValue& identity,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIfWithArgument( matrix,
                             (decltype( matrix.getRows() )) 0,
                             matrix.getRows(),
                             std::forward< Condition >( condition ),
                             std::forward< Fetch >( fetch ),
                             reduction,
                             std::forward< Keep >( keep ),
                             identity,
                             launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep, typename FetchValue >
void
reduceAllRowsIfWithArgument( const Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
                             const FetchValue& identity,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIfWithArgument( matrix,
                             (decltype( matrix.getRows() )) 0,
                             matrix.getRows(),
                             std::forward< Condition >( condition ),
                             std::forward< Fetch >( fetch ),
                             reduction,
                             std::forward< Keep >( keep ),
                             identity,
                             launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIfWithArgument( Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIfWithArgument( matrix,
                             (decltype( matrix.getRows() )) 0,
                             matrix.getRows(),
                             std::forward< Condition >( condition ),
                             std::forward< Fetch >( fetch ),
                             reduction,
                             std::forward< Keep >( keep ),
                             launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Keep >
void
reduceAllRowsIfWithArgument( const Matrix& matrix,
                             Condition&& condition,
                             Fetch&& fetch,
                             Reduction&& reduction,
                             Keep&& keep,
                             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsIfWithArgument( matrix,
                             (decltype( matrix.getRows() )) 0,
                             matrix.getRows(),
                             std::forward< Condition >( condition ),
                             std::forward< Fetch >( fetch ),
                             reduction,
                             std::forward< Keep >( keep ),
                             launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue >
void
reduceRowsIfWithArgument( Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Keep&& keep,
                          const FetchValue& identity,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsIfWithArgument( matrix_view,
                                                                                       begin,
                                                                                       end,
                                                                                       std::forward< Condition >( condition ),
                                                                                       std::forward< Fetch >( fetch ),
                                                                                       reduction,
                                                                                       std::forward< Keep >( keep ),
                                                                                       identity,
                                                                                       launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep,
          typename FetchValue >
void
reduceRowsIfWithArgument( const Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Keep&& keep,
                          const FetchValue& identity,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsIfWithArgument(
      matrix.getConstView(),
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Keep >( keep ),
      identity,
      launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIfWithArgument( Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Keep&& keep,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsIfWithArgument( matrix_view,
                                                                                       begin,
                                                                                       end,
                                                                                       std::forward< Condition >( condition ),
                                                                                       std::forward< Fetch >( fetch ),
                                                                                       reduction,
                                                                                       std::forward< Keep >( keep ),
                                                                                       launchConfig );
}

template< typename Matrix,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Keep >
void
reduceRowsIfWithArgument( const Matrix& matrix,
                          IndexBegin begin,
                          IndexEnd end,
                          Condition&& condition,
                          Fetch&& fetch,
                          Reduction&& reduction,
                          Keep&& keep,
                          Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsIfWithArgument(
      matrix.getConstView(),
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Keep >( keep ),
      launchConfig );
}

}  // namespace TNL::Matrices
