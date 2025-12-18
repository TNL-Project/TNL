// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "detail/TraversingOperations.h"

namespace TNL::Matrices {

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( Matrix& matrix,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forElements(
      matrix_view, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Matrix& matrix,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Matrix::ConstViewType >::forElements(
      matrix.getConstView(), begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Function >
void
forAllElements( Matrix& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forElements(
      matrix_view, (IndexType) 0, matrix.getRows(), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Function >
void
forAllElements( const Matrix& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   detail::TraversingOperations< typename Matrix::ConstViewType >::forElements(
      matrix.getConstView(), (IndexType) 0, matrix.getRows(), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( Matrix& matrix,
             const Array& rowIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forElements(
      matrix_view, rowIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Matrix& matrix,
             const Array& rowIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Matrix::ConstViewType >::forElements(
      matrix.getConstView(), rowIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename Function >
void
forElements( Matrix& matrix,
             const Array& rowIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forElements(
      matrix_view, rowIndexes, (IndexType) 0, rowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename Function >
void
forElements( const Matrix& matrix,
             const Array& rowIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   detail::TraversingOperations< typename Matrix::ConstViewType >::forElements( matrix.getConstView(),
                                                                                rowIndexes,
                                                                                (IndexType) 0,
                                                                                rowIndexes.getSize(),
                                                                                std::forward< Function >( function ),
                                                                                launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( Matrix& matrix,
               IndexBegin begin,
               IndexEnd end,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forElementsIf(
      matrix_view, begin, end, std::forward< Condition >( condition ), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Matrix& matrix,
               IndexBegin begin,
               IndexEnd end,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Matrix::ConstViewType >::forElementsIf( matrix.getConstView(),
                                                                                  begin,
                                                                                  end,
                                                                                  std::forward< Condition >( condition ),
                                                                                  std::forward< Function >( function ),
                                                                                  launchConfig );
}

template< typename Matrix, typename Condition, typename Function >
void
forAllElementsIf( Matrix& matrix,
                  Condition&& condition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forElementsIf( matrix_view,
                                                                             (IndexType) 0,
                                                                             matrix.getRows(),
                                                                             std::forward< Condition >( condition ),
                                                                             std::forward< Function >( function ),
                                                                             launchConfig );
}

template< typename Matrix, typename Condition, typename Function >
void
forAllElementsIf( const Matrix& matrix,
                  Condition&& condition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   detail::TraversingOperations< typename Matrix::ConstViewType >::forElementsIf( matrix.getConstView(),
                                                                                  (IndexType) 0,
                                                                                  matrix.getRows(),
                                                                                  std::forward< Condition >( condition ),
                                                                                  std::forward< Function >( function ),
                                                                                  launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forRows( Matrix& matrix,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forRows(
      matrix_view, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forRows( const Matrix& matrix,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Matrix::ConstViewType >::forRows(
      matrix.getConstView(), begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Function >
void
forAllRows( Matrix& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   auto matrix_view = matrix.getView();
   forRows( matrix_view, (IndexType) 0, matrix.getRows(), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Function >
void
forAllRows( const Matrix& matrix, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   forRows( matrix.getConstView(), (IndexType) 0, matrix.getRows(), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forRows( Matrix& matrix,
         const Array& rowIndexes,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forRows(
      matrix_view, rowIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename IndexBegin, typename IndexEnd, typename Function, typename T >
void
forRows( const Matrix& matrix,
         const Array& rowIndexes,
         IndexBegin begin,
         IndexEnd end,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Matrix::ConstViewType >::forRows(
      matrix.getConstView(), rowIndexes, begin, end, std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename Function, typename T >
void
forRows( Matrix& matrix, const Array& rowIndexes, Function&& function, Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   auto matrix_view = matrix.getView();
   forRows( matrix_view, rowIndexes, (IndexType) 0, rowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename Array, typename Function, typename T >
void
forRows( const Matrix& matrix,
         const Array& rowIndexes,
         Function&& function,
         Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   forRows( matrix, rowIndexes, (IndexType) 0, rowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename RowCondition, typename Function, typename T >
void
forRowsIf( Matrix& matrix,
           IndexBegin begin,
           IndexEnd end,
           RowCondition&& rowCondition,
           Function&& function,
           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::TraversingOperations< typename Matrix::ViewType >::forRowsIf( matrix_view,
                                                                         begin,
                                                                         end,
                                                                         std::forward< RowCondition >( rowCondition ),
                                                                         std::forward< Function >( function ),
                                                                         launchConfig );
}

template< typename Matrix, typename IndexBegin, typename IndexEnd, typename RowCondition, typename Function, typename T >
void
forRowsIf( const Matrix& matrix,
           IndexBegin begin,
           IndexEnd end,
           RowCondition&& rowCondition,
           Function&& function,
           Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::TraversingOperations< typename Matrix::ConstViewType >::forRowsIf( matrix.getConstView(),
                                                                              begin,
                                                                              end,
                                                                              std::forward< RowCondition >( rowCondition ),
                                                                              std::forward< Function >( function ),
                                                                              launchConfig );
}

template< typename Matrix, typename RowCondition, typename Function >
void
forAllRowsIf( Matrix& matrix,
              RowCondition&& rowCondition,
              Function&& function,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   forRowsIf( matrix,
              (typename Matrix::IndexType) 0,
              matrix.getRows(),
              std::forward< RowCondition >( rowCondition ),
              std::forward< Function >( function ),
              launchConfig );
}

template< typename Matrix, typename RowCondition, typename Function >
void
forAllRowsIf( const Matrix& matrix,
              RowCondition&& rowCondition,
              Function&& function,
              Algorithms::Segments::LaunchConfiguration launchConfig )
{
   forRowsIf( matrix,
              (typename Matrix::IndexType) 0,
              matrix.getRows(),
              std::forward< RowCondition >( rowCondition ),
              std::forward< Function >( function ),
              launchConfig );
}

}  //namespace TNL::Matrices
