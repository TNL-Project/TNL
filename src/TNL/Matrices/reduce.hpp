// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include "reduce.h"
#include "detail/ReductionOperations.h"
#include "detail/RowSelection.h"

namespace TNL::Matrices {
template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRows(
   Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   reduceRows(
      matrix,
      static_cast< IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRows(
   const Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   reduceRows(
      matrix,
      static_cast< IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRows(
   Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      static_cast< IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRows(
   const Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using IndexType = typename Matrix::IndexType;
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      static_cast< IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRows(
   Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows(
      matrix_view,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRows(
   const Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows(
      matrix.getConstView(),
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRows(
   Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRows(
   const Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRows(
   Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRows(
      matrix_view,
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRows(
   const Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRows(
      matrix.getConstView(),
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceRows(
   Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceRows(
   const Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRows(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using IndexType = typename Matrix::IndexType;
   // The index passed to the store function is the position within the whole array of row indexes.
   const IndexType offset = begin;
   auto storeWithOffset = [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const FetchValue& value ) mutable
   {
      store( indexOfRowIdx + offset, rowIdx, value );
   };
   reduceRows(
      matrix,
      rowIndexes.getConstView( begin, end ),
      std::forward< Fetch >( fetch ),
      reduction,
      storeWithOffset,
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRows(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using IndexType = typename Matrix::IndexType;
   // The index passed to the store function is the position within the whole array of row indexes.
   const IndexType offset = begin;
   auto storeWithOffset = [ = ] __cuda_callable__( IndexType indexOfRowIdx, IndexType rowIdx, const FetchValue& value ) mutable
   {
      store( indexOfRowIdx + offset, rowIdx, value );
   };
   reduceRows(
      matrix,
      rowIndexes.getConstView( begin, end ),
      std::forward< Fetch >( fetch ),
      reduction,
      storeWithOffset,
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRows(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRows(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRows(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Matrix::IndexType
reduceAllRowsIf(
   Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
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
reduceAllRowsIf(
   const Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
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
reduceAllRowsIf(
   Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
      matrix.getRows(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Matrix::IndexType
reduceAllRowsIf(
   const Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
      matrix.getRows(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   Matrix& matrix,
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
   return detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsIf(
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

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   const Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsIf(
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

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   Matrix& matrix,
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
   return reduceRowsIf(
      matrix,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   const Matrix& matrix,
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
   return reduceRowsIf(
      matrix,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   auto matrix_view = matrix.getView();
   return detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsIf(
      matrix_view,
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   return detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsIf(
      matrix.getConstView(),
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   return reduceRowsIf(
      matrix,
      rowIndexes,
      static_cast< typename Matrix::IndexType >( 0 ),
      rowIndexes.getSize(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   return reduceRowsIf(
      matrix,
      rowIndexes,
      static_cast< typename Matrix::IndexType >( 0 ),
      rowIndexes.getSize(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Array, typename Condition, typename Fetch, typename Reduction, typename Store, typename T >
typename Matrix::IndexType
reduceRowsIf(
   Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf(
      matrix,
      rowIndexes,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Array, typename Condition, typename Fetch, typename Reduction, typename Store, typename T >
typename Matrix::IndexType
reduceRowsIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsIf(
      matrix,
      rowIndexes,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRowsWithArgument(
   Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument(
      matrix,
      static_cast< typename Matrix::IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllRowsWithArgument(
   const Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   reduceRowsWithArgument(
      matrix,
      static_cast< typename Matrix::IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRowsWithArgument(
   Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix,
      static_cast< typename Matrix::IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Fetch, typename Reduction, typename Store >
void
reduceAllRowsWithArgument(
   const Matrix& matrix,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix,
      static_cast< typename Matrix::IndexType >( 0 ),
      matrix.getRows(),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRowsWithArgument(
   Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument(
      matrix_view,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRowsWithArgument(
   const Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument(
      matrix.getConstView(),
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRowsWithArgument(
   Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRowsWithArgument(
   const Matrix& matrix,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix.getConstView(),
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRowsWithArgument(
   Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   auto matrix_view = matrix.getView();
   detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgument(
      matrix_view,
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename FetchValue, typename T >
void
reduceRowsWithArgument(
   const Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgument(
      matrix.getConstView(),
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceRowsWithArgument(
   Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix,
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Array, typename Fetch, typename Reduction, typename Store, typename T >
void
reduceRowsWithArgument(
   const Matrix& matrix,
   const Array& rowIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix,
      rowIndexes,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRowsWithArgument(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using IndexType = typename Matrix::IndexType;
   // The index passed to the store function is the position within the whole array of row indexes.
   const IndexType offset = begin;
   auto storeWithOffset = [ = ] __cuda_callable__(
                             IndexType indexOfRowIdx,
                             IndexType rowIdx,
                             IndexType localIdx,
                             IndexType columnIdx,
                             const FetchValue& value,
                             bool emptyRow ) mutable
   {
      store( indexOfRowIdx + offset, rowIdx, localIdx, columnIdx, value, emptyRow );
   };
   reduceRowsWithArgument(
      matrix,
      rowIndexes.getConstView( begin, end ),
      std::forward< Fetch >( fetch ),
      reduction,
      storeWithOffset,
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
void
reduceRowsWithArgument(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using IndexType = typename Matrix::IndexType;
   // The index passed to the store function is the position within the whole array of row indexes.
   const IndexType offset = begin;
   auto storeWithOffset = [ = ] __cuda_callable__(
                             IndexType indexOfRowIdx,
                             IndexType rowIdx,
                             IndexType localIdx,
                             IndexType columnIdx,
                             const FetchValue& value,
                             bool emptyRow ) mutable
   {
      store( indexOfRowIdx + offset, rowIdx, localIdx, columnIdx, value, emptyRow );
   };
   reduceRowsWithArgument(
      matrix,
      rowIndexes.getConstView( begin, end ),
      std::forward< Fetch >( fetch ),
      reduction,
      storeWithOffset,
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRowsWithArgument(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
void
reduceRowsWithArgument(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   reduceRowsWithArgument(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Matrix::IndexType
reduceAllRowsWithArgumentIf(
   Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsWithArgumentIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
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
reduceAllRowsWithArgumentIf(
   const Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   return reduceRowsWithArgumentIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
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
reduceAllRowsWithArgumentIf(
   Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
      matrix.getRows(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Matrix::IndexType
reduceAllRowsWithArgumentIf(
   const Matrix& matrix,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf(
      matrix,
      static_cast< decltype( matrix.getRows() ) >( 0 ),
      matrix.getRows(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   Matrix& matrix,
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

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   const Matrix& matrix,
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

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   Matrix& matrix,
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
   return reduceRowsWithArgumentIf(
      matrix,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   const Matrix& matrix,
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
   return reduceRowsWithArgumentIf(
      matrix,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   auto matrix_view = matrix.getView();
   return detail::ReductionOperations< typename Matrix::ViewType >::reduceRowsWithArgumentIf(
      matrix_view,
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   return detail::ReductionOperations< typename Matrix::ConstViewType >::reduceRowsWithArgumentIf(
      matrix.getConstView(),
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename IndexBegin,
   typename IndexEnd,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf(
      matrix,
      rowIndexes,
      begin,
      end,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   return reduceRowsWithArgumentIf(
      matrix,
      rowIndexes,
      static_cast< typename Matrix::IndexType >( 0 ),
      rowIndexes.getSize(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template<
   typename Matrix,
   typename Array,
   typename Condition,
   typename Fetch,
   typename Reduction,
   typename Store,
   typename FetchValue,
   typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   return reduceRowsWithArgumentIf(
      matrix,
      rowIndexes,
      static_cast< typename Matrix::IndexType >( 0 ),
      rowIndexes.getSize(),
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      identity,
      launchConfig );
}

template< typename Matrix, typename Array, typename Condition, typename Fetch, typename Reduction, typename Store, typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf(
      matrix,
      rowIndexes,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

template< typename Matrix, typename Array, typename Condition, typename Fetch, typename Reduction, typename Store, typename T >
typename Matrix::IndexType
reduceRowsWithArgumentIf(
   const Matrix& matrix,
   const Array& rowIndexes,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig )
{
   detail::checkRowIndexesDevice< typename Matrix::DeviceType, Array >();
   using FetchValue =
      decltype( fetch( typename Matrix::IndexType(), typename Matrix::IndexType(), typename Matrix::RealType() ) );
   return reduceRowsWithArgumentIf(
      matrix,
      rowIndexes,
      std::forward< Condition >( condition ),
      std::forward< Fetch >( fetch ),
      reduction,
      std::forward< Store >( store ),
      std::decay_t< Reduction >::template getIdentity< FetchValue >(),
      launchConfig );
}

}  // namespace TNL::Matrices
