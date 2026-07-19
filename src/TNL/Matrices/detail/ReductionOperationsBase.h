// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>
#include "RowSelection.h"

namespace TNL::Matrices::detail {

/**
 * \brief Base class for matrix reduction operations shared across all specializations.
 *
 * This base class provides implementations that \ref ReductionOperations
 * specializations inherit to avoid code duplication. Currently it provides
 * the conditional \c reduceRowsIf and \c reduceRowsWithArgumentIf methods
 * (both range and array overloads). The unconditional \c reduceRows and
 * \c reduceRowsWithArgument methods are implemented by each specialization
 * individually, since they differ per matrix format.
 *
 * The \c *If methods follow a compress + gather + delegate strategy (see
 * \ref buildSelectedRowIndexes and \ref buildSelectedRowIndexesFromArray in RowSelection.h):
 *
 * 1. Materialize the row-condition mask into a vector via \ref TNL::Algorithms::compressFast.
 * 2. For the array overloads, gather the actual row indexes from the user-supplied
 *    \e rowIndexes array using the compressed mask.
 * 3. Delegate to \ref ReductionOperations<Matrix>::reduceRows or
 *    \ref ReductionOperations<Matrix>::reduceRowsWithArgument with the filtered
 *    row indexes.
 *
 * The compress+gather approach is universally applicable, so no specialization
 * needs to override the \c *If methods.
 *
 * \tparam Matrix The matrix type (view or owning) the operations act on.
 */
template< typename Matrix >
struct ReductionOperationsBase
{
   // TODO: `launchConfig` (Algorithms::Segments::LaunchConfiguration) is accepted by the methods
   // below but never forwarded to Algorithms::parallelFor -- the types don't match
   // (Segments::LaunchConfiguration vs. Device::LaunchConfiguration). Fix once it's benchmarked
   // whether this actually matters; likely via launchConfig.getBackendLaunchConfiguration() on
   // GPU devices.

   using IndexType = typename Matrix::IndexType;
   using DeviceType = typename Matrix::DeviceType;
   using ConstMatrixView = typename Matrix::ConstViewType;

   // ===================== reduceRowsIf (range) =====================

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
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
      if( end <= begin )
         return 0;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRows(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRows(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }

   // ===================== reduceRowsIf (array) =====================

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
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
      if( end <= begin )
         return 0;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRows(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsIf(
      const ConstMatrixView& matrix,
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
      if( end <= begin )
         return 0;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRows(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }

   // ===================== reduceRowsWithArgumentIf (range) =====================

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
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
      if( end <= begin )
         return 0;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }

   template<
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Fetch&& fetch,
      Reduction&& reduction,
      Store&& store,
      const FetchValue& identity,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return 0;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }

   // ===================== reduceRowsWithArgumentIf (array) =====================

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
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
      if( end <= begin )
         return 0;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }

   template<
      typename Array,
      typename IndexBegin,
      typename IndexEnd,
      typename Condition,
      typename Fetch,
      typename Reduction,
      typename Store,
      typename FetchValue >
   static IndexType
   reduceRowsWithArgumentIf(
      const ConstMatrixView& matrix,
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
      if( end <= begin )
         return 0;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return 0;
      ReductionOperations< Matrix >::reduceRowsWithArgument(
         matrix,
         selectedRowIndexes,
         std::forward< Fetch >( fetch ),
         std::forward< Reduction >( reduction ),
         std::forward< Store >( store ),
         identity,
         launchConfig );
      return selectedRowIndexes.getSize();
   }
};

}  // namespace TNL::Matrices::detail
