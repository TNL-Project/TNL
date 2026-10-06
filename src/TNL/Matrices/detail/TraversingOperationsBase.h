// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>
#include "RowSelection.h"

namespace TNL::Matrices::detail {

/**
 * \brief Base class for matrix traversal operations shared across all specializations.
 *
 * This base class provides implementations that \ref TraversingOperations
 * specializations inherit to avoid code duplication. Currently it provides
 * the conditional `forElementsIf` and `forRowsIf` methods. The unconditional
 * `forElements` and `forRows` methods are implemented by each specialization
 * individually, since they differ per matrix format.
 *
 * The `*If` methods follow a compress + delegate strategy (see \ref buildSelectedRowIndexes
 * and \ref buildSelectedRowIndexesFromArray in RowSelection.h):
 *
 * 1. Materialize the row-condition mask into a vector via \ref TNL::Algorithms::compressFast.
 * 2. Delegate to \ref TraversingOperations<Matrix>::forElements or
 *    \ref TraversingOperations<Matrix>::forRows with the filtered row indexes.
 *
 * Dense and Sparse specializations of \ref TraversingOperations override
 * `forElementsIf` to use optimized conditional GPU kernels from the
 * Segments layer (\ref TNL::Algorithms::Segments::forElementsIf). The other
 * specializations (Tridiagonal, Multidiagonal, Lambda) inherit the default
 * compress+delegate implementation from this base.
 *
 * \tparam Matrix The matrix type (view or owning) the operations act on.
 */
template< typename Matrix >
struct TraversingOperationsBase
{
   using IndexType = typename Matrix::IndexType;
   using DeviceType = typename Matrix::DeviceType;
   using ConstMatrixView = typename Matrix::ConstViewType;

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf(
      Matrix& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forElements(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forElements(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forRowsIf(
      Matrix& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forRows(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   template< typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forRowsIf(
      const ConstMatrixView& matrix,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes =
         buildSelectedRowIndexes< IndexType, DeviceType >( begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forRows(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf(
      Matrix& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forElements(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forElementsIf(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forElements(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forRowsIf(
      Matrix& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forRows(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   template< typename Array, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
   static void
   forRowsIf(
      const ConstMatrixView& matrix,
      const Array& rowIndexes,
      IndexBegin begin,
      IndexEnd end,
      Condition&& condition,
      Function&& function,
      Algorithms::Segments::LaunchConfiguration launchConfig )
   {
      if( end <= begin )
         return;
      auto selectedRowIndexes = buildSelectedRowIndexesFromArray< IndexType, DeviceType >(
         rowIndexes, begin, end, std::forward< Condition >( condition ) );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      TraversingOperations< Matrix >::forRows(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }
};

}  // namespace TNL::Matrices::detail
