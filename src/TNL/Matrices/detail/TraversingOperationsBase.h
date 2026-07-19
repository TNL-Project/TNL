// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Matrices::detail {

/**
 * \brief Base class for matrix traversal operations shared across all specializations.
 *
 * This base class provides implementations that \ref TraversingOperations
 * specializations inherit to avoid code duplication. Currently it provides
 * the conditional \c forElementsIf and \c forRowsIf methods. The unconditional
 * \c forElements and \c forRows methods are implemented by each specialization
 * individually, since they differ per matrix format.
 *
 * The \c *If methods follow a compress + delegate strategy:
 *
 * 1. Materialize the row-condition mask into a vector via \ref TNL::Algorithms::compressFast.
 * 2. Delegate to \ref TraversingOperations<Matrix>::forElements or
 *    \ref TraversingOperations<Matrix>::forRows with the filtered row indexes.
 *
 * Dense and Sparse specializations of \ref TraversingOperations override
 * \c forElementsIf to use optimized conditional GPU kernels from the
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

   // TODO: `launchConfig` (Algorithms::Segments::LaunchConfiguration) is accepted by the methods
   // below but never forwarded to Algorithms::parallelFor -- the types don't match
   // (Segments::LaunchConfiguration vs. Device::LaunchConfiguration). Fix once it's benchmarked
   // whether this actually matters; likely via launchConfig.getBackendLaunchConfiguration() on
   // GPU devices.

   // ===================== forElementsIf (range) =====================

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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      // Build a 0/1 mask: 1 where condition(rowIdx) holds, 0 otherwise
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType & value ) mutable
         {
            value = condition( rowIdx + begin ) ? 1 : 0;
         } );
      // Compress the mask into a dense array of matching row indexes
      auto selectedRowIndexes = Algorithms::compressFast< VectorType >( conditionMask );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      selectedRowIndexes += begin;
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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      // Build a 0/1 mask: 1 where condition(rowIdx) holds, 0 otherwise
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType & value ) mutable
         {
            value = condition( rowIdx + begin ) ? 1 : 0;
         } );
      // Compress the mask into a dense array of matching row indexes
      auto selectedRowIndexes = Algorithms::compressFast< VectorType >( conditionMask );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      selectedRowIndexes += begin;
      TraversingOperations< Matrix >::forElements(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   // ===================== forRowsIf (range) =====================

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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      // Build a 0/1 mask: 1 where condition(rowIdx) holds, 0 otherwise
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType & value ) mutable
         {
            value = condition( rowIdx + begin ) ? 1 : 0;
         } );
      // Compress the mask into a dense array of matching row indexes
      auto selectedRowIndexes = Algorithms::compressFast< VectorType >( conditionMask );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      selectedRowIndexes += begin;
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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      // Build a 0/1 mask: 1 where condition(rowIdx) holds, 0 otherwise
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType rowIdx, IndexType & value ) mutable
         {
            value = condition( rowIdx + begin ) ? 1 : 0;
         } );
      // Compress the mask into a dense array of matching row indexes
      auto selectedRowIndexes = Algorithms::compressFast< VectorType >( conditionMask );
      if( selectedRowIndexes.getSize() == 0 )
         return;
      selectedRowIndexes += begin;
      TraversingOperations< Matrix >::forRows(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   // ===================== forElementsIf (array) =====================

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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      // Build a 0/1 mask over positions [begin, end): condition receives the row index, not the position
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType positionIdx, IndexType & value ) mutable
         {
            value = condition( rowIndexes_view[ positionIdx + begin ] ) ? 1 : 0;
         } );
      // Compress the mask into matching positions within [0, end - begin)
      auto matchingPositions = Algorithms::compressFast< VectorType >( conditionMask );
      if( matchingPositions.getSize() == 0 )
         return;

      // Gather: map each matching position to the actual row index via rowIndexes
      VectorType selectedRowIndexes( matchingPositions.getSize() );
      auto selectedRowIndexes_view = selectedRowIndexes.getView();
      auto matchingPositions_view = matchingPositions.getConstView();
      Algorithms::parallelFor< DeviceType >(
         0,
         matchingPositions.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            selectedRowIndexes_view[ i ] = rowIndexes_view[ matchingPositions_view[ i ] + begin ];
         } );

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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      // Build a 0/1 mask over positions [begin, end): condition receives the row index, not the position
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType positionIdx, IndexType & value ) mutable
         {
            value = condition( rowIndexes_view[ positionIdx + begin ] ) ? 1 : 0;
         } );
      // Compress the mask into matching positions within [0, end - begin)
      auto matchingPositions = Algorithms::compressFast< VectorType >( conditionMask );
      if( matchingPositions.getSize() == 0 )
         return;

      // Gather: map each matching position to the actual row index via rowIndexes
      VectorType selectedRowIndexes( matchingPositions.getSize() );
      auto selectedRowIndexes_view = selectedRowIndexes.getView();
      auto matchingPositions_view = matchingPositions.getConstView();
      Algorithms::parallelFor< DeviceType >(
         0,
         matchingPositions.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            selectedRowIndexes_view[ i ] = rowIndexes_view[ matchingPositions_view[ i ] + begin ];
         } );

      TraversingOperations< Matrix >::forElements(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }

   // ===================== forRowsIf (array) =====================

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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      // Build a 0/1 mask over positions [begin, end): condition receives the row index, not the position
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType positionIdx, IndexType & value ) mutable
         {
            value = condition( rowIndexes_view[ positionIdx + begin ] ) ? 1 : 0;
         } );
      // Compress the mask into matching positions within [0, end - begin)
      auto matchingPositions = Algorithms::compressFast< VectorType >( conditionMask );
      if( matchingPositions.getSize() == 0 )
         return;

      // Gather: map each matching position to the actual row index via rowIndexes
      VectorType selectedRowIndexes( matchingPositions.getSize() );
      auto selectedRowIndexes_view = selectedRowIndexes.getView();
      auto matchingPositions_view = matchingPositions.getConstView();
      Algorithms::parallelFor< DeviceType >(
         0,
         matchingPositions.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            selectedRowIndexes_view[ i ] = rowIndexes_view[ matchingPositions_view[ i ] + begin ];
         } );

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
      using VectorType = Containers::Vector< IndexType, DeviceType, IndexType >;
      auto rowIndexes_view = rowIndexes.getConstView();

      // Build a 0/1 mask over positions [begin, end): condition receives the row index, not the position
      VectorType conditionMask( end - begin );
      conditionMask.forAllElements(
         [ = ] __cuda_callable__( IndexType positionIdx, IndexType & value ) mutable
         {
            value = condition( rowIndexes_view[ positionIdx + begin ] ) ? 1 : 0;
         } );
      // Compress the mask into matching positions within [0, end - begin)
      auto matchingPositions = Algorithms::compressFast< VectorType >( conditionMask );
      if( matchingPositions.getSize() == 0 )
         return;

      // Gather: map each matching position to the actual row index via rowIndexes
      VectorType selectedRowIndexes( matchingPositions.getSize() );
      auto selectedRowIndexes_view = selectedRowIndexes.getView();
      auto matchingPositions_view = matchingPositions.getConstView();
      Algorithms::parallelFor< DeviceType >(
         0,
         matchingPositions.getSize(),
         [ = ] __cuda_callable__( IndexType i ) mutable
         {
            selectedRowIndexes_view[ i ] = rowIndexes_view[ matchingPositions_view[ i ] + begin ];
         } );

      TraversingOperations< Matrix >::forRows(
         matrix, selectedRowIndexes, 0, selectedRowIndexes.getSize(), std::forward< Function >( function ), launchConfig );
   }
};

}  // namespace TNL::Matrices::detail
