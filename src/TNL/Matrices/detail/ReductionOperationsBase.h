// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>

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
 * The \c *If methods follow a compress + gather + delegate strategy:
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
         return 0;
      selectedRowIndexes += begin;
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
         return 0;
      selectedRowIndexes += begin;
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
         return 0;

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
         return 0;

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
         return 0;
      selectedRowIndexes += begin;
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
         return 0;
      selectedRowIndexes += begin;
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
         return 0;

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
         return 0;

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
