// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Matrices::detail {

/**
 * \brief Builds a dense array of row indexes in `[begin, end)` for which `condition(rowIdx)` holds.
 *
 * Shared by the `*If` methods of \ref TraversingOperationsBase and \ref ReductionOperationsBase
 * (range overloads): materializes the condition as a 0/1 mask and compresses it via
 * \ref TNL::Algorithms::compressFast.
 */
template< typename IndexType, typename DeviceType, typename IndexBegin, typename IndexEnd, typename Condition >
Containers::Vector< IndexType, DeviceType, IndexType >
buildSelectedRowIndexes( IndexBegin begin, IndexEnd end, Condition&& condition )
{
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
   if( selectedRowIndexes.getSize() > 0 )
      selectedRowIndexes += begin;
   return selectedRowIndexes;
}

/**
 * \brief Builds a dense array of the row indexes `rowIndexes[pos]` for `pos` in `[begin, end)` for
 * which `condition(rowIndexes[pos])` holds.
 *
 * Shared by the `*If` methods of \ref TraversingOperationsBase and \ref ReductionOperationsBase
 * (array overloads): materializes the condition as a 0/1 mask over positions, compresses it into
 * matching positions via \ref TNL::Algorithms::compressFast, then gathers the actual row indexes.
 */
template< typename IndexType, typename DeviceType, typename Array, typename IndexBegin, typename IndexEnd, typename Condition >
Containers::Vector< IndexType, DeviceType, IndexType >
buildSelectedRowIndexesFromArray( const Array& rowIndexes, IndexBegin begin, IndexEnd end, Condition&& condition )
{
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
      return matchingPositions;

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
   return selectedRowIndexes;
}

}  // namespace TNL::Matrices::detail
