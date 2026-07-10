// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/compress.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Matrices::detail {

/**
 * \brief Default implementation of conditional matrix traversal methods (\c *If variants).
 *
 * This base class provides the shared \c forElementsIf and \c forRowsIf methods
 * for all matrix specializations. The strategy is:
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

   // ===================== forElementsIf (range) =====================
   //
   // Default implementation: materialize the condition mask via compressFast
   // then delegate to forElements with the filtered row indexes.
   // Dense/Sparse specializations override this to use optimized conditional
   // GPU kernels from the Segments layer.

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
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return;
      filteredIndexes += begin;
      TraversingOperations< Matrix >::forElements(
         matrix,
         filteredIndexes,
         (IndexType) 0,
         filteredIndexes.getSize(),
         std::forward< Function >( function ),
         launchConfig );
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
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return;
      filteredIndexes += begin;
      TraversingOperations< Matrix >::forElements(
         matrix,
         filteredIndexes,
         (IndexType) 0,
         filteredIndexes.getSize(),
         std::forward< Function >( function ),
         launchConfig );
   }

   // ===================== forRowsIf (range) =====================
   //
   // Default implementation: materialize the condition mask via compressFast
   // then delegate to forRows with the filtered row indexes.

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
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return;
      filteredIndexes += begin;
      TraversingOperations< Matrix >::forRows(
         matrix,
         filteredIndexes,
         (IndexType) 0,
         filteredIndexes.getSize(),
         std::forward< Function >( function ),
         launchConfig );
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
      VectorType conditions( end - begin );
      conditions.forAllElements(
         [ = ] __cuda_callable__( IndexType i, IndexType & value ) mutable
         {
            value = condition( i + begin ) ? 1 : 0;
         } );
      auto filteredIndexes = Algorithms::compressFast< VectorType >( conditions );
      if( filteredIndexes.getSize() == 0 )
         return;
      filteredIndexes += begin;
      TraversingOperations< Matrix >::forRows(
         matrix,
         filteredIndexes,
         (IndexType) 0,
         filteredIndexes.getSize(),
         std::forward< Function >( function ),
         launchConfig );
   }
};

}  // namespace TNL::Matrices::detail
