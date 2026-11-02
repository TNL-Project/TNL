// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <type_traits>

#include <TNL/Algorithms/Segments/multiReduce.h>
#include <TNL/Functional.h>
#include <TNL/Math.h>
#include <TNL/Matrices/MatrixBase.h>
#include <TNL/TypeTraits.h>
#include "SparseDenseOperations.h"

namespace TNL::Matrices {

namespace detail {

// Computes the columns [columnOffset, columnOffset + TileSize) of C = alpha * A * B + beta * C.
template< int TileSize, typename SparseMatrixView, typename DenseMatrixBView, typename DenseMatrixCView, typename Real >
void
sparseDenseMatrixProductTile(
   const SparseMatrixView& A,
   const DenseMatrixBView& B,
   DenseMatrixCView& C,
   typename SparseMatrixView::IndexType columnOffset,
   Real alpha,
   Real beta,
   const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using IndexType = typename SparseMatrixView::IndexType;
   using ComputeRealType = typename SparseMatrixView::ComputeRealType;

   const auto values = A.getValues().getConstView();
   const auto columnIndexes = A.getColumnIndexes().getConstView();
   // the captures are explicit, nvcc does not allow the first capture inside the constexpr if
   auto fetch = [ values, columnIndexes, B, columnOffset ] __cuda_callable__( IndexType globalIdx, int j ) -> ComputeRealType
   {
      const IndexType column = columnIndexes[ globalIdx ];
      TNL_ASSERT_TRUE( column >= 0 || column == paddingIndex< IndexType >, "Wrong column index." );
      // The check of the padding elements is a branch, which separates the multiplication from the
      // addition in the reduction, so the compiler cannot fuse them into an FMA instruction. This matters
      // when the product is compute-bound (e.g. in double precision), hence the check is done only for
      // segments with padding elements.
      if constexpr( SparseMatrixView::SegmentsViewType::havePadding() ) {
         if( column == paddingIndex< IndexType > )
            return 0;
      }
      if constexpr( SparseMatrixView::isBinary() )
         return B( column, columnOffset + j );
      else
         return values[ globalIdx ] * B( column, columnOffset + j );
   };
   auto storer = [ = ] __cuda_callable__( IndexType row, int j, const ComputeRealType& value ) mutable
   {
      auto& element = C( row, columnOffset + j );
      if( beta == Real{ 0 } )
         element = alpha * value;
      else
         element = alpha * value + beta * element;
   };
   Algorithms::Segments::multiReduceAllSegments< TileSize >(
      A.getSegments(), fetch, TNL::Plus{}, storer, ComputeRealType{ 0 }, launchConfig );
}

// Calls f( std::integral_constant< int, size >{} ) for size in [1, MaxSize].
template< int MaxSize, typename Function >
void
dispatchTileSize( int size, Function&& f )
{
   if constexpr( MaxSize > 0 ) {
      if( size == MaxSize )
         f( std::integral_constant< int, MaxSize >{} );
      else
         dispatchTileSize< MaxSize - 1 >( size, std::forward< Function >( f ) );
   }
}

}  // namespace detail

template< int MaxTileSize, typename SparseMatrix, typename DenseMatrixB, typename DenseMatrixC, typename Real >
void
sparseDenseMatrixProduct(
   const SparseMatrix& A,
   const DenseMatrixB& B,
   DenseMatrixC& C,
   Real alpha,
   Real beta,
   const Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   static_assert( MaxTileSize > 0, "The maximal tile size must be positive." );
   static_assert( ! SparseMatrix::isSymmetric(), "sparseDenseMatrixProduct does not support symmetric matrices yet." );
   using IndexType = typename SparseMatrix::IndexType;

   if( A.getColumns() != B.getRows() )
      throw std::invalid_argument(
         "sparseDenseMatrixProduct: the number of columns of A does not match the number of rows of B" );
   if( A.getRows() != C.getRows() )
      throw std::invalid_argument( "sparseDenseMatrixProduct: the number of rows of A does not match the number of rows of C" );
   if( B.getColumns() != C.getColumns() )
      throw std::invalid_argument(
         "sparseDenseMatrixProduct: the number of columns of B does not match the number of columns of C" );

   const auto AView = A.getConstView();
   const auto BView = B.getConstView();
   auto CView = C.getView();
   const IndexType k = B.getColumns();
   for( IndexType columnOffset = 0; columnOffset < k; columnOffset += MaxTileSize ) {
      const int tileSize = TNL::min( k - columnOffset, static_cast< IndexType >( MaxTileSize ) );
      detail::dispatchTileSize< MaxTileSize >(
         tileSize,
         [ & ]( auto tileSize_ )
         {
            detail::sparseDenseMatrixProductTile< decltype( tileSize_ )::value >(
               AView, BView, CView, columnOffset, alpha, beta, launchConfig );
         } );
   }
}

}  // namespace TNL::Matrices
