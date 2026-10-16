// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Benchmarks::MatrixAssembly {

/**
 * \brief Finite difference discretization of the Laplace operator on a structured grid.
 *
 * The grid has \e xSize, \e ySize and \e zSize nodes along the axes (unused axes have size 1).
 * Nodes are numbered lexicographically, i.e. `row = i + xSize * ( j + ySize * k )`.
 * Boundary nodes produce Dirichlet rows with a single unit diagonal element, interior
 * nodes produce the standard 3-, 5- or 7-point stencil with the space steps
 * `h = 1 / ( size - 1 )`. Elements of each row are generated in ascending order of
 * column indexes, so the generated data is sorted in the row-major order.
 *
 * All methods marked with \ref __cuda_callable__ can be used in GPU kernels and the
 * object can be captured by value in lambda functions.
 */
template< typename Real, typename Index >
struct LaplaceProblem
{
   using RealType = Real;
   using IndexType = Index;

   LaplaceProblem() = default;

   LaplaceProblem( int dimension, Index xSize, Index ySize, Index zSize )
   : dimension( dimension ),
     xSize( xSize ),
     ySize( dimension > 1 ? ySize : 1 ),
     zSize( dimension > 2 ? zSize : 1 )
   {
      hxSquareInverse = xSize > 1 ? ( xSize - 1 ) * ( xSize - 1 ) : 1;
      hySquareInverse = this->ySize > 1 ? ( this->ySize - 1 ) * ( this->ySize - 1 ) : 1;
      hzSquareInverse = this->zSize > 1 ? ( this->zSize - 1 ) * ( this->zSize - 1 ) : 1;
   }

   [[nodiscard]] __cuda_callable__
   int
   getDimension() const
   {
      return dimension;
   }

   [[nodiscard]] __cuda_callable__
   Index
   getRows() const
   {
      return xSize * ySize * zSize;
   }

   [[nodiscard]] std::string
   getGridSizeString() const
   {
      std::string result = std::to_string( xSize );
      if( dimension > 1 )
         result += "x" + std::to_string( ySize );
      if( dimension > 2 )
         result += "x" + std::to_string( zSize );
      return result;
   }

   //! \brief Maximal number of non-zero elements in one row.
   [[nodiscard]] __cuda_callable__
   Index
   getStencilSize() const
   {
      return 2 * dimension + 1;
   }

   //! \brief Number of non-zero elements of the whole matrix.
   [[nodiscard]] __cuda_callable__
   Index
   getNonzeroElementsCount() const
   {
      auto interior = []( Index n ) -> Index
      {
         return n > 2 ? n - 2 : 0;
      };
      Index interiorNodes = interior( xSize );
      if( dimension > 1 )
         interiorNodes *= interior( ySize );
      if( dimension > 2 )
         interiorNodes *= interior( zSize );
      return getRows() - interiorNodes + interiorNodes * getStencilSize();
   }

   [[nodiscard]] __cuda_callable__
   bool
   isBoundaryRow( Index row ) const
   {
      const Index i = row % xSize;
      if( i == 0 || i == xSize - 1 )
         return true;
      if( dimension > 1 ) {
         const Index j = ( row / xSize ) % ySize;
         if( j == 0 || j == ySize - 1 )
            return true;
      }
      if( dimension > 2 ) {
         const Index k = row / ( xSize * ySize );
         if( k == 0 || k == zSize - 1 )
            return true;
      }
      return false;
   }

   //! \brief Number of non-zero elements in the given row.
   [[nodiscard]] __cuda_callable__
   Index
   getRowLength( Index row ) const
   {
      return isBoundaryRow( row ) ? 1 : getStencilSize();
   }

   /**
    * \brief Calls `f( localIdx, column, value )` for all non-zero elements of the row.
    *
    * The elements are visited in ascending order of the column indexes.
    */
   template< typename Function >
   __cuda_callable__
   void
   forRowElements( Index row, Function&& f ) const
   {
      if( isBoundaryRow( row ) ) {
         f( Index( 0 ), row, Real( 1 ) );
         return;
      }
      const Real hx = hxSquareInverse;
      const Real hy = hySquareInverse;
      const Real hz = hzSquareInverse;
      Index localIdx = 0;
      if( dimension > 2 )
         f( localIdx++, row - xSize * ySize, -hz );
      if( dimension > 1 )
         f( localIdx++, row - xSize, -hy );
      f( localIdx++, row - 1, -hx );
      Real diagonal = 2 * hx;
      if( dimension > 1 )
         diagonal += 2 * hy;
      if( dimension > 2 )
         diagonal += 2 * hz;
      f( localIdx++, row, diagonal );
      f( localIdx++, row + 1, -hx );
      if( dimension > 1 )
         f( localIdx++, row + xSize, -hy );
      if( dimension > 2 )
         f( localIdx++, row + xSize * ySize, -hz );
   }

   /**
    * \brief Computes the CSR row pointers (offsets of the rows in the COO/CSR arrays).
    *
    * \param offsets is resized to `getRows() + 1` elements and filled with the row offsets.
    */
   template< typename Device >
   void
   computeRowOffsets( Containers::Vector< Index, Device, Index >& offsets ) const
   {
      const Index rows = getRows();
      offsets.setSize( rows + 1 );
      auto offsetsView = offsets.getView();
      const LaplaceProblem problem = *this;
      Algorithms::parallelFor< Device >(
         0,
         rows,
         [ = ] __cuda_callable__( Index row ) mutable
         {
            offsetsView[ row ] = problem.getRowLength( row );
         } );
      offsets.setElement( rows, 0 );
      Algorithms::inplaceExclusiveScan( offsets );
   }

   /**
    * \brief Writes the matrix elements in the COO format into the given arrays.
    *
    * Any of the pointers \e rowIndexes, \e columnIndexes and \e values may be \e nullptr
    * in which case the corresponding array is not written. The arrays must be allocated
    * on \e Device and have at least \ref getNonzeroElementsCount elements.
    *
    * \param offsets are the row offsets computed by \ref computeRowOffsets.
    */
   template< typename Device, typename RowIndex, typename ColumnIndex, typename Value >
   void
   writeCOO(
      const Containers::Vector< Index, Device, Index >& offsets,
      RowIndex* rowIndexes,
      ColumnIndex* columnIndexes,
      Value* values ) const
   {
      const auto offsetsView = offsets.getConstView();
      const LaplaceProblem problem = *this;
      Algorithms::parallelFor< Device >(
         0,
         getRows(),
         [ = ] __cuda_callable__( Index row ) mutable
         {
            const Index begin = offsetsView[ row ];
            problem.forRowElements(
               row,
               [ & ]( Index localIdx, Index column, Real value )
               {
                  if( rowIndexes != nullptr )
                     rowIndexes[ begin + localIdx ] = row;
                  if( columnIndexes != nullptr )
                     columnIndexes[ begin + localIdx ] = column;
                  if( values != nullptr )
                     values[ begin + localIdx ] = value;
               } );
         } );
   }

   /**
    * \brief Compares a matrix given by host CSR arrays with the exact matrix.
    *
    * The order of the elements within each row does not matter (e.g. Hypre stores the
    * diagonal element first). Explicit zeros and padding elements must be removed by the
    * caller.
    *
    * \return Empty string on success, otherwise a description of the first difference.
    */
   template< typename RowPointer, typename ColumnIndex, typename Value >
   [[nodiscard]] std::string
   compare(
      const std::vector< RowPointer >& rowPointers,
      const std::vector< ColumnIndex >& columnIndexes,
      const std::vector< Value >& values ) const
   {
      const Index rows = getRows();
      if( (Index) rowPointers.size() != rows + 1 )
         return "wrong number of rows: " + std::to_string( (long) rowPointers.size() - 1 ) + " instead of "
              + std::to_string( rows );
      if( (Index) rowPointers[ rows ] != getNonzeroElementsCount() )
         return "wrong number of non-zero elements: " + std::to_string( (long) rowPointers[ rows ] ) + " instead of "
              + std::to_string( getNonzeroElementsCount() );
      for( Index row = 0; row < rows; row++ ) {
         const Index begin = rowPointers[ row ];
         const Index end = rowPointers[ row + 1 ];
         if( end - begin != getRowLength( row ) )
            return "wrong length of row " + std::to_string( row );
         std::string error;
         forRowElements(
            row,
            [ & ]( Index localIdx, Index column, Real value )
            {
               if( ! error.empty() )
                  return;
               Index idx = begin;
               while( idx < end && (Index) columnIndexes[ idx ] != column )
                  idx++;
               if( idx == end )
                  error = "missing element at row " + std::to_string( row ) + ", column " + std::to_string( column );
               else if( Real( values[ idx ] ) != value )
                  error = "wrong value at row " + std::to_string( row ) + ", column " + std::to_string( column ) + ": "
                        + std::to_string( (double) values[ idx ] ) + " instead of " + std::to_string( (double) value );
            } );
         if( ! error.empty() )
            return error;
      }
      return "";
   }

protected:
   int dimension = 1;
   Index xSize = 0;
   Index ySize = 1;
   Index zSize = 1;
   Real hxSquareInverse = 1;
   Real hySquareInverse = 1;
   Real hzSquareInverse = 1;
};

}  // namespace TNL::Benchmarks::MatrixAssembly
