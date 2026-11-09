// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>

namespace TNL::Benchmarks::MatrixAssembly {

/**
 * \brief Common part of the problems whose matrices are assembled in the benchmark.
 *
 * A problem (the class \e Problem derived from this class) describes a sparse square matrix and must provide:
 *
 * - `getName()` and `getSizeString()` – strings used in the metadata of the benchmark,
 * - `getRows()`, `getNonzeroElementsCount()` and `getMaxRowLength()`,
 * - `template< typename Device > getView()` – a lightweight object which can be captured by value in lambda
 *   functions running on \e Device and which provides the \ref __cuda_callable__ methods
 *   `getRowLength( row )` and `forRowElements( row, f )`. The latter calls `f( localIdx, column, value )` for
 *   all non-zero elements of the row in ascending order of the column indexes.
 *
 * The view is the only part of the problem used in the kernels, i.e. it represents the data
 * from which the matrix is assembled (the grid or the graph).
 */
template< typename Problem, typename Real, typename Index >
struct AssemblyProblemBase
{
   using RealType = Real;
   using IndexType = Index;

   /**
    * \brief Computes the CSR row pointers (offsets of the rows in the COO/CSR arrays) on \e Device.
    *
    * \param offsets is resized to `getRows() + 1` elements and filled with the row offsets.
    */
   template< typename Device >
   void
   computeRowOffsets( Containers::Vector< Index, Device, Index >& offsets ) const
   {
      const Index rows = problem().getRows();
      offsets.setSize( rows + 1 );
      auto offsetsView = offsets.getView();
      const auto view = problem().template getView< Device >();
      Algorithms::parallelFor< Device >(
         0,
         rows,
         [ = ] __cuda_callable__( Index row ) mutable
         {
            offsetsView[ row ] = view.getRowLength( row );
         } );
      offsets.setElement( rows, 0 );
      Algorithms::inplaceExclusiveScan( offsets );
   }

   /**
    * \brief Writes the matrix elements in the COO format into the given arrays on \e Device.
    *
    * Any of the pointers \e rowIndexes, \e columnIndexes and \e values may be \e nullptr
    * in which case the corresponding array is not written. The arrays must be allocated
    * on \e Device and have at least `getNonzeroElementsCount()` elements.
    * The elements are written sorted in the row-major order.
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
      const auto view = problem().template getView< Device >();
      Algorithms::parallelFor< Device >(
         0,
         problem().getRows(),
         [ = ] __cuda_callable__( Index row ) mutable
         {
            const Index begin = offsetsView[ row ];
            view.forRowElements(
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
      const Index rows = problem().getRows();
      const Index nonzeros = problem().getNonzeroElementsCount();
      if( (Index) rowPointers.size() != rows + 1 )
         return "wrong number of rows: " + std::to_string( (long) rowPointers.size() - 1 ) + " instead of "
              + std::to_string( rows );
      if( (Index) rowPointers[ rows ] != nonzeros )
         return "wrong number of non-zero elements: " + std::to_string( (long) rowPointers[ rows ] ) + " instead of "
              + std::to_string( nonzeros );
      const auto view = problem().template getView< Devices::Host >();
      // elements of one row sorted by the column indexes
      std::vector< std::pair< Index, Real > > rowElements;
      for( Index row = 0; row < rows; row++ ) {
         const Index begin = rowPointers[ row ];
         const Index end = rowPointers[ row + 1 ];
         if( end - begin != view.getRowLength( row ) )
            return "wrong length of row " + std::to_string( row );
         rowElements.clear();
         for( Index idx = begin; idx < end; idx++ )
            rowElements.emplace_back( columnIndexes[ idx ], values[ idx ] );
         std::sort(
            rowElements.begin(),
            rowElements.end(),
            []( const auto& a, const auto& b )
            {
               return a.first < b.first;
            } );
         std::string error;
         view.forRowElements(
            row,
            [ & ]( Index localIdx, Index column, Real value )
            {
               if( ! error.empty() )
                  return;
               const auto& [ storedColumn, storedValue ] = rowElements[ localIdx ];
               if( storedColumn != column )
                  error = "missing element at row " + std::to_string( row ) + ", column " + std::to_string( column );
               else if( storedValue != value )
                  error = "wrong value at row " + std::to_string( row ) + ", column " + std::to_string( column ) + ": "
                        + std::to_string( (double) storedValue ) + " instead of " + std::to_string( (double) value );
            } );
         if( ! error.empty() )
            return error;
      }
      return "";
   }

private:
   [[nodiscard]] const Problem&
   problem() const
   {
      return static_cast< const Problem& >( *this );
   }
};

}  // namespace TNL::Benchmarks::MatrixAssembly
