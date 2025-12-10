// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>

#include <TNL/Backend/Macros.h>
#include <TNL/DiscreteMath.h>

namespace TNL::Matrices::details {

template< typename Index, bool RowMajorOrder >
class MultidiagonalMatrixIndexer
{
public:
   using IndexType = Index;

   [[nodiscard]] static constexpr bool
   getRowMajorOrder()
   {
      return RowMajorOrder;
   }

   __cuda_callable__
   MultidiagonalMatrixIndexer() = default;

   __cuda_callable__
   MultidiagonalMatrixIndexer( const IndexType& rows,
                               const IndexType& columns,
                               const IndexType& diagonals,
                               const IndexType& nonemptyRows )
   : rows( rows ),
     columns( columns ),
     diagonals( diagonals ),
     nonemptyRows( nonemptyRows )
   {}

   __cuda_callable__
   MultidiagonalMatrixIndexer( const MultidiagonalMatrixIndexer& indexer ) = default;

   __cuda_callable__
   MultidiagonalMatrixIndexer( MultidiagonalMatrixIndexer&& indexer ) = default;

   __cuda_callable__
   MultidiagonalMatrixIndexer&
   operator=( const MultidiagonalMatrixIndexer& indexer ) = default;

   __cuda_callable__
   MultidiagonalMatrixIndexer&
   operator=( MultidiagonalMatrixIndexer&& indexer ) = default;

   void
   set( const IndexType& rows, const IndexType& columns, const IndexType& diagonals, const IndexType& nonemptyRows )
   {
      this->rows = rows;
      this->columns = columns;
      this->diagonals = diagonals;
      this->nonemptyRows = nonemptyRows;
      if( TNL::integerMultiplyOverflow( this->diagonals, this->nonemptyRows ) )
         throw std::overflow_error(
            "MultidiagonalMatrix: multiplication overflow - the storage size required for the matrix is "
            "larger than the maximal value of used index type." );
   }

   [[nodiscard]] __cuda_callable__
   const IndexType&
   getRows() const
   {
      return this->rows;
   }

   [[nodiscard]] __cuda_callable__
   const IndexType&
   getColumns() const
   {
      return this->columns;
   }

   [[nodiscard]] __cuda_callable__
   const IndexType&
   getDiagonals() const
   {
      return this->diagonals;
   }

   [[nodiscard]] __cuda_callable__
   const IndexType&
   getNonemptyRowsCount() const
   {
      return this->nonemptyRows;
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getStorageSize() const
   {
      return this->diagonals * this->nonemptyRows;
   }

   [[nodiscard]] __cuda_callable__
   IndexType
   getGlobalIndex( const Index rowIdx, const Index localIdx ) const
   {
      TNL_ASSERT_GE( localIdx, 0, "" );
      TNL_ASSERT_LT( localIdx, diagonals, "" );
      TNL_ASSERT_GE( rowIdx, 0, "" );
      TNL_ASSERT_LT( rowIdx, this->rows, "" );

      if( RowMajorOrder )
         return diagonals * rowIdx + localIdx;
      else
         return localIdx * nonemptyRows + rowIdx;
   }

protected:
   IndexType rows = 0;
   IndexType columns = 0;
   IndexType diagonals = 0;
   IndexType nonemptyRows = 0;
};

}  // namespace TNL::Matrices::details
