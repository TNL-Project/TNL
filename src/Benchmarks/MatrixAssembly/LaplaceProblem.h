// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <string>

#include <TNL/Backend/Macros.h>

#include "AssemblyProblem.h"

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
 * object can be captured by value in lambda functions, so it serves as its own view
 * (see \ref AssemblyProblemBase).
 */
template< typename Real, typename Index >
struct LaplaceProblem : public AssemblyProblemBase< LaplaceProblem< Real, Index >, Real, Index >
{
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

   [[nodiscard]] std::string
   getName() const
   {
      return "Laplace " + std::to_string( dimension ) + "D";
   }

   //! \brief The view used in kernels on \e Device is the problem itself.
   template< typename Device >
   [[nodiscard]] LaplaceProblem
   getView() const
   {
      return *this;
   }

   [[nodiscard]] __cuda_callable__
   Index
   getRows() const
   {
      return xSize * ySize * zSize;
   }

   [[nodiscard]] std::string
   getSizeString() const
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

   //! \brief Maximal number of non-zero elements in one row of the matrix.
   [[nodiscard]] __cuda_callable__
   Index
   getMaxRowLength() const
   {
      return getNonzeroElementsCount() > getRows() ? getStencilSize() : 1;
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
