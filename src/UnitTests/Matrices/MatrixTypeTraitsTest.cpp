#include "gtest/gtest.h"

#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Matrices/LambdaMatrix.h>

using namespace TNL::Matrices;

TEST( MatrixTypeTraitsTest, is_matrix_type_v )
{
   static_assert( ! is_matrix_type_v< float > );
   static_assert( ! is_matrix_type_v< int > );
   static_assert( ! is_matrix_type_v< TNL::Devices::Host > );

   using DenseMatrix = DenseMatrix< float, TNL::Devices::Host, int >;
   static_assert( is_matrix_type_v< DenseMatrix > );
   static_assert( is_matrix_type_v< typename DenseMatrix::ViewType > );
   static_assert( is_matrix_type_v< typename DenseMatrix::ConstViewType > );

   using CSR = SparseMatrix< float, TNL::Devices::Host, int, GeneralMatrix, TNL::Algorithms::Segments::CSR >;
   static_assert( is_matrix_type_v< CSR > );
   static_assert( is_matrix_type_v< typename CSR::ViewType > );
   static_assert( is_matrix_type_v< typename CSR::ConstViewType > );

   using Multidiagonal = MultidiagonalMatrix< float, TNL::Devices::Host, int >;
   static_assert( is_matrix_type_v< Multidiagonal > );
   static_assert( is_matrix_type_v< typename Multidiagonal::ViewType > );
   static_assert( is_matrix_type_v< typename Multidiagonal::ConstViewType > );

   using Tridiagonal = TridiagonalMatrix< float, TNL::Devices::Host, int >;
   static_assert( is_matrix_type_v< Tridiagonal > );
   static_assert( is_matrix_type_v< typename Tridiagonal::ViewType > );
   static_assert( is_matrix_type_v< typename Tridiagonal::ConstViewType > );

   auto matrixElements = []( int rows, int columns, int rowIdx, int localIdx, int& columnIdx, float& value ) {};
   auto rowLengths = []( int rows, int columns, int rowIdx ) -> int
   {
      return 0;
   };
   using Lambda = LambdaMatrix< decltype( matrixElements ), decltype( rowLengths ) >;
   static_assert( is_matrix_type_v< Lambda > );
}

TEST( MatrixTypeTraitsTest, is_dense_matrix_type_v )
{
   static_assert( ! is_dense_matrix_type_v< float > );
   static_assert( ! is_dense_matrix_type_v< int > );
   static_assert( ! is_dense_matrix_type_v< TNL::Devices::Host > );

   using DenseMatrix = DenseMatrix< float, TNL::Devices::Host, int >;
   static_assert( is_dense_matrix_type_v< DenseMatrix > );
   static_assert( is_dense_matrix_type_v< typename DenseMatrix::ViewType > );
   static_assert( is_dense_matrix_type_v< typename DenseMatrix::ConstViewType > );

   using CSR = SparseMatrix< float, TNL::Devices::Host, int, GeneralMatrix, TNL::Algorithms::Segments::CSR >;
   static_assert( ! is_dense_matrix_type_v< CSR > );
   static_assert( ! is_dense_matrix_type_v< typename CSR::ViewType > );
   static_assert( ! is_dense_matrix_type_v< typename CSR::ConstViewType > );

   using Multidiagonal = MultidiagonalMatrix< float, TNL::Devices::Host, int >;
   static_assert( ! is_dense_matrix_type_v< Multidiagonal > );
   static_assert( ! is_dense_matrix_type_v< typename Multidiagonal::ViewType > );
   static_assert( ! is_dense_matrix_type_v< typename Multidiagonal::ConstViewType > );

   using Tridiagonal = TridiagonalMatrix< float, TNL::Devices::Host, int >;
   static_assert( ! is_dense_matrix_type_v< Tridiagonal > );
   static_assert( ! is_dense_matrix_type_v< typename Tridiagonal::ViewType > );
   static_assert( ! is_dense_matrix_type_v< typename Tridiagonal::ConstViewType > );

   auto matrixElements = []( int rows, int columns, int rowIdx, int localIdx, int& columnIdx, float& value ) {};
   auto rowLengths = []( int rows, int columns, int rowIdx ) -> int
   {
      return 0;
   };
   using Lambda = LambdaMatrix< decltype( matrixElements ), decltype( rowLengths ) >;
   static_assert( ! is_dense_matrix_type_v< Lambda > );
}

#include "../main.h"
