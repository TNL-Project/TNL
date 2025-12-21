#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forAllRowsExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 1.0 );

   /***
    * Use forAllRows to multiply each row by its row index.
    */
   using DenseRowView = typename TNL::Matrices::DenseMatrix< double, Device >::RowView;

   auto multiplyByRowIndex = [] __cuda_callable__( DenseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      for( int i = 0; i < row.getSize(); i++ )
         row.setValue( i, row.getValue( i ) * rowIdx );
   };

   TNL::Matrices::forAllRows( denseMatrix, multiplyByRowIndex );
   std::cout << "Dense matrix with rows multiplied by row index:" << std::endl;
   std::cout << denseMatrix << std::endl;

   /***
    * Create a 5x5 sparse matrix.
    */
   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 3, 3, 3, 1 }, 5 );

   /***
    * Use forAllRows to set up a tridiagonal matrix.
    */
   using SparseRowView = typename TNL::Matrices::SparseMatrix< double, Device >::RowView;

   auto setupTridiagonal = [] __cuda_callable__( SparseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      const int size = 5;

      if( rowIdx == 0 )
         row.setElement( 0, rowIdx, 2.0 );
      else if( rowIdx == size - 1 )
         row.setElement( 0, rowIdx, 2.0 );
      else {
         row.setElement( 0, rowIdx - 1, 1.0 );
         row.setElement( 1, rowIdx, 2.0 );
         row.setElement( 2, rowIdx + 1, 1.0 );
      }
   };

   TNL::Matrices::forAllRows( sparseMatrix, setupTridiagonal );
   std::cout << "Sparse tridiagonal matrix:" << std::endl;
   std::cout << sparseMatrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   forAllRowsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   forAllRowsExample< TNL::Devices::Cuda >();
#endif
}
