#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forRowsExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 0.0 );

   /***
    * Use forRows to process rows 1 to 4 (inclusive).
    */
   using DenseRowView = typename TNL::Matrices::DenseMatrix< double, Device >::RowView;

   auto processDenseRow = [] __cuda_callable__( DenseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      for( int i = 0; i < row.getSize(); i++ )
         if( i <= rowIdx )
            row.setValue( i, rowIdx + i );
   };

   TNL::Matrices::forRows( denseMatrix, 1, 4, processDenseRow );
   std::cout << "Dense matrix with rows 1-3 processed:" << std::endl;
   std::cout << denseMatrix << std::endl;

   /***
    * Create a 5x5 sparse matrix.
    */
   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 3, 3, 3, 1 }, 5 );

   /***
    * Use forRows to set up a tridiagonal structure.
    */
   using SparseRowView = typename TNL::Matrices::SparseMatrix< double, Device >::RowView;

   auto processSparseRow = [] __cuda_callable__( SparseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      const int size = 5;

      if( rowIdx == 0 )
         row.setElement( 0, rowIdx, 2.0 );  // diagonal element
      else if( rowIdx == size - 1 )
         row.setElement( 0, rowIdx, 2.0 );  // diagonal element
      else {
         row.setElement( 0, rowIdx - 1, 1.0 );  // below diagonal
         row.setElement( 1, rowIdx, 2.0 );      // diagonal
         row.setElement( 2, rowIdx + 1, 1.0 );  // above diagonal
      }
   };

   TNL::Matrices::forRows( sparseMatrix, 0, 5, processSparseRow );
   std::cout << "Sparse tridiagonal matrix:" << std::endl;
   std::cout << sparseMatrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   forRowsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   forRowsExample< TNL::Devices::Cuda >();
#endif
}
