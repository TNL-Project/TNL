#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forRowsWithIndexesExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 0.0 );

   /***
    * Create a vector with row indexes to process.
    */
   TNL::Containers::Vector< int, Device > rowIndexes{ 0, 2, 4 };

   /***
    * Use forRows with row indexes to set specific matrix rows.
    */
   using DenseRowView = typename TNL::Matrices::DenseMatrix< double, Device >::RowView;

   auto processDenseRow = [] __cuda_callable__( DenseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      for( int i = 0; i < row.getSize(); i++ )
         row.setValue( i, rowIdx * 10 + i );
   };

   TNL::Matrices::forRows( denseMatrix, rowIndexes, processDenseRow );
   std::cout << "Dense matrix with selected rows (0, 2, 4) set:" << std::endl;
   std::cout << denseMatrix << std::endl;

   /***
    * Create a 5x5 sparse matrix.
    */
   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 3, 3, 3, 1 }, 5 );

   /***
    * Use forRows with row indexes to process selected rows.
    */
   using SparseRowView = typename TNL::Matrices::SparseMatrix< double, Device >::RowView;

   auto processSparseRow = [] __cuda_callable__( SparseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      const int size = 5;

      if( rowIdx == 0 || rowIdx == size - 1 )
         row.setElement( 0, rowIdx, 2.0 );
      else {
         row.setElement( 0, rowIdx - 1, 1.0 );
         row.setElement( 1, rowIdx, 2.0 );
         row.setElement( 2, rowIdx + 1, 1.0 );
      }
   };

   TNL::Matrices::forRows( sparseMatrix, rowIndexes, processSparseRow );
   std::cout << "Sparse matrix with selected rows (0, 2, 4) set:" << std::endl;
   std::cout << sparseMatrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   forRowsWithIndexesExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   forRowsWithIndexesExample< TNL::Devices::Cuda >();
#endif
}
