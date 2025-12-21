#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forAllRowsIfExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 0.0 );

   /***
    * Use forAllRowsIf to process only odd-numbered rows.
    */
   auto oddRowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 1;
   };

   using DenseRowView = typename TNL::Matrices::DenseMatrix< double, Device >::RowView;

   auto processDenseRow = [] __cuda_callable__( DenseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      for( int i = 0; i < row.getSize(); i++ )
         row.setValue( i, rowIdx * 10 + i );
   };

   TNL::Matrices::forAllRowsIf( denseMatrix, oddRowCondition, processDenseRow );
   std::cout << "Dense matrix with only odd rows set:" << std::endl;
   std::cout << denseMatrix << std::endl;

   /***
    * Create a 5x5 sparse matrix.
    */
   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 3, 3, 3, 1 }, 5 );

   /***
    * Use forAllRowsIf to process all rows except the first and last.
    */
   auto innerRowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx > 0 && rowIdx < 4;
   };

   using SparseRowView = typename TNL::Matrices::SparseMatrix< double, Device >::RowView;

   auto processSparseRow = [] __cuda_callable__( SparseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      row.setElement( 0, rowIdx - 1, 1.0 );
      row.setElement( 1, rowIdx, 2.0 );
      row.setElement( 2, rowIdx + 1, 1.0 );
   };

   TNL::Matrices::forAllRowsIf( sparseMatrix, innerRowCondition, processSparseRow );
   std::cout << "Sparse matrix with inner rows (1-3) set:" << std::endl;
   std::cout << sparseMatrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   forAllRowsIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:" << std::endl;
   forAllRowsIfExample< TNL::Devices::Cuda >();
#endif
}
