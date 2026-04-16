#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forRowsIfExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 0.0 );

   /***
    * Use forRowsIf to process only even-numbered rows.
    */
   auto evenRowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;
   };

   using DenseRowView = typename TNL::Matrices::DenseMatrix< double, Device >::RowView;

   auto processDenseRow = [] __cuda_callable__( DenseRowView & row )
   {
      const int rowIdx = row.getRowIndex();
      for( int i = 0; i < row.getSize(); i++ )
         row.setValue( i, rowIdx + i );
   };

   TNL::Matrices::forRowsIf( denseMatrix, 0, 5, evenRowCondition, processDenseRow );
   std::cout << "Dense matrix with only even rows set:\n";
   std::cout << denseMatrix << '\n';

   /***
    * Create a 5x5 sparse matrix.
    */
   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 3, 3, 3, 1 }, 5 );

   /***
    * Use forRowsIf to process only rows where rowIdx > 0 and rowIdx < 4.
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

   TNL::Matrices::forRowsIf( sparseMatrix, 0, 5, innerRowCondition, processSparseRow );
   std::cout << "Sparse matrix with only inner rows (1-3) set:\n";
   std::cout << sparseMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   forRowsIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   forRowsIfExample< TNL::Devices::Cuda >();
#endif
}
