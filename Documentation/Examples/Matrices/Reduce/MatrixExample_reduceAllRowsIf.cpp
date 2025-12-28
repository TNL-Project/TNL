#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
reduceAllRowsIfExample()
{
   /***
    * Create a 6x6 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 6, 6 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values.
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx + columnIdx;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Compute sums only for rows with even indices.
    */
   TNL::Containers::Vector< double, Device > evenRowSums( matrix.getRows() );
   TNL::Containers::Vector< double, Device > compressedEvenRowSums( matrix.getRows() );
   auto evenRowSums_view = evenRowSums.getView();
   auto compressedEvenRowSums_view = compressedEvenRowSums.getView();

   auto fetch = [] __cuda_callable__( int rowIdx, int columnIdx, const double& value ) -> double
   {
      return value;
   };

   auto rowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;  // Only even row indices
   };

   auto keep = [ = ] __cuda_callable__( int indexOfRowIdx, int rowIdx, const double& sum ) mutable
   {
      evenRowSums_view[ rowIdx ] = sum;
      compressedEvenRowSums_view[ indexOfRowIdx ] = sum;
   };

   // Initialize with -1 to see which rows were processed
   evenRowSums.setValue( -1.0 );

   auto evenRowsCount = TNL::Matrices::reduceAllRowsIf( matrix, rowCondition, fetch, TNL::Plus{}, keep );

   std::cout << "Sums for even-indexed rows (odd indices show -1): " << evenRowSums << std::endl;
   std::cout << "Compressed sums for even-indexed rows: " << compressedEvenRowSums.getView( 0, evenRowsCount ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceAllRowsIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceAllRowsIfExample< TNL::Devices::Cuda >();
#endif
}
