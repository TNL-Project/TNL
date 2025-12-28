#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
reduceRowsExample()
{
   /***
    * Create a 7x7 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 7, 7 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values.
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx * 10 + columnIdx;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Dense matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Compute sums for rows 2-5 (range variant).
    */
   TNL::Containers::Vector< double, Device > rangeSums( 4 );  // 4 rows
   auto rangeSums_view = rangeSums.getView();

   auto fetch = [] __cuda_callable__( int rowIdx, int columnIdx, const double& value ) -> double
   {
      return value;
   };

   auto keepRange = [ = ] __cuda_callable__( int rowIdx, const double& sum ) mutable
   {
      rangeSums_view[ rowIdx - 2 ] = sum;  // Offset by begin index
   };

   TNL::Matrices::reduceRows( matrix, 2, 6, fetch, TNL::Plus{}, keepRange );

   std::cout << "Sums for rows 2-5: " << rangeSums << std::endl;

   /***
    * Compute sums for specific rows (array variant).
    */
   TNL::Containers::Array< int, Device > rowIndexes{ 0, 2, 4, 6 };
   TNL::Containers::Vector< double, Device > arraySums( matrix.getRows() );
   TNL::Containers::Vector< double, Device > compressedSums( rowIndexes.getSize() );
   auto arraySums_view = arraySums.getView();
   auto compressedSums_view = compressedSums.getView();

   auto keepArray = [ = ] __cuda_callable__( int indexOfRowIdx, int rowIdx, const double& sum ) mutable
   {
      arraySums_view[ rowIdx ] = sum;
      compressedSums_view[ indexOfRowIdx ] = sum;
   };

   TNL::Matrices::reduceRows( matrix, rowIndexes, fetch, TNL::Plus{}, keepArray );

   std::cout << "Sums for rows [0, 2, 4, 6]: " << arraySums << std::endl;
   std::cout << "Compressed sums for rows [0, 2, 4, 6]: " << compressedSums.getView( 0, rowIndexes.getSize() ) << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceRowsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceRowsExample< TNL::Devices::Cuda >();
#endif
}
