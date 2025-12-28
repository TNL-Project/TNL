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
reduceRowsIfExample()
{
   /***
    * Create an 8x8 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 8, 8 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values.
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx * 10 + columnIdx;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Compute sums for rows 2-6, but only for even-indexed rows (range + condition).
    */
   TNL::Containers::Vector< double, Device > rangeSums( 5 );
   auto rangeSums_view = rangeSums.getView();

   auto fetch = [] __cuda_callable__( int rowIdx, int columnIdx, const double& value ) -> double
   {
      return value;
   };

   auto evenRowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx % 2 == 0;
   };

   auto keepRange = [ = ] __cuda_callable__( int indexOfRowIdx, int rowIdx, const double& sum ) mutable
   {
      rangeSums_view[ rowIdx - 2 ] = sum;
   };

   rangeSums.setValue( -1.0 );

   TNL::Matrices::reduceRowsIf( matrix, 2, 7, evenRowCondition, fetch, TNL::Plus{}, keepRange );

   std::cout << "Sums for rows 2-6 (only even indices, others show -1): " << rangeSums << std::endl;

   /***
    * Compute maxima for specific rows, but only if row index > 3 (array + condition).
    */
   TNL::Containers::Vector< double, Device > arrayMaxima( matrix.getRows() );
   TNL::Containers::Vector< double, Device > compressedMaxima( matrix.getRows() );
   auto arrayMaxima_view = arrayMaxima.getView();
   auto compressedMaxima_view = compressedMaxima.getView();

   auto rowCondition = [] __cuda_callable__( int rowIdx ) -> bool
   {
      return rowIdx > 3 && rowIdx % 2 == 1;  // Only odd row indices greater than 3
   };

   auto keepArray = [ = ] __cuda_callable__( int indexOfRowIdx, int rowIdx, const double& max ) mutable
   {
      arrayMaxima_view[ rowIdx ] = max;
      compressedMaxima_view[ indexOfRowIdx ] = max;
   };

   arrayMaxima.setValue( -1.0 );

   auto processedRows = TNL::Matrices::reduceAllRowsIf( matrix, rowCondition, fetch, TNL::Max{}, keepArray );

   std::cout << "Maxima for rows [1, 3, 5, 7] where rowIdx > 3: " << arrayMaxima << std::endl;
   std::cout << "Compressed maxima for rows [1, 3, 5, 7] where rowIdx > 3: " << compressedMaxima.getView( 0, processedRows )
             << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceRowsIfExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceRowsIfExample< TNL::Devices::Cuda >();
#endif
}
