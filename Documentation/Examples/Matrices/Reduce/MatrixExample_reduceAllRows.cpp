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
reduceAllRowsExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   matrix.setValue( 0.0 );

   /***
    * Fill the matrix with values: row i has values i, i+1, i+2, ...
    */
   auto fillMatrix = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx + columnIdx + 1;
   };
   TNL::Matrices::forAllElements( matrix, fillMatrix );

   std::cout << "Matrix:" << std::endl;
   std::cout << matrix << std::endl;

   /***
    * Compute row sums using reduceAllRows.
    */
   TNL::Containers::Vector< double, Device > rowSums( matrix.getRows() );
   auto rowSums_view = rowSums.getView();

   auto fetch = [] __cuda_callable__( int rowIdx, int columnIdx, const double& value ) -> double
   {
      return value;
   };

   auto store = [ = ] __cuda_callable__( int rowIdx, const double& sum ) mutable
   {
      rowSums_view[ rowIdx ] = sum;
   };

   TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Plus{}, store );

   std::cout << "Row sums: " << rowSums << std::endl;

   /***
    * Compute row maxima.
    */
   TNL::Containers::Vector< double, Device > rowMaxima( matrix.getRows() );
   auto rowMaxima_view = rowMaxima.getView();

   auto storeMax = [ = ] __cuda_callable__( int rowIdx, const double& max ) mutable
   {
      rowMaxima_view[ rowIdx ] = max;
   };

   TNL::Matrices::reduceAllRows( matrix, fetch, TNL::Max{}, storeMax );

   std::cout << "Row maxima: " << rowMaxima << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:" << std::endl;
   reduceAllRowsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << std::endl << "Running on CUDA device:" << std::endl;
   reduceAllRowsExample< TNL::Devices::Cuda >();
#endif
}
