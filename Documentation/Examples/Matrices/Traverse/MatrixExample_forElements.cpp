#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forElementsExample()
{
   /***
    * Create a 5x5 dense matrix.
    */
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   denseMatrix.setValue( 0.0 );

   /***
    * Use forElements to set lower triangular matrix elements.
    */
   auto setLowerTriangular = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      if( columnIdx <= rowIdx )
         value = rowIdx + columnIdx;
   };

   TNL::Matrices::forElements( denseMatrix, 0, denseMatrix.getRows(), setLowerTriangular );
   std::cout << "Dense matrix with lower triangular elements set:\n";
   std::cout << denseMatrix << '\n';

   /***
    * Create a 5x5 sparse matrix.
    */
   TNL::Matrices::SparseMatrix< double, Device > sparseMatrix( { 1, 2, 3, 4, 5 }, 5 );

   /***
    * Use forElements to initialize sparse matrix elements.
    */
   auto setSparse = [] __cuda_callable__( int rowIdx, int localIdx, int& columnIdx, double& value )
   {
      if( rowIdx >= localIdx ) {
         columnIdx = localIdx;
         value = rowIdx + localIdx + 1;
      }
   };

   TNL::Matrices::forElements( sparseMatrix, 0, sparseMatrix.getRows(), setSparse );
   std::cout << "Sparse matrix initialized:\n";
   std::cout << sparseMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   forElementsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   forElementsExample< TNL::Devices::Cuda >();
#endif
}
