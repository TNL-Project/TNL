#include <iostream>
#include <functional>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   const int matrixSize = 5;
   auto diagonalsOffsets = { -2, -1, 0 };
   using MatrixType = TNL::Matrices::MultidiagonalMatrix< double, Device >;
   TNL::Pointers::SharedPointer< MatrixType > matrix (
      matrixSize,  // number of matrix rows
      matrixSize,  // number of matrix columns
      diagonalsOffsets );
   matrix->setElements(
      {  { 0.0, 0.0, 1.0 },
         { 0.0, 2.0, 1.0 },
         { 3.0, 2.0, 1.0 },
         { 3.0, 2.0, 1.0 },
         { 3.0, 2.0, 1.0 } } );

   /***
    * Fetch lambda function returns diagonal element in each row.
    */
   const MatrixType* matrix_device = &matrix.template getData< Device >();
   auto fetch = [=] __cuda_callable__ ( int rowIdx ) mutable -> double {
      auto row = matrix_device->getRow( rowIdx );
      return row.getValue( 2 ); // get value from subdiagonal with index 2, i.e. the main diagonal
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use MultidiagonalMatrixView. See
    * MultidiagonalMatrixView::getConstRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();

   /***
    * Compute the matrix trace.
    */
   int trace = TNL::Algorithms::reduce< Device >( 0, matrix->getRows(), fetch, std::plus<>{}, 0 );
   std::cout << "Matrix reads as: " << std::endl << *matrix << std::endl;
   std::cout << "Matrix trace is: " << trace << "." << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host: " << std::endl;
   getRowExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Getting matrix rows on CUDA device: " << std::endl;
   getRowExample< TNL::Devices::Cuda >();
#endif
}
