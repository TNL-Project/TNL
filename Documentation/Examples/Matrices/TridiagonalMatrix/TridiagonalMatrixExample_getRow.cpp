#include <iostream>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   const int matrixSize( 5 );
   using MatrixType = TNL::Matrices::TridiagonalMatrix< double, Device >;
   TNL::Pointers::SharedPointer< MatrixType > matrix(
      matrixSize,  // number of matrix rows
      matrixSize  // number of matrix columns
   );

   MatrixType* matrix_device = &matrix.template modifyData< Device >();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrix_device->getRow( rowIdx );
      if( rowIdx > 0 )
         row.setElement( 0, -1.0 );  // elements below the diagonal
      row.setElement( 1, 2.0 );      // elements on the diagonal
      if( rowIdx < matrixSize - 1 )  // elements above the diagonal
         row.setElement( 2, -1.0 );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use TridiagonalMatrixView. See
    * TridiagonalMatrixView::getRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();

   /***
    * Set the matrix elements.
    */
   TNL::Algorithms::parallelFor< Device >( 0, matrix->getRows(), f );
   std::cout << std::endl << *matrix << std::endl;
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
