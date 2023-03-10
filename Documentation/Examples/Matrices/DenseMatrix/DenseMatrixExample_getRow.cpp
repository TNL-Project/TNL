#include <iostream>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>

template< typename Device >
void getRowExample()
{
   using MatrixType = TNL::Matrices::DenseMatrix< double, Device >;
   TNL::Pointers::SharedPointer< MatrixType > matrix( 5, 5 );

   MatrixType* matrix_device = &matrix.template modifyData< Device >();
   auto f = [=] __cuda_callable__ ( int rowIdx ) mutable {
      auto row = matrix_device->getRow( rowIdx );
      row.setValue( rowIdx, 10 * ( rowIdx + 1 ) );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use DenseMatrixView. See
    * DenseMatrixView::getRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();

   /***
    * Set the matrix elements.
    */
   TNL::Algorithms::parallelFor< Device >( 0, matrix->getRows(), f );
   std::cout << matrix << std::endl;
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
