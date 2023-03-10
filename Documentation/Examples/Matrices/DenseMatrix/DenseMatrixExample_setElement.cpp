#include <iostream>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Pointers/SharedPointer.h>
#include <TNL/Pointers/SmartPointersRegister.h>

template< typename Device >
void setElements()
{
   using MatrixType = TNL::Matrices::DenseMatrix< double, Device >;
   TNL::Pointers::SharedPointer< MatrixType > matrix( 5, 5 );
   for( int i = 0; i < 5; i++ )
      matrix->setElement( i, i, i );

   std::cout << "Matrix set from the host:" << std::endl;
   std::cout << *matrix << std::endl;

   MatrixType* matrix_device = &matrix.template modifyData< Device >();
   auto f = [=] __cuda_callable__ ( const TNL::Containers::StaticArray< 2, int >& i ) mutable {
      matrix_device->addElement( i[ 0 ], i[ 1 ], 5.0 );
   };

   /***
    * For the case when Device is CUDA device we need to synchronize smart
    * pointers. To avoid this you may use DenseMatrixView. See
    * DenseMatrixView::getRow example for details.
    */
   TNL::Pointers::synchronizeSmartPointersOnDevice< Device >();

   TNL::Containers::StaticArray< 2, int > begin = { 0, 0 };
   TNL::Containers::StaticArray< 2, int > end = { 5, 5 };
   TNL::Algorithms::parallelFor< Device >( begin, end, f );

   std::cout << "Matrix set from its native device:" << std::endl;
   std::cout << *matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Set elements on host:" << std::endl;
   setElements< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Set elements on CUDA device:" << std::endl;
   setElements< TNL::Devices::Cuda >();
#endif
}
