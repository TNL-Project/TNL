#include <iostream>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
setElements()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   auto matrixView = matrix.getView();
   for( int i = 0; i < 5; i++ )
      matrixView.setElement( i, i, i );  // or matrix.setElement

   std::cout << "Matrix set from the host:" << std::endl;
   std::cout << matrix << std::endl;

   auto f = [ = ] __cuda_callable__( const TNL::Containers::StaticArray< 2, int >& i ) mutable
   {
      matrixView.addElement( i[ 0 ], i[ 1 ], 5.0 );
   };
   TNL::Containers::StaticArray< 2, int > begin = { 0, 0 };
   TNL::Containers::StaticArray< 2, int > end = { 5, 5 };
   TNL::Algorithms::parallelFor< Device >( begin, end, f );

   std::cout << "Matrix set from its native device:" << std::endl;
   std::cout << matrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Set elements on host:" << std::endl;
   setElements< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Set elements on CUDA device:" << std::endl;
   setElements< TNL::Devices::Cuda >();
#endif
}
