#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forAllElementsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   auto matrixView = matrix.getView();

   auto f = [] __cuda_callable__( int rowIdx, int columnIdx, int globalIdx, double& value )
   {
      if( rowIdx >= columnIdx )
         value = rowIdx + columnIdx;
   };

   matrixView.forAllElements( f );  // or matrix.forAllElements
   std::cout << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host:\n";
   forAllElementsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA device:\n";
   forAllElementsExample< TNL::Devices::Cuda >();
#endif
}
