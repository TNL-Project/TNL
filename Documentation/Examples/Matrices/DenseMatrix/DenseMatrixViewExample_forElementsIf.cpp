#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forElementsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   auto matrixView = matrix.getView();

   auto condition = [] __cuda_callable__( int rowIdx )
   {
      return rowIdx % 2 == 0;
   };
   auto f = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx + localIdx + 1;
   };

   matrixView.forElementsIf( 0, matrix.getRows(), condition, f );  // or matrix.forElements
   std::cout << matrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host: " << std::endl;
   forElementsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA device: " << std::endl;
   forElementsExample< TNL::Devices::Cuda >();
#endif
}
