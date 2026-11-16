#include <iostream>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

template< typename Device >
void
forElementsExample()
{
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix(
      5,    // number of matrix rows
      5 );  // number of matrix columns

   auto matrixView = matrix.getView();

   auto condition = [] __cuda_callable__( int rowIdx )
   {
      return rowIdx % 2 == 0;
   };
   auto f = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx + 1;
   };

   TNL::Matrices::forElementsIf( matrixView, 0, matrix.getRows(), condition, f );
   std::cout << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host:\n";
   forElementsExample< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Creating matrix on GPU (" << TNL::Backend::getBackendName() << "):\n";
   forElementsExample< TNL::Devices::GPU >();
#endif
}
