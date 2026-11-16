#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

template< typename Device >
void
forElementsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   auto matrixView = matrix.getView();

   auto f = [] __cuda_callable__(
               int rowIdx, int columnIdx_, int columnIdx, double& value )  // columnIdx is the same as localIdx for dense matrix
   {
      if( columnIdx <= rowIdx )
         value = rowIdx + columnIdx;
   };

   matrixView.forElements( 0, matrix.getRows(), f );  // or matrix.forElements
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
