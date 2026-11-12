#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

template< typename Device >
void
forAllElementsExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 1, 2, 3, 4, 5 }, 5 );
   auto matrixView = matrix.getView();

   auto f = [] __cuda_callable__( int rowIdx, int localIdx, int& columnIdx, double& value )
   {
      // This is important, some matrix formats may allocate more matrix elements
      // than we requested. These padding elements are processed here as well.
      if( rowIdx >= localIdx ) {
         columnIdx = localIdx;
         value = rowIdx + localIdx + 1;
      }
   };

   matrixView.forAllElements( f );  // or matrix.forAllElements
   std::cout << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host:\n";
   forAllElementsExample< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Creating matrix on GPU (" << TNL::Backend::getBackendName() << "):\n";
   forAllElementsExample< TNL::Devices::GPU >();
#endif
}
