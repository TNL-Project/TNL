#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forElementsExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 1, 2, 3, 4, 5 }, 5 );
   auto matrixView = matrix.getView();

   auto condition = [] __cuda_callable__( int rowIdx )
   {
      return rowIdx % 2 == 0;
   };
   auto f = [] __cuda_callable__( int rowIdx, int localIdx, int& columnIdx, double& value )
   {
      // This is important, some matrix formats may allocate more matrix elements
      // than we requested. These padding elements are processed here as well.
      if( rowIdx >= localIdx ) {
         columnIdx = localIdx;
         value = rowIdx + localIdx + 1;
      }
   };

   matrixView.forElementsIf( 0, matrix.getRows(), condition, f );  // or matrix.forElements
   std::cout << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on host:\n";
   forElementsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA device:\n";
   forElementsExample< TNL::Devices::Cuda >();
#endif
}
