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

   auto f = [] __cuda_callable__( int rowIdx, int localIdx, int& columnIdx, double& value )
   {
      // This is important, some matrix formats may allocate more matrix elements
      // than we requested. These padding elements are processed here as well.
      if( rowIdx >= localIdx ) {
         columnIdx = localIdx;
         value = rowIdx + localIdx + 1;
      }
   };

   TNL::Containers::Array< double, Device > rowIndexes{ 0, 2, 4, 0, 1, 3, 5, 9 };
   // The following iterates only over the rows with indexes 0, 2, and 4, i.e. indexes
   // at positions 0, 1, and 2 in the rowIndexes array. The rest is ignored.
   matrixView.forElements( rowIndexes, 0, 2, f );  // or matrix.forElements
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
