#include <iostream>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forElementsExample()
{
   TNL::Matrices::MultidiagonalMatrix< double, Device > matrix( 5,              // number of matrix rows
                                                                5,              // number of matrix columns
                                                                { 0, 1, 2 } );  // matrix diagonals offsets

   auto matrixView = matrix.getView();

   auto condition = [] __cuda_callable__( int rowIdx )
   {
      return rowIdx % 2 == 0;
   };
   auto f = [] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double& value )
   {
      value = rowIdx + 1;
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
