#include <iostream>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

template< typename Device >
void
laplaceOperatorMatrix()
{
   const int gridSize( 6 );
   const int matrixSize = gridSize;
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix(
      matrixSize,  // number of rows
      matrixSize   // number of columns
   );
   matrix.setElements(
      {
         // clang-format off
         {  0.0, 1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         { -1.0, 2.0, -1.0 },
         {  0.0, 1.0 },
         // clang-format on
      } );
   auto view = matrix.getView();

   TNL::Containers::Vector< int, Device > rowLengths;
   view.getCompressedRowLengths( rowLengths );  // or matrix.getCompressedRowLengths
   std::cout << "Laplace operator matrix:\n" << matrix << '\n';
   std::cout << "Compressed row lengths: " << rowLengths << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating Laplace operator matrix on CPU ...\n";
   laplaceOperatorMatrix< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Creating Laplace operator matrix on GPU (" << TNL::Backend::getBackendName() << ") ...\n";
   laplaceOperatorMatrix< TNL::Devices::GPU >();
#endif
}
