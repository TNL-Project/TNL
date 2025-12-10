#include <iostream>
#include <iomanip>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
getElements()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix(
      // number of matrix rows
      5,
      // number of matrix columns
      5,
      // matrix elements definition
      {
         // clang-format off
         { 0, 0,  2.0 },
         { 1, 0, -1.0 }, { 1, 1, 2.0 }, { 1, 2, -1.0 },
         { 2, 1, -1.0 }, { 2, 2, 2.0 }, { 2, 3, -1.0 },
         { 3, 2, -1.0 }, { 3, 3, 2.0 }, { 3, 4, -1.0 },
         { 4, 4,  2.0 },
         // clang-format on
      } );

   for( int i = 0; i < 5; i++ ) {
      for( int j = 0; j < 5; j++ )
         std::cout << std::setw( 5 ) << matrix.getElement( i, j );
      std::cout << '\n';
   }
}

int
main( int argc, char* argv[] )
{
   std::cout << "Get elements on host:\n";
   getElements< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Get elements on CUDA device:\n";
   getElements< TNL::Devices::Cuda >();
#endif
}
