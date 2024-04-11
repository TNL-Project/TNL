#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
getInPlaceTranspositionExample()
{
   // clang-format off
   TNL::Matrices::DenseMatrix< double, Device > matrix{
      { 1, 2, 3, 4 },
      { 6, 7, 8, 9 },
      { 10, 11, 12, 13 },
      { 14, 15, 16, 17 }
   };
   // clang-format on

   std::cout << "Dense matrix: " << std::endl << matrix << std::endl;

   matrix.getInPlaceTransposition();

   std::cout << "Transposed dense matrix: " << std::endl << matrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ... " << std::endl;
   getInPlaceTranspositionExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA GPU ... " << std::endl;
   getInPlaceTranspositionExample< TNL::Devices::Cuda >();
#endif
}
