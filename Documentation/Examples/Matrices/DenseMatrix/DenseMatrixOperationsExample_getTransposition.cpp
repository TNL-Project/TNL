#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
getTranspositionExample()
{
   // clang-format off
   TNL::Matrices::DenseMatrix< double, Device > matrix{
   { 1, 2, 3 },
   { 4, 5, 6 },
   { 7, 8, 9 },
   { 10, 11, 12 },
   { 13, 14, 15 } };
   // clang-format on

   std::cout << "Dense matrix:\n" << matrix << '\n';

   TNL::Matrices::DenseMatrix< double, Device > outputMatrix;

   outputMatrix.getTransposition( matrix );

   std::cout << "Transposed dense matrix:\n" << outputMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ...\n";
   getTranspositionExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA GPU ...\n";
   getTranspositionExample< TNL::Devices::Cuda >();
#endif
}
