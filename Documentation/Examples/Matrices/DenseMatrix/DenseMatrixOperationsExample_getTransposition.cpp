#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
getTranspositionExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix{ { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 }, { 10, 11, 12 }, { 13, 14, 15 } };

   std::cout << "Dense matrix: " << std::endl << matrix << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > outputMatrix;

   outputMatrix.getTransposition( matrix );

   std::cout << "Transposed dense matrix: " << std::endl << outputMatrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ... " << std::endl;
   getTranspositionExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA GPU ... " << std::endl;
   getTranspositionExample< TNL::Devices::Cuda >();
#endif
}
