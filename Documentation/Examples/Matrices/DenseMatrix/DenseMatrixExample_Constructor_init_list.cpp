#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
initializerListExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix{
      // clang-format off
      {  1,  2,  3,  4,  5,  6 },
      {  7,  8,  9, 10, 11, 12 },
      { 13, 14, 15, 16, 17, 18 },
      // clang-format on
   };

   std::cout << "General dense matrix:\n" << matrix << '\n';

   TNL::Matrices::DenseMatrix< double, Device > triangularMatrix{
      // clang-format off
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 },
      // clang-format on
   };

   std::cout << "Triangular dense matrix:\n" << triangularMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ...\n";
   initializerListExample< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Creating matrices on GPU (" << TNL::Backend::getBackendName() << ") ...\n";
   initializerListExample< TNL::Devices::GPU >();
#endif
}
