#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
getElementsCountExample()
{
   TNL::Matrices::DenseMatrix< double, Device > triangularMatrix{
      // clang-format off
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 },
      // clang-format on
   };
   auto triangularMatrixView = triangularMatrix.getConstView();

   std::cout << "Matrix elements count is " << triangularMatrixView.getAllocatedElementsCount() << ".\n";
   std::cout << "Non-zero matrix elements count is " << triangularMatrixView.getNonzeroElementsCount() << ".\n";
}

int
main( int argc, char* argv[] )
{
   std::cout << "Computing matrix elements on host:\n";
   getElementsCountExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Computing matrix elements on CUDA device:\n";
   getElementsCountExample< TNL::Devices::Cuda >();
#endif
}
