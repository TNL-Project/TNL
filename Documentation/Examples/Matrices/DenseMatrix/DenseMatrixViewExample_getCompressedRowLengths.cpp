#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
getCompressedRowLengthsExample()
{
   TNL::Matrices::DenseMatrix< double, Device > denseMatrix{
      // clang-format off
      {  1 },
      {  2,  3 },
      {  4,  5,  6 },
      {  7,  8,  9, 10 },
      { 11, 12, 13, 14, 15 },
      // clang-format on
   };
   auto denseMatrixView = denseMatrix.getConstView();

   std::cout << denseMatrixView << '\n';

   TNL::Containers::Vector< int, Device > rowLengths;
   denseMatrixView.getCompressedRowLengths( rowLengths );

   std::cout << "Compressed row lengths are: " << rowLengths << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Getting compressed row lengths on host:\n";
   getCompressedRowLengthsExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Getting compressed row lengths on CUDA device:\n";
   getCompressedRowLengthsExample< TNL::Devices::Cuda >();
#endif
}
