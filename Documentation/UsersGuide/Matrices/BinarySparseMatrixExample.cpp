#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
binarySparseMatrixExample()
{
   TNL::Matrices::SparseMatrix< bool, Device, int > binaryMatrix(
      // number of matrix rows
      5,
      // number of matrix columns
      5,
      // matrix elements definition
      {
         // clang-format off
         { 0, 0, 1.0 }, { 0, 1, 2.0 }, { 0, 2, 3.0 }, { 0, 3, 4.0 }, { 0, 4, 5.0 },
         { 1, 0, 2.0 }, { 1, 1, 1.0 },
         { 2, 0, 3.0 }, { 2, 2, 1.0 },
         { 3, 0, 4.0 }, { 3, 3, 1.0 },
         { 4, 0, 5.0 }, { 4, 4, 1.0 },
         // clang-format on
      } );

   std::cout << "Binary sparse matrix: \n" << binaryMatrix << '\n';

   TNL::Containers::Vector< double, Device > inVector( 5, 1.1 ), outVector( 5, 0.0 );
   binaryMatrix.vectorProduct( inVector, outVector );
   std::cout << "Product with vector " << inVector << " is " << outVector << "\n\n";

   TNL::Matrices::SparseMatrix< bool, Device, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR, double >
      binaryMatrix2;
   binaryMatrix2 = binaryMatrix;
   binaryMatrix2.vectorProduct( inVector, outVector );
   std::cout << "Product with vector in double precision " << inVector << " is " << outVector << "\n\n";
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ... \n";
   binarySparseMatrixExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA GPU ... \n";
   binarySparseMatrixExample< TNL::Devices::Cuda >();
#endif
}
