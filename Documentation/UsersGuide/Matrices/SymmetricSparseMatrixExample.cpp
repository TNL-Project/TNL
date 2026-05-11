#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
symmetricSparseMatrixExample()
{
   TNL::Matrices::SparseMatrix< double, Device, int, TNL::Matrices::SymmetricMatrix > symmetricMatrix(
      5,  // number of matrix rows
      5,  // number of matrix columns
      {
         // matrix elements definition
         // clang-format off
         { 0, 0, 1.0 },
         { 1, 0, 2.0 }, { 1, 1, 1.0 },
         { 2, 0, 3.0 }, { 2, 2, 1.0 },
         { 3, 0, 4.0 }, { 3, 3, 1.0 },
         { 4, 0, 5.0 }, { 4, 4, 1.0 },
         // clang-format on
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricLower );

   std::cout << "Symmetric sparse matrix:\n" << symmetricMatrix << '\n';

   TNL::Containers::Vector< double, Device > inVector( 5, 1.0 );
   TNL::Containers::Vector< double, Device > outVector( 5, 0.0 );
   symmetricMatrix.vectorProduct( inVector, outVector );
   std::cout << "Product with vector " << inVector << " is " << outVector << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ...\n";
   symmetricSparseMatrixExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA GPU ...\n";
   symmetricSparseMatrixExample< TNL::Devices::Cuda >();
#endif
}
