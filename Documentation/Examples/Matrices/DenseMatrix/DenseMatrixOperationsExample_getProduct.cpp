#include "TNL/Algorithms/Segments/ElementsOrganization.h"
#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device, TNL::Matrices::ElementsOrganization Organization >
void
matrixProductExample( TNL::Matrices::TransposeState transposeA, TNL::Matrices::TransposeState transposeB )
{
   std::cout << "Matrix product (";
   std::cout << ( transposeA == TNL::Matrices::TransposeState::Transpose ? "Transposed x " : "Normal x " );
   std::cout << ( transposeB == TNL::Matrices::TransposeState::Transpose ? "Transposed)" : "Normal)" ) << '\n';

   TNL::Matrices::DenseMatrix< double, Device, int, Organization > matrix1;
   TNL::Matrices::DenseMatrix< double, Device, int, Organization > matrix2;

   if( transposeA == TNL::Matrices::TransposeState::Transpose ) {
      // clang-format off
      matrix1 = {
      { 5, 6, 6 },
      { 6, 5, 7 },
      { 7, 2, 1 },
      { 3, 3, 9 } };
      // clang-format on
   }
   else {
      // clang-format off
      matrix1 = {
      { 5, 6, 7, 3 },
      { 6, 5, 2, 3 },
      { 6, 7, 1, 9 } };
      // clang-format on
   }

   std::cout << "Dense matrix 1:\n" << matrix1 << '\n';

   if( transposeB == TNL::Matrices::TransposeState::Transpose ) {
      // clang-format off
      matrix2 = {
      { 4, 3, 2, 6 },
      { 3, 4, 2, 8 },
      { 1, 4, 7, 9 },
      { 2, 3, 8, 0 },
      { 7, 6, 5, 4 } };
      // clang-format on
   }
   else {
      // clang-format off
      matrix2 = {
      { 4, 3, 2, 6, 7 },
      { 3, 4, 2, 8, 6 },
      { 1, 4, 7, 9, 5 },
      { 2, 3, 8, 0, 4 } };
      // clang-format on
   }

   std::cout << "Dense matrix 2:\n" << matrix2 << '\n';

   TNL::Matrices::DenseMatrix< double, Device, int, Organization > resultMatrix;
   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0, transposeA, transposeB );

   std::cout << "Product:\n" << resultMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ...\n";
   std::cout << "Stored in Row Major Order ...\n";
   matrixProductExample< TNL::Devices::Host, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >(
      TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::None );
   std::cout << "Stored in Column Major Order\n";
   matrixProductExample< TNL::Devices::Host, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >(
      TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::None );

#ifdef __CUDACC__
   std::cout << "Creating matrices on CUDA GPU ...\n";
   std::cout << "Stored in Row Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >(
      TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::None );
   std::cout << "Stored in Column Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >(
      TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::None );

   std::cout << "Stored in Row Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >(
      TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::None );
   std::cout << "Stored in Column Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >(
      TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::None );

   std::cout << "Stored in Row Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >(
      TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::Transpose );
   std::cout << "Stored in Column Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >(
      TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::Transpose );

   std::cout << "Stored in Row Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >(
      TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::Transpose );
   std::cout << "Stored in Column Major Order ...\n";
   matrixProductExample< TNL::Devices::Cuda, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >(
      TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::Transpose );
#endif
}
