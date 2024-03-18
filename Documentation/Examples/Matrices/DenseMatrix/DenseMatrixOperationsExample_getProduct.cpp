#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
getProductExample1()
{
   std::cout << "Normal x Normal" << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   std::cout << "Dense matrix 1: " << std::endl << matrix1 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix2{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   std::cout << "Dense matrix 2: " << std::endl << matrix2 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > resultMatrix;

   resultMatrix.getMatrixProduct( matrix1, matrix2 );

   std::cout << "Product: " << std::endl << resultMatrix << std::endl;
}

template< typename Device >
void
getProductExample2()
{
   std::cout << "Transposed x Normal" << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix1{ { 5, 6, 6 }, { 6, 5, 7 }, { 7, 2, 1 }, { 3, 3, 9 } };

   std::cout << "Dense matrix 1: " << std::endl << matrix1 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix2{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   std::cout << "Dense matrix 2: " << std::endl << matrix2 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > resultMatrix;

   resultMatrix.getMatrixProduct(
      matrix1, matrix2, 1.0, TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::None );

   std::cout << "Product: " << std::endl << resultMatrix << std::endl;
}

template< typename Device >
void
getProductExample3()
{
   std::cout << "Normal x Transposed" << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   std::cout << "Dense matrix 1: " << std::endl << matrix1 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix2{
      { 4, 3, 2, 6, 7 }, { 3, 4, 2, 8, 6 }, { 1, 4, 7, 9, 5 }, { 2, 3, 8, 0, 4 }
   };

   std::cout << "Dense matrix 2: " << std::endl << matrix2 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > resultMatrix;

   resultMatrix.getMatrixProduct(
      matrix1, matrix2, 1.0, TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::Transpose );

   std::cout << "Product: " << std::endl << resultMatrix << std::endl;
}

template< typename Device >
void
getProductExample4()
{
   std::cout << "Transposed x Transposed" << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix1{ { 5, 6, 6 }, { 6, 5, 7 }, { 7, 2, 1 }, { 3, 3, 9 } };

   std::cout << "Dense matrix 1: " << std::endl << matrix1 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > matrix2{
      { 4, 3, 2, 6, 7 }, { 3, 4, 2, 8, 6 }, { 1, 4, 7, 9, 5 }, { 2, 3, 8, 0, 4 }
   };

   std::cout << "Dense matrix 2: " << std::endl << matrix2 << std::endl;

   TNL::Matrices::DenseMatrix< double, Device > resultMatrix;

   resultMatrix.getMatrixProduct(
      matrix1, matrix2, 1.0, TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::Transpose );

   std::cout << "Product: " << std::endl << resultMatrix << std::endl;
}
int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrix on CPU ... " << std::endl;
   getProductExample1< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrix on CUDA GPU ... " << std::endl;
   getProductExample1< TNL::Devices::Cuda >();
   getProductExample2< TNL::Devices::Cuda >();
   getProductExample3< TNL::Devices::Cuda >();
   getProductExample4< TNL::Devices::Cuda >();
#endif
}
