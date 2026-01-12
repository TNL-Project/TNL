#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
sparseInitializerListExample()
{
   /***
    * Create a 5x5 dense matrix using sparse tuple format {row, column, value}.
    * This is useful when you want to specify only non-zero elements.
    * All other elements will be initialized to zero.
    */
   // clang-format off
   TNL::Matrices::DenseMatrix< double, Device > sparseMatrix(
      5,
      5,  // 5x5 matrix
      {  // Sparse data: {row, column, value}
         { 0, 0, 1.0 },                { 0, 2, 2.0 },
                        { 1, 1, 3.0 },               { 1, 3, 4.0 },
         { 2, 0, 5.0 },                { 2, 2, 6.0 },               { 2, 4, 7.0 },
                        { 3, 1, 8.0 },               { 3, 3, 9.0 },
         { 4, 0, 10.0 },                                            { 4, 4, 11.0 } } );
   // clang-format on

   std::cout << "Dense matrix created from sparse tuple data:" << std::endl << sparseMatrix << std::endl;

   /***
    * You can also create symmetric matrices by providing only the lower
    * (or upper) triangular part with SymmetricLower encoding.
    */
   // clang-format off
   TNL::Matrices::DenseMatrix< double, Device > symmetricMatrix(
      4,
      4,
      {  // Only lower triangular elements
         { 0, 0, 1.0 },
         { 1, 0, 2.0 }, { 1, 1, 3.0 },
         { 2, 0, 4.0 }, { 2, 1, 5.0 }, { 2, 2, 6.0 },
         { 3, 0, 7.0 }, { 3, 1, 8.0 }, { 3, 2, 9.0 }, { 3, 3, 10.0 } },
      TNL::Matrices::MatrixElementsEncoding::SymmetricLower );
   // clang-format on
   std::cout << "Symmetric matrix from lower triangular data:" << std::endl << symmetricMatrix << std::endl;
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ... " << std::endl;
   sparseInitializerListExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrices on CUDA GPU ... " << std::endl;
   sparseInitializerListExample< TNL::Devices::Cuda >();
#endif
}
