#include <iostream>
#include <map>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
stdMapConstructorExample()
{
   /***
    * Create a 5x5 dense matrix from std::map.
    * The map uses pairs {row, column} as keys and matrix element values.
    * This is useful when building matrices dynamically or from external data.
    */
   std::map< std::pair< int, int >, double > matrixData;

   // Build the matrix data
   matrixData[ { 0, 0 } ] = 1.0;
   matrixData[ { 0, 2 } ] = 2.0;
   matrixData[ { 1, 1 } ] = 3.0;
   matrixData[ { 1, 3 } ] = 4.0;
   matrixData[ { 2, 0 } ] = 5.0;
   matrixData[ { 2, 2 } ] = 6.0;
   matrixData[ { 2, 4 } ] = 7.0;
   matrixData[ { 3, 1 } ] = 8.0;
   matrixData[ { 3, 3 } ] = 9.0;
   matrixData[ { 4, 0 } ] = 10.0;
   matrixData[ { 4, 4 } ] = 11.0;

   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5, matrixData );

   std::cout << "Dense matrix created from std::map:\n" << matrix << '\n';

   /***
    * You can also create symmetric matrices by providing only the upper
    * triangular part with SymmetricUpper encoding.
    */
   std::map< std::pair< int, int >, double > symmetricData;

   // Only upper triangular part
   symmetricData[ { 0, 0 } ] = 1.0;
   symmetricData[ { 0, 1 } ] = 2.0;
   symmetricData[ { 0, 2 } ] = 4.0;
   symmetricData[ { 0, 3 } ] = 7.0;
   symmetricData[ { 1, 1 } ] = 3.0;
   symmetricData[ { 1, 2 } ] = 5.0;
   symmetricData[ { 1, 3 } ] = 8.0;
   symmetricData[ { 2, 2 } ] = 6.0;
   symmetricData[ { 2, 3 } ] = 9.0;
   symmetricData[ { 3, 3 } ] = 10.0;

   TNL::Matrices::DenseMatrix< double, Device > symmetricMatrix(
      4, 4, symmetricData, TNL::Matrices::MatrixElementsEncoding::SymmetricUpper );

   std::cout << "Symmetric matrix from upper triangular std::map data:\n" << symmetricMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ...\n";
   stdMapConstructorExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Creating matrices on CUDA GPU ...\n";
   stdMapConstructorExample< TNL::Devices::Cuda >();
#endif
}
