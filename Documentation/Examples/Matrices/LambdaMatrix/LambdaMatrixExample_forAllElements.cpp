#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/LambdaMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

template< typename Device >
void
forAllElementsExample()
{
   /***
    * Lambda functions defining the matrix.
    */
   auto compressedRowLengths = [ = ] __cuda_callable__( const int rows, const int columns, const int rowIdx ) -> int
   {
      return columns;
   };
   auto matrixElements =
      [ = ] __cuda_callable__(
         const int rows, const int columns, const int rowIdx, const int localIdx, int& columnIdx, double& value )
   {
      columnIdx = localIdx;
      value = TNL::max( rowIdx - columnIdx + 1, 0 );
   };

   using MatrixFactory = TNL::Matrices::LambdaMatrixFactory< double, Device, int >;
   auto matrix = MatrixFactory::create( 5, 5, matrixElements, compressedRowLengths );

   TNL::Matrices::DenseMatrix< double, Device > denseMatrix( 5, 5 );
   auto denseView = denseMatrix.getView();

   auto f = [ = ] __cuda_callable__( int rowIdx, int localIdx, int columnIdx, double value ) mutable
   {
      denseView.setElement( rowIdx, columnIdx, value );
   };

   TNL::Matrices::forAllElements( matrix, f );
   std::cout << "Original lambda matrix:\n" << matrix << '\n';
   std::cout << "Dense matrix:\n" << denseMatrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Copying matrix on host:\n";
   forAllElementsExample< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Copying matrix on GPU (" << TNL::Backend::getBackendName() << "):\n";
   forAllElementsExample< TNL::Devices::GPU >();
#endif
}
