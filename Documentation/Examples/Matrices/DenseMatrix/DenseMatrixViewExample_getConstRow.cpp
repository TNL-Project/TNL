#include <iostream>
#include <functional>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/GPU.h>

template< typename Device >
void
getRowExample()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix{
      // clang-format off
      { 1, 0, 0, 0, 0 },
      { 1, 2, 0, 0, 0 },
      { 1, 2, 3, 0, 0 },
      { 1, 2, 3, 4, 0 },
      { 1, 2, 3, 4, 5 },
      // clang-format on
   };

   /***
    * We need a matrix view to pass the matrix to lambda function even on GPU.
    */
   const auto matrixView = matrix.getConstView();

   /***
    * Fetch lambda function returns diagonal element in each row.
    */
   auto fetch = [ = ] __cuda_callable__( int rowIdx ) -> double
   {
      auto row = matrixView.getRow( rowIdx );
      return row.getValue( rowIdx );
   };

   int trace = TNL::Algorithms::reduce< Device >( 0, matrix.getRows(), fetch, std::plus<>{}, 0 );
   std::cout << "Matrix trace is " << trace << ".\n";
}

int
main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host:\n";
   getRowExample< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Getting matrix rows on GPU (" << TNL::Backend::getBackendName() << "):\n";
   getRowExample< TNL::Devices::GPU >();
#endif
}
