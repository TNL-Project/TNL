#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
getRowExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix(
      //
      5,
      5,
      {
         // clang-format off
         { 0, 0, 1 },
         { 1, 0, 1 }, { 1, 1, 2 },
         { 2, 0, 1 }, { 2, 1, 2 }, { 2, 2, 3 },
         { 3, 0, 1 }, { 3, 1, 2 }, { 3, 2, 3 }, { 3, 3, 4 },
         { 4, 0, 1 }, { 4, 1, 2 }, { 4, 2, 3 }, { 4, 3, 4 }, { 4, 4, 5 },
         // clang-format on
      } );
   const auto matrixView = matrix.getView();

   /***
    * Fetch lambda function returns diagonal element in each row.
    */
   auto fetch = [ = ] __cuda_callable__( int rowIdx ) -> double
   {
      auto row = matrixView.getRow( rowIdx );
      return row.getValue( rowIdx );
   };

   /***
    * Compute the matrix trace.
    */
   int trace = TNL::Algorithms::reduce< Device >( 0, matrix.getRows(), fetch, std::plus<>{}, 0 );
   std::cout << "Matrix trace is " << trace << ".\n";
}

int
main( int argc, char* argv[] )
{
   std::cout << "Getting matrix rows on host:\n";
   getRowExample< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Getting matrix rows on CUDA device:\n";
   getRowExample< TNL::Devices::Cuda >();
#endif
}
