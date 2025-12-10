#include <iostream>
#include <iomanip>
#include <functional>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
reduceRows()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix(
      // number of matrix rows
      5,
      // number of matrix columns
      5,
      // matrix elements definition
      {
         // clang-format off
         { 0, 0, 1 },
         { 1, 1, 1 }, { 1, 2, 8 },
         { 2, 2, 1 }, { 2, 3, 9 },
         { 3, 3, 1 }, { 3, 4, 9 },
         { 4, 4, 1 },
         // clang-format on
      } );

   /***
    * Find largest element in each row.
    */
   TNL::Containers::Vector< double, Device > rowMax( matrix.getRows() );

   /***
    * Prepare vector view for lambdas.
    */
   auto rowMaxView = rowMax.getView();

   /***
    * Fetch lambda just returns absolute value of matrix elements.
    */
   auto fetch = [] __cuda_callable__( int rowIdx, int columnIdx, const double& value ) -> double
   {
      return TNL::abs( value );
   };

   /***
    * Keep lambda store the largest value in each row to the vector rowMax.
    */
   auto keep = [ = ] __cuda_callable__( int rowIdx, const double& value ) mutable
   {
      rowMaxView[ rowIdx ] = value;
   };

   /***
    * Compute the largest values in each row.
    */
   matrix.reduceRows( 0, matrix.getRows(), fetch, TNL::Max{}, keep );

   std::cout << "The matrix reads as:\n" << matrix << '\n';
   std::cout << "Max. elements in rows are: " << rowMax << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Rows reduction on host:\n";
   reduceRows< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Rows reduction on CUDA device:\n";
   reduceRows< TNL::Devices::Cuda >();
#endif
}
