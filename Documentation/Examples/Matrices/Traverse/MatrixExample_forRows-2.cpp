#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void
forRowsExample2()
{
   /***
    * Create a sparse matrix and set up a tridiagonal structure.
    */
   const int size = 5;
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 1, 3, 3, 3, 1 }, size );

   using RowView = typename TNL::Matrices::SparseMatrix< double, Device >::RowView;

   auto setupRow = [] __cuda_callable__( RowView & row )
   {
      const int rowIdx = row.getRowIndex();
      const int size = 5;

      if( rowIdx == 0 )
         row.setElement( 0, rowIdx, 2.0 );
      else if( rowIdx == size - 1 )
         row.setElement( 0, rowIdx, 2.0 );
      else {
         row.setElement( 0, rowIdx - 1, 1.0 );
         row.setElement( 1, rowIdx, 2.0 );
         row.setElement( 2, rowIdx + 1, 1.0 );
      }
   };

   TNL::Matrices::forRows( matrix, 0, size, setupRow );
   std::cout << "Initial tridiagonal matrix:\n";
   std::cout << matrix << '\n';

   /***
    * Normalize each row by dividing by the sum of its elements.
    */
   auto normalizeRow = [] __cuda_callable__( RowView & row )
   {
      double sum = 0.0;
      for( auto element : row )
         sum += element.value();

      for( auto element : row )
         element.value() /= sum;
   };

   TNL::Matrices::forRows( matrix, 0, size, normalizeRow );
   std::cout << "Row-normalized matrix:\n";
   std::cout << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Running on host:\n";
   forRowsExample2< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running on CUDA device:\n";
   forRowsExample2< TNL::Devices::Cuda >();
#endif
}
