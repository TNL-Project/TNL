#include <iostream>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

template< typename Device >
void setElements()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix( { 1, 1, 1, 1, 1 }, 5 );

   /****
    * Get the matrix view.
    */
   auto view = matrix.getView();
   for( int i = 0; i < 5; i++ )
      view.setElement( i, i, i );

   std::cout << "Matrix set from the host:" << std::endl;
   std::cout << matrix << std::endl;

   auto f = [=] __cuda_callable__ ( int i ) mutable {
      view.setElement( i, i, -i );
   };

   TNL::Algorithms::parallelFor< Device >( 0, 5, f );

   std::cout << "Matrix set from its native device:" << std::endl;
   std::cout << matrix << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Set elements on host:" << std::endl;
   setElements< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Set elements on CUDA device:" << std::endl;
   setElements< TNL::Devices::Cuda >();
#endif
}
