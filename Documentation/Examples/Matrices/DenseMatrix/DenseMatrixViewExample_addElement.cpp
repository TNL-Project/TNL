#include <iostream>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
addElements()
{
   TNL::Matrices::DenseMatrix< double, Device > matrix( 5, 5 );
   auto matrixView = matrix.getView();

   for( int i = 0; i < 5; i++ )
      matrixView.setElement( i, i, i );  // or matrix.setElement

   std::cout << "Initial matrix is:\n" << matrix << '\n';

   for( int i = 0; i < 5; i++ )
      for( int j = 0; j < 5; j++ )
         matrixView.addElement( i, j, 1.0, 5.0 );  // or matrix.addElement

   std::cout << "Matrix after addition is:\n" << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Add elements on host:\n";
   addElements< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Add elements on CUDA device:\n";
   addElements< TNL::Devices::Cuda >();
#endif
}
