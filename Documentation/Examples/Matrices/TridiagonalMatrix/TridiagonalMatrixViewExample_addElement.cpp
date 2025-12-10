#include <iostream>
#include <TNL/Matrices/TridiagonalMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
addElements()
{
   const int matrixSize( 5 );
   TNL::Matrices::TridiagonalMatrix< double, Device > matrix( matrixSize,  // number of rows
                                                              matrixSize   // number of columns
   );
   auto view = matrix.getView();

   for( int i = 0; i < matrixSize; i++ )
      view.setElement( i, i, i );  // or matrix.setElement

   std::cout << "Initial matrix is:\n" << matrix << '\n';

   for( int i = 0; i < matrixSize; i++ ) {
      if( i > 0 )
         view.addElement( i, i - 1, 1.0, 5.0 );  // or matrix.addElement
      view.addElement( i, i, 1.0, 5.0 );         // or matrix.addElement
      if( i < matrixSize - 1 )
         view.addElement( i, i + 1, 1.0, 5.0 );  // or matrix.addElement
   }

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
