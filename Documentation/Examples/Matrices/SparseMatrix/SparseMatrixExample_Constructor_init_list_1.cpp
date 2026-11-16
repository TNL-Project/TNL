#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
initializerListExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix{
      { 1, 2, 3, 4, 5 },  // row capacities
      6,
   };  // number of matrix columns

   for( int row = 0; row < matrix.getRows(); row++ )
      for( int column = 0; column <= row; column++ )
         matrix.setElement( row, column, row - column + 1 );
   std::cout << "General sparse matrix:\n" << matrix << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating matrices on CPU ...\n";
   initializerListExample< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Creating matrices on GPU (" << TNL::Backend::getBackendName() << ") ...\n";
   initializerListExample< TNL::Devices::GPU >();
#endif
}
