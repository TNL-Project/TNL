#include <iostream>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Devices/Host.h>

template< typename Device >
void
getSerializationTypeExample()
{
   TNL::Matrices::SparseMatrix< double, Device > matrix;

   std::cout << "Matrix type is: " << matrix.getSerializationType();
}

int
main( int argc, char* argv[] )
{
   std::cout << "Get serialization type on CPU ...\n";
   getSerializationTypeExample< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   std::cout << "Get serialization type on GPU (" << TNL::Backend::getBackendName() << ") ...\n";
   getSerializationTypeExample< TNL::Devices::GPU >();
#endif
}
