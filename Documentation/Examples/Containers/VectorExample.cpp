#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>

using namespace TNL;

template< typename Device >
void
VectorExample()
{
   Containers::Vector< int, Device > vector1( 5 );
   vector1 = 0;

   Containers::Vector< int, Device > vector2( 3 );
   vector2 = 1;
   vector2.swap( vector1 );
   vector2.setElement( 2, 4 );

   std::cout << "First vector:" << vector1.getData() << '\n';
   std::cout << "Second vector:" << vector2.getData() << '\n';

   vector2.reset();
   std::cout << "Second vector after reset:" << vector2.getData() << '\n';

   Containers::Vector< int, Device > vect = { 1, 2, -3, 3 };
   std::cout << "The smallest element is:" << min( vect ) << '\n';
   std::cout << "The absolute biggest element is:" << max( abs( vect ) ) << '\n';
   std::cout << "Sum of all vector elements:" << sum( vect ) << '\n';
   vect *= 2.0;
   std::cout << "Vector multiplied by 2:" << vect << '\n';
}

int
main()
{
   std::cout << "Running vector example on the host system:\n";
   VectorExample< Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Running vector example on the CUDA device:\n";
   VectorExample< Devices::Cuda >();
#endif
}
