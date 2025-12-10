#include <iostream>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>

using namespace TNL;
using namespace TNL::Containers;

template< typename Device >
void
expressions()
{
   using RealType = float;
   using VectorType = Vector< RealType, Device >;

   /****
    * Create vectors
    */
   const int size = 11;
   VectorType a( size );
   VectorType b( size );
   VectorType c( size );
   a.forAllElements(
      [] __cuda_callable__( int i, RealType& value )
      {
         value = 3.14 * ( i - 5.0 ) / 5.0;
      } );
   b = a * a;
   c = 3 * a + sign( a ) * sin( a );
   std::cout << "a = " << a << '\n';
   std::cout << "sin( a ) = " << sin( a ) << '\n';
   std::cout << "abs( sin( a ) ) = " << abs( sin( a ) ) << '\n';
   std::cout << "b = " << b << '\n';
   std::cout << "c = " << c << '\n';
}

int
main( int argc, char* argv[] )
{
   /****
    * Perform test on CPU
    */
   std::cout << "Expressions on CPU ...\n";
   expressions< Devices::Host >();

   /****
    * Perform test on GPU
    */
#ifdef __CUDACC__
   std::cout << '\n';
   std::cout << "Expressions on GPU ...\n";
   expressions< Devices::Cuda >();
#endif
}
