#include <iostream>
#include <list>
#include <vector>
#include <TNL/Containers/Array.h>

using namespace TNL;
using namespace std;

/***
 * The following works for any device (CPU, GPU ...).
 */
template< typename Device >
void
arrayExample()
{
   const int size = 10;
   using ArrayType = Containers::Array< int, Device >;
   ArrayType a1( size );
   ArrayType a2( size );

   /***
    * You may initiate the array using setElement
    */
   for( int i = 0; i < size; i++ )
      a1.setElement( i, i );
   std::cout << "a1 = " << a1 << '\n';

   /***
    * You may also assign value to all array elements ...
    */
   a2 = 0;
   std::cout << "a2 = " << a2 << '\n';

   /***
    * ... or assign STL list and vector.
    */
   std::list< float > l = { 1.0, 2.0, 3.0 };
   std::vector< float > v = { 5.0, 6.0, 7.0 };
   a1 = l;
   std::cout << "a1 = " << a1 << '\n';
   a1 = v;
   std::cout << "a1 = " << a1 << '\n';

   /***
    * You may swap array data with the swap method.
    */
   a1.swap( a2 );

   /***
    * You may save it to file and load again
    */
   File( "a1.tnl", std::ios_base::out ) << a1;
   File( "a1.tnl", std::ios_base::in ) >> a2;

   std::remove( "a1.tnl" );

   if( a2 != a1 )
      std::cerr << "Something is wrong!!!\n";

   std::cout << "a2 = " << a2 << '\n';
}

int
main()
{
   std::cout << "The first test runs on CPU ...\n";
   arrayExample< Devices::Host >();
#ifdef __CUDACC__
   std::cout << "The second test runs on GPU ...\n";
   arrayExample< Devices::Cuda >();
#endif
}
