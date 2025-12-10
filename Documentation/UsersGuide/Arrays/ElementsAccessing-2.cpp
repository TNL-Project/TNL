#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

int
main( int argc, char* argv[] )
{
   /****
    * Create new arrays on both host and device
    */
   const int size = 5;
   Array< float, Devices::Host > host_array( size );
   Array< float, Devices::Cuda > device_array( size );

   /****
    * Initiate the arrays with setElement
    */
   for( int i = 0; i < size; i++ ) {
      host_array.setElement( i, i );
      device_array.setElement( i, i );
   }

   /****
    * Compare the arrays using getElement
    */
   for( int i = 0; i < size; i++ )
      if( host_array.getElement( i ) == device_array.getElement( i ) )
         std::cout << "Elements at position " << i << " match.\n";

   /****
    * Print the results
    */
   std::cout << '\n';
   std::cout << "host_array = " << host_array << '\n';
   std::cout << "device_array = " << device_array << '\n';
}
