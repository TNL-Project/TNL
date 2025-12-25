#include <iostream>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/ArrayView.h>

using namespace TNL;
using namespace TNL::Containers;

int
main( int argc, char* argv[] )
{
   /****
    * Create new array
    */
   const int size = 5;
   Array< float > a( size );

   /****
    * Bind an array view with it
    */
   ArrayView< float > a_view = a.getView();
   auto another_view = a.getView();
   auto const_view = a.getConstView();

   another_view = -5;
   std::cout << " a = " << a << '\n';
   std::cout << " a_view = " << a_view << '\n';
   std::cout << " another_view = " << another_view << '\n';
   std::cout << " const_view = " << const_view << '\n';

   //const_view = 3; this would not compile, because constant views provide read-only access
}
