#include <iostream>
#include <TNL/String.h>

using namespace TNL;

int
main()
{
   String string;
   string.setSize( 1024 );
   std::cout << "String size = " << string.getSize() << '\n';
   std::cout << "Allocated size = " << string.getAllocatedSize() << '\n';
}
