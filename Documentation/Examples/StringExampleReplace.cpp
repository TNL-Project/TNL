#include <iostream>
#include <TNL/String.h>

using namespace TNL;

int
main()
{
   String phrase( "Say yes yes yes!" );
   std::cout << "phrase.replace( \"yes\", \"no\", 1 ) = " << phrase.replace( "yes", "no", 1 ) << '\n';
   std::cout << "phrase.replace( \"yes\", \"no\", 2 ) = " << phrase.replace( "yes", "no", 2 ) << '\n';
   std::cout << "phrase.replace( \"yes\", \"no\", 3 ) = " << phrase.replace( "yes", "no", 3 ) << '\n';
}
