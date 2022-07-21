#include <iostream>
#include <TNL/Meshes/Grid.h>

int main( int argc, char* argv[] )
{
   TNL::Meshes::Grid< 4 > grid( 10, 10, 10, 10 );
   std::cout << grid << std::endl;
   return EXIT_SUCCESS;
}
