#include <iostream>
#include <TNL/Meshes/Grid.h>

int
main( int argc, char* argv[] )
{
   TNL::Meshes::Grid< 1 > grid_1D( 10 );
   grid_1D.setDomain( { 0.0 }, { 1.0 } );
   std::cout << grid_1D << '\n';

   TNL::Meshes::Grid< 2 > grid_2D( 10, 20 );
   grid_2D.setDomain( { 0.0, 0.0 }, { 1.0, 2.0 } );
   std::cout << grid_2D << '\n';

   TNL::Meshes::Grid< 3 > grid_3D( 10, 20, 30 );
   grid_3D.setDomain( { 0.0, 0.0, 0.0 }, { 1.0, 2.0, 3.0 } );
   std::cout << grid_3D << '\n';

   return EXIT_SUCCESS;
}
