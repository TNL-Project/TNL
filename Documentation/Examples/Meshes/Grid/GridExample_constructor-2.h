#include <iostream>
#include <TNL/Meshes/Grid.h>

template< int Dimension >
void
createGrid()
{
   using GridType = TNL::Meshes::Grid< Dimension >;
   using CoordinatesType = typename GridType::CoordinatesType;
   GridType grid( 10 );
   CoordinatesType origin( 0.0 );
   CoordinatesType proportions( 1.0 );
   grid.setDomain( origin, proportions );

   std::cout << grid << '\n';
}

int
main( int argc, char* argv[] )
{
   std::cout << "Creating grid with dimension equal one.\n";
   createGrid< 1 >();

   std::cout << "Creating grid with dimension equal two.\n";
   createGrid< 2 >();

   std::cout << "Creating grid with dimension equal three.\n";
   createGrid< 3 >();

   std::cout << "Creating grid with dimension equal four.\n";
   createGrid< 4 >();

   return EXIT_SUCCESS;
}
