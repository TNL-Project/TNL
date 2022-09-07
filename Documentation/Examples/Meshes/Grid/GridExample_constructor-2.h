#include <iostream>
#include <TNL/Meshes/Grid.h>

template< int Dimension >
void createGrid()
{
   using GridType = TNL::Meshes::Grid< Dimension >;
   using CoordinatesType = typename GridType::CoordinatesType;
   GridType grid( 10 );
   CoordinatesType origin( 0.0 ), proportions( 1.0 );
   grid.setDomain( origin, proportions );

   std::cout << grid << std::endl;
}

int main( int argc, char* argv[] )
{
   std::cout << "Creating grid with dimension equal one." << std::endl;
   createGrid< 1 >();

   std::cout << "Creating grid with dimension equal two." << std::endl;
   createGrid< 2 >();

   std::cout << "Creating grid with dimension equal three." << std::endl;
   createGrid< 3 >();

   std::cout << "Creating grid with dimension equal four." << std::endl;
   createGrid< 4 >();

   return EXIT_SUCCESS;
}
