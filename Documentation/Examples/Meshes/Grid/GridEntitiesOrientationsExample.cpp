#include <iostream>
#include <iomanip>
#include <TNL/Meshes/GridEntitiesOrientations.h>

template< int GridDimension >
void writeGridEntitiesOrientations()
{
   using EntitiesOrientations = TNL::Meshes::GridEntitiesOrientations< GridDimension >;
   using NormalsType = typename EntitiesOrientations::NormalsType;

   std::cout << "Grid dimension      Entity dimension    Entity orientation index      Total orientation index      Normals    " << std::endl;
   std::cout << "------------------------------------------------------------------------------------------------------------------" << std::endl;
   EntitiesOrientations entitiesOrientations;
   for( int entityDimension = 0; entityDimension <= GridDimension; entityDimension++ )
      for( int orientationIndex = 0; orientationIndex < EntitiesOrientations::getOrientationsCount( entityDimension ); orientationIndex++ )
      {
         int totalOrientationIndex = EntitiesOrientations::getTotalOrientationIndex( entityDimension, orientationIndex );

         std::cout << std::setw( 10 ) << GridDimension
                   << std::setw( 20 ) << entityDimension
                   << std::setw( 25 ) << orientationIndex
                   << std::setw( 25 ) << totalOrientationIndex
                   << std::setw( 20 ) << entitiesOrientations.getNormals( totalOrientationIndex ) << std::endl;
      }
   std::cout << "------------------------------------------------------------------------------------------------------------------" << std::endl;
}

int main( int argc, char* argv[] )
{
   writeGridEntitiesOrientations< 1 >();
   writeGridEntitiesOrientations< 2 >();
   writeGridEntitiesOrientations< 3 >();
   writeGridEntitiesOrientations< 4 >();
}
