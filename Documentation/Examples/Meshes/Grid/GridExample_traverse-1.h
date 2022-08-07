#include <iostream>
#include <TNL/Meshes/Grid.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void traverseGrid()
{
   constexpr int Dimension = 2;
   const int grid_size = 5;
   using GridType = TNL::Meshes::Grid< Dimension, double, Device >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using VectorType = TNL::Containers::Vector< double, Device >;
   using GridCell = typename GridType::Cell;
   using GridFace = typename GridType::Face;
   GridType grid( grid_size );
   CoordinatesType origin( 0.0 ), proportions( 1.0 );
   grid.setDomain( origin, proportions );
   VectorType cells( grid.template getEntitiesCount< Dimension >(), 0.0 );
   VectorType faces( grid.template getEntitiesCount< Dimension - 1 >(), 0.0 );

   auto cells_view = cells.getView();
   auto faces_view = faces.getView();
   grid.template forAll< Dimension >( [=] __cuda_callable__ ( const GridCell& cell ) mutable {
      cells_view[ cell.getIndex() ] = cell.getIndex();
   } );

   for( int i = grid_size-1; i>= 0; i-- ) {
      for( int j = 0; j < grid_size; j++ ) {
         GridCell cell( grid, {j, i} );
         auto idx = cell.getIndex();
         std::cout << cells.getElement( idx ) << "\t ";
      }
      std::cout << std::endl;
   }

   grid.template forAll< Dimension - 1 >( [=] __cuda_callable__ ( const GridFace& face ) mutable {
      std::cout << "Face: " << face << std::endl;
      const CoordinatesType direction =  CoordinatesType( 1, 1 ) - face.getBasis();
      double sum = 0.0;
      double count = 0.0;
      if( face.getCoordinates() - direction > CoordinatesType( 0, 0 ) ) {
         auto neighbour = face.template getNeighbourEntity< Dimension >( -direction );
         //std::cout << "1111 " << neighbour.getCoordinates() << std::endl;
         //sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      std::cout << ">>> direction = " << direction << std::endl;
      if( ( face.getCoordinates() + direction ) < face.getGrid().getDimensions() ) {

         auto neighbour = face.template getNeighbourEntity< Dimension >( direction );
         //std::cout << "2222 " << neighbour.getCoordinates() << std::endl;
         //sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      double average = sum / count;
      //std::cout << neighbour.getCoordinates() << "  " << neighbour_2.getCoordinates() << std::endl;
   } );
}

int main( int argc, char* argv[] )
{
    traverseGrid< TNL::Devices::Host >();
    return EXIT_SUCCESS;
}
