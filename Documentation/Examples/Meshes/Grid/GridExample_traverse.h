#include <iostream>
#include <iomanip>
#include <TNL/Meshes/Grid.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void traverseGrid()
{
   /***
    * Define grid dimension and size.
    */
   static constexpr int Dimension = 2;
   const int grid_size = 5;

   /***
    * Setup necessary type.
    */
   using GridType = TNL::Meshes::Grid< Dimension, double, Device >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using PointType = typename GridType::PointType;
   using VectorType = TNL::Containers::Vector< double, Device >;

   /***
    * Setup types of grid entities.
    */
   using GridCell = typename GridType::Cell;
   using GridFace = typename GridType::Face;
   using GridVertex = typename GridType::Vertex;

   /***
    * Create an instance of a grid.
    */
   GridType grid( grid_size );
   PointType origin( 0.0 ), proportions( 1.0 );
   grid.setDomain( origin, proportions );

   /***
    * Allocate vectors for values stored in particular grid entities.
    */
   VectorType cells( grid.template getEntitiesCount< Dimension >(), 0.0 );
   VectorType faces( grid.template getEntitiesCount< Dimension - 1 >(), 0.0 );
   VectorType vertexes( grid.template getEntitiesCount< 0 >(), 0.0 );

   /***
    * Prepare views for the data at the grid entities so that we can
    * manipulate them in lambda functions runnig eventually on GPU.
    */
   auto cells_view = cells.getView();
   auto faces_view = faces.getView();
   auto vertexes_view = vertexes.getView();

   /***
    * Setup value of each cell to its index in the grid.
    */
   grid.template forAllEntities< Dimension >( [=] __cuda_callable__ ( const GridCell& cell ) mutable {
      cells_view[ cell.getIndex() ] = cell.getIndex();
   } );

   /***
    * Print values of all cells in the grid.
    */
   std::cout << "Values of cells .... " << std::endl;
   for( int i = grid_size-1; i>= 0; i-- ) {
      for( int j = 0; j < grid_size; j++ ) {
         GridCell cell( grid, {j, i} );
         auto idx = cell.getIndex();
         std::cout << std::right << std::setw( 12 ) << cells.getElement( idx );
      }
      std::cout << std::endl;
   }
   std::cout << std::endl;

   /***
    * Setup values of all faces to an average value of its neighbour cells.
    */
   grid.template forAllEntities< Dimension - 1 >( [=] __cuda_callable__ ( const GridFace& face ) mutable {
      const CoordinatesType normal =  face.getNormals();
      double sum = 0.0;
      double count = 0.0;
      if( TNL::all(greaterEqual( face.getCoordinates() - normal, 0 )) ) {
         auto neighbour = face.template getNeighbourEntity< Dimension >( -normal );
         sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      if( TNL::all(less( face.getCoordinates(), face.getGrid().getDimensions() )) ) {
         auto neighbour = face.template getNeighbourEntity< Dimension >( { 0, 0 } );
         sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      faces_view[ face.getIndex() ] = sum / count;
   } );

   /***
    * Print values of all faces in the grid.
    */
   std::cout << "Values of faces ..." << std::endl;
   for( int i = grid_size; i>= 0; i-- ) {
      std::cout << std::right << std::setw( 6 ) << " ";
      for( int j = 0; j < grid_size; j++ ) {
         GridFace face( grid, {j, i}, {0,1} );
         auto idx = face.getIndex();
         std::cout << std::right << std::setw( 12 ) << faces.getElement( idx );
      }
      std::cout << std::endl;
      if( i > 0 )
      for( int j = 0; j <= grid_size; j++ ) {
         GridFace face( grid, {j, i - 1}, { 1,0 } );
         auto idx = face.getIndex();
         std::cout << std::right << std::setw( 12 ) << faces.getElement( idx );
      }
      std::cout << std::endl;
   }

   /***
    * Setup values of all vertexes to an average value of its neighbouring cells.
    */
   grid.template forAllEntities< 0 >( [=] __cuda_callable__ ( const GridVertex& vertex ) mutable {
      double sum = 0.0;
      double count = 0.0;
      auto grid_dimensions = vertex.getGrid().getDimensions();
      if( vertex.getCoordinates().x() > 0 && vertex.getCoordinates().y() > 0 ) {
         auto neighbour = vertex.template getNeighbourEntity< Dimension >( { -1,-1 } );
         sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      if( vertex.getCoordinates().x() > 0 && vertex.getCoordinates().y() < grid_dimensions.y() ) {
         auto neighbour = vertex.template getNeighbourEntity< Dimension >( { -1,0 } );
         sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      if( vertex.getCoordinates().x() < grid_dimensions.x() && vertex.getCoordinates().y() > 0 ) {
         auto neighbour = vertex.template getNeighbourEntity< Dimension >( { 0,-1 } );
         sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      if( TNL::all(less( vertex.getCoordinates(), vertex.getGrid().getDimensions() )) ) {
         auto neighbour = vertex.template getNeighbourEntity< Dimension >( {0,0} );
         sum += cells_view[ neighbour.getIndex() ];
         count++;
      }
      vertexes_view[ vertex.getIndex() ] = sum / count;
   } );

   /***
    * Print values of all vertexes in the grid.
    */
   std::cout << "Values of vertexes .... " << std::endl;
   for( int i = grid_size; i>= 0; i-- ) {
      for( int j = 0; j <= grid_size; j++ ) {
         GridVertex vertex( grid, {j, i} );
         auto idx = vertex.getIndex();
         std::cout << std::right << std::setw( 12 ) << vertexes.getElement( idx );
      }
      std::cout << std::endl;
   }
}

int main( int argc, char* argv[] )
{
   std::cout << "Traversing grid on CPU..." << std::endl;
   traverseGrid< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Traversing grid on CUDA GPU..." << std::endl;
   traverseGrid< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}
