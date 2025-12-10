#include <iostream>
#include <iomanip>
#include <TNL/Meshes/Grid.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void
traverseGrid()
{
   //! [setup]
   // Define grid dimension and size.
   static constexpr int Dimension = 2;
   const int grid_size = 5;

   // Setup necessary types.
   using GridType = TNL::Meshes::Grid< Dimension, double, Device >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using PointType = typename GridType::PointType;
   using VectorType = TNL::Containers::Vector< double, Device >;

   // Setup types of grid entities.
   using GridCell = typename GridType::Cell;
   using GridFace = typename GridType::Face;
   using GridVertex = typename GridType::Vertex;
   //! [setup]

   //! [create grid]
   // Create an instance of a grid.
   GridType grid( grid_size );
   PointType origin( 0.0 );
   PointType proportions( 1.0 );
   grid.setDomain( origin, proportions );
   //! [create grid]

   //! [allocate vectors]
   // Allocate vectors for values stored in particular grid entities.
   VectorType cells( grid.template getEntitiesCount< Dimension >(), 0.0 );
   VectorType faces( grid.template getEntitiesCount< Dimension - 1 >(), 0.0 );
   VectorType vertices( grid.template getEntitiesCount< 0 >(), 0.0 );
   //! [allocate vectors]

   //! [prepare vector views]
   // Prepare views for the data at the grid entities so that we can
   // manipulate them in lambda functions running eventually on GPU.
   auto cells_view = cells.getView();
   auto faces_view = faces.getView();
   auto vertices_view = vertices.getView();
   //! [prepare vector views]

   //! [initialize cells]
   // Setup value of each cell to its index in the grid.
   grid.template forAllEntities< Dimension >(
      [ = ] __cuda_callable__( const GridCell& cell ) mutable
      {
         cells_view[ cell.getIndex() ] = cell.getIndex();
      } );
   //! [initialize cells]

   //! [print cells]
   // Print values of all cells in the grid.
   std::cout << "Values of cells ....\n";
   for( int i = grid_size - 1; i >= 0; i-- ) {
      for( int j = 0; j < grid_size; j++ ) {
         GridCell cell( grid, { j, i } );
         auto idx = cell.getIndex();
         std::cout << std::right << std::setw( 12 ) << cells.getElement( idx );
      }
      std::cout << '\n';
   }
   std::cout << '\n';
   //! [print cells]

   //! [initialize faces]
   // Setup values of all faces to an average value of its neighbour cells.
   grid.template forAllEntities< Dimension - 1 >(
      [ = ] __cuda_callable__( const GridFace& face ) mutable
      {
         const CoordinatesType& normal = face.getNormals();
         double sum = 0.0;
         double count = 0.0;
         if( TNL::all( greaterEqual( face.getCoordinates() - normal, 0 ) ) ) {
            auto neighbour = face.template getNeighbourEntity< Dimension >( -normal );
            sum += cells_view[ neighbour.getIndex() ];
            count++;
         }
         if( TNL::all( less( face.getCoordinates(), face.getGrid().getDimensions() ) ) ) {
            auto neighbour = face.template getNeighbourEntity< Dimension >( { 0, 0 } );
            sum += cells_view[ neighbour.getIndex() ];
            count++;
         }
         faces_view[ face.getIndex() ] = sum / count;
      } );
   //! [initialize faces]

   //! [print faces]
   // Print values of all faces in the grid.
   std::cout << "Values of faces ...\n";
   for( int i = grid_size; i >= 0; i-- ) {
      std::cout << std::right << std::setw( 6 ) << " ";
      for( int j = 0; j < grid_size; j++ ) {
         GridFace face( grid, { j, i }, { 0, 1 } );
         auto idx = face.getIndex();
         std::cout << std::right << std::setw( 12 ) << faces.getElement( idx );
      }
      std::cout << '\n';
      if( i > 0 )
         for( int j = 0; j <= grid_size; j++ ) {
            GridFace face( grid, { j, i - 1 }, { 1, 0 } );
            auto idx = face.getIndex();
            std::cout << std::right << std::setw( 12 ) << faces.getElement( idx );
         }
      std::cout << '\n';
   }
   //! [print faces]

   //! [initialize vertices]
   // Setup values of all vertices to an average value of its neighboring cells
   grid.template forAllEntities< 0 >(
      [ = ] __cuda_callable__( const GridVertex& vertex ) mutable
      {
         double sum = 0.0;
         double count = 0.0;
         auto grid_dimensions = vertex.getGrid().getDimensions();
         if( vertex.getCoordinates().x() > 0 && vertex.getCoordinates().y() > 0 ) {
            auto neighbour = vertex.template getNeighbourEntity< Dimension >( { -1, -1 } );
            sum += cells_view[ neighbour.getIndex() ];
            count++;
         }
         if( vertex.getCoordinates().x() > 0 && vertex.getCoordinates().y() < grid_dimensions.y() ) {
            auto neighbour = vertex.template getNeighbourEntity< Dimension >( { -1, 0 } );
            sum += cells_view[ neighbour.getIndex() ];
            count++;
         }
         if( vertex.getCoordinates().x() < grid_dimensions.x() && vertex.getCoordinates().y() > 0 ) {
            auto neighbour = vertex.template getNeighbourEntity< Dimension >( { 0, -1 } );
            sum += cells_view[ neighbour.getIndex() ];
            count++;
         }
         if( TNL::all( less( vertex.getCoordinates(), vertex.getGrid().getDimensions() ) ) ) {
            auto neighbour = vertex.template getNeighbourEntity< Dimension >( { 0, 0 } );
            sum += cells_view[ neighbour.getIndex() ];
            count++;
         }
         vertices_view[ vertex.getIndex() ] = sum / count;
      } );
   //! [initialize vertices]

   //! [print vertices]
   // Print values of all vertices in the grid.
   std::cout << "Values of vertices ....\n";
   for( int i = grid_size; i >= 0; i-- ) {
      for( int j = 0; j <= grid_size; j++ ) {
         GridVertex vertex( grid, { j, i } );
         auto idx = vertex.getIndex();
         std::cout << std::right << std::setw( 12 ) << vertices.getElement( idx );
      }
      std::cout << '\n';
   }
   //! [print vertices]
}

int
main( int argc, char* argv[] )
{
   std::cout << "Traversing grid on CPU...\n";
   traverseGrid< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Traversing grid on CUDA GPU...\n";
   traverseGrid< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}
