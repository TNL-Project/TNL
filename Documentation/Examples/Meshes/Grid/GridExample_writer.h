#include <iostream>
#include <iomanip>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/Writers/VTIWriter.h>
#include <TNL/Meshes/Writers/VTKWriter.h>
#include <TNL/Meshes/Writers/GnuplotWriter.h>
#include <TNL/Containers/Vector.h>

template< typename Device >
void
writeGrid()
{
   // Define grid dimension and size.
   static constexpr int Dimension = 2;
   const int grid_size = 5;

   // Setup necessary type.
   using GridType = TNL::Meshes::Grid< Dimension, double, Device >;
   using PointType = typename GridType::PointType;
   using VectorType = TNL::Containers::Vector< double, Device >;

   // Setup types of grid entities.
   using GridCell = typename GridType::Cell;
   using GridVertex = typename GridType::Vertex;

   // Create an instance of a grid.
   GridType grid( grid_size );
   PointType origin( 0.0 );
   PointType proportions( 1.0 );
   grid.setDomain( origin, proportions );

   // Allocate vectors for values stored in particular grid entities.
   VectorType cells( grid.template getEntitiesCount< Dimension >(), 0.0 );
   VectorType vertices( grid.template getEntitiesCount< 0 >(), 0.0 );

   // Prepare views for the data at the grid entities so that we can
   // manipulate them in lambda functions running eventually on GPU.
   auto cells_view = cells.getView();
   auto vertices_view = vertices.getView();

   // Setup value of each cell to its index in the grid.
   grid.template forAllEntities< Dimension >(
      [ = ] __cuda_callable__( const GridCell& cell ) mutable
      {
         cells_view[ cell.getIndex() ] = cell.getIndex();
      } );

   // Write values of all cells in the grid into a file in VTI format.
   TNL::String cells_file_name_vti( "GridExample-cells-values-" + TNL::getType( Device{} ) + ".vti" );
   std::cout << "Writing a file " << cells_file_name_vti << " ..." << std::endl;
   std::fstream cells_file_vti;
   cells_file_vti.open( cells_file_name_vti.getString(), std::ios::out );
   TNL::Meshes::Writers::VTIWriter< GridType > cells_vti_writer( cells_file_vti );
   cells_vti_writer.writeImageData( grid );
   cells_vti_writer.writeCellData( cells, "cell-values" );

   // Write values of all cells in the grid into a file in VTK format.
   TNL::String cells_file_name_vtk( "GridExample-cells-values-" + TNL::getType( Device{} ) + ".vtk" );
   std::cout << "Writing a file " << cells_file_name_vtk << " ..." << std::endl;
   std::fstream cells_file_vtk;
   cells_file_vtk.open( cells_file_name_vtk.getString(), std::ios::out );
   TNL::Meshes::Writers::VTKWriter< GridType > cells_vtk_writer( cells_file_vtk );
   cells_vtk_writer.writeEntities( grid );
   cells_vtk_writer.writeCellData( cells, "cell-values" );

   // Write values of all cells in the grid into a file in Gnuplot format.
   TNL::String cells_file_name_gplt( "GridExample-cells-values-" + TNL::getType( Device{} ) + ".gplt" );
   std::cout << "Writing a file " << cells_file_name_gplt << " ..." << std::endl;
   std::fstream cells_file_gplt;
   cells_file_gplt.open( cells_file_name_gplt.getString(), std::ios::out );
   TNL::Meshes::Writers::GnuplotWriter< GridType > cells_gplt_writer( cells_file_gplt );
   cells_gplt_writer.writeEntities( grid );
   cells_gplt_writer.writeCellData( grid, cells, "cell-values" );

   // Setup values of all vertexes to an average value of its neighboring cells.
   grid.template forAllEntities< 0 >( [=] __cuda_callable__ ( const GridVertex& vertex ) mutable {
      double sum = 0.0;
      double count = 0.0;
      auto grid_dimensions = vertex.getGrid().getDimensions();
      if( vertex.getCoordinates().x() > 0 && vertex.getCoordinates().y() > 0 ) {
         sum += cells_view[ vertex.template getEntityIndex< Dimension >( { -1,-1 }, 0 ) ];
         count++;
      }
      if( vertex.getCoordinates().x() > 0 && vertex.getCoordinates().y() < grid_dimensions.y() ) {
         sum += cells_view[ vertex.template getEntityIndex< Dimension >( { -1,0 }, 0 ) ];
         count++;
      }
      if( vertex.getCoordinates().x() < grid_dimensions.x() && vertex.getCoordinates().y() > 0 ) {
         sum += cells_view[ vertex.template getEntityIndex< Dimension >( { 0,-1 }, 0 ) ];
         count++;
      }
      if( TNL::all(less( vertex.getCoordinates(), vertex.getGrid().getDimensions() )) ) {
         sum += cells_view[ vertex.template getEntityIndex< Dimension >( { 0,0 }, 0 ) ];
         count++;
      }
      vertexes_view[ vertex.getIndex() ] = sum / count;
   } );

   // Write values of all vertices in the grid to a file in VTI format
   TNL::String vertices_file_name_vti( "GridExample-vertices-values-" + TNL::getType( Device{} ) + ".vti" );
   std::cout << "Writing a file " << vertices_file_name_vti << " ..." << std::endl;
   std::fstream vertices_file_vti;
   vertices_file_vti.open( vertices_file_name_vti.getString(), std::ios::out );
   TNL::Meshes::Writers::VTIWriter< GridType > vertices_vti_writer( vertices_file_vti );
   vertices_vti_writer.writeImageData( grid );
   vertices_vti_writer.writePointData( vertices, "vertices-values" );

   // Write values of all vertices in the grid to a file in VTK format
   TNL::String vertices_file_name_vtk( "GridExample-vertices-values-" + TNL::getType( Device{} ) + ".vtk" );
   std::cout << "Writing a file " << vertices_file_name_vtk << " ..." << std::endl;
   std::fstream vertices_file_vtk;
   vertices_file_vtk.open( vertices_file_name_vtk.getString(), std::ios::out );
   TNL::Meshes::Writers::VTIWriter< GridType > vertices_vtk_writer( vertices_file_vtk );
   vertices_vtk_writer.writeEntities( grid );
   vertices_vtk_writer.writePointData( vertices, "vertices-values" );

   // Write values of all vertices in the grid to a file in Gnuplot format
   TNL::String vertices_file_name_gplt( "GridExample-vertices-values-" + TNL::getType( Device{} ) + ".gplt" );
   std::cout << "Writing a file " << vertices_file_name_gplt << " ..." << std::endl;
   std::fstream vertices_file_gplt;
   vertices_file_gplt.open( vertices_file_name_gplt.getString(), std::ios::out );
   TNL::Meshes::Writers::GnuplotWriter< GridType > vertices_gplt_writer( vertices_file_gplt );
   vertices_gplt_writer.writeEntities( grid );
   vertices_gplt_writer.writePointData( grid, vertices, "vertices-values" );
}

int
main( int argc, char* argv[] )
{
   std::cout << "Traversing grid on CPU..." << std::endl;
   writeGrid< TNL::Devices::Host >();

#ifdef __CUDACC__
   std::cout << "Traversing grid on CUDA GPU..." << std::endl;
   writeGrid< TNL::Devices::Cuda >();
#endif
   return EXIT_SUCCESS;
}
