#include <iomanip>
#include <iostream>

#include <TNL/Containers/Vector.h>
#include <TNL/Math.h>
#include <TNL/MPI/ScopedInitializer.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGridSynchronizer.h>
#include <TNL/Meshes/DistributedMeshes/reduce.h>

template< typename Device >
void
solveHeatEquation()
{
   //! [setup]
   using GridType = TNL::Meshes::Grid< 2, double, Device, int >;
   using DistributedGridType = TNL::Meshes::DistributedMeshes::DistributedMesh< GridType >;
   using SynchronizerType = TNL::Meshes::DistributedMeshes::DistributedGridSynchronizer< DistributedGridType >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using PointType = typename GridType::PointType;
   using VectorType = TNL::Containers::Vector< double, Device, int >;
   using Cell = typename GridType::Cell;
   using Face = typename GridType::Face;
   //! [setup]

   //! [create distributed grid]
   // The global grid covers the domain [ 0, 1 ] x [ 0, 1 ].
   const int gridSize = 64;
   GridType globalGrid;
   globalGrid.setSizes( CoordinatesType( gridSize ) );
   globalGrid.setDomain( PointType( 0.0 ), PointType( 1.0 ) );

   // Decompose the global grid into the subdomains, one per MPI rank,
   // with one layer of ghost cells between the neighboring subdomains.
   DistributedGridType distributedGrid;
   distributedGrid.setGlobalGrid( globalGrid );
   distributedGrid.setGhostLevels( 1 );
   const GridType& grid = distributedGrid.getLocalMesh();
   //! [create distributed grid]

   //! [allocate vectors]
   // The values of the solution in the cells and the fluxes on the faces of the local grid.
   VectorType u( grid.template getEntitiesCount< 2 >(), 0.0 );
   VectorType fluxes( grid.template getEntitiesCount< 1 >(), 0.0 );
   //! [allocate vectors]

   //! [initial condition]
   // The initial condition is set on the cells owned by this rank. The local grid has the origin and the space
   // steps of the subdomain, so the cell centers are the points of the global domain. The boundary cells keep
   // the zero value, which is the homogeneous Dirichlet boundary condition.
   auto uView = u.getView();
   auto fluxesView = fluxes.getView();
   TNL::Meshes::forInteriorEntities< 2 >(
      grid,
      [ = ] __cuda_callable__( const Cell& cell ) mutable
      {
         const PointType x = cell.getCenter() - PointType( 0.5 );
         uView[ cell.getIndex() ] = TNL::exp( -50.0 * ( x[ 0 ] * x[ 0 ] + x[ 1 ] * x[ 1 ] ) );
      } );
   //! [initial condition]

   //! [synchronizers]
   // The synchronizer copies the values of the ghost entities from the ranks which own them. The values are
   // read and written by the following functions, so they can be stored in any container.
   SynchronizerType synchronizer( distributedGrid );
   auto readCell = [ = ] __cuda_callable__( const Cell& cell )
   {
      return uView[ cell.getIndex() ];
   };
   auto writeCell = [ = ] __cuda_callable__( const Cell& cell, double value ) mutable
   {
      uView[ cell.getIndex() ] = value;
   };
   auto readFace = [ = ] __cuda_callable__( const Face& face )
   {
      return fluxesView[ face.getIndex() ];
   };
   auto writeFace = [ = ] __cuda_callable__( const Face& face, double value ) mutable
   {
      fluxesView[ face.getIndex() ] = value;
   };
   //! [synchronizers]

   //! [total heat]
   // The total heat is the integral of the solution over the domain, i.e. the sum of the heat in the cells.
   // Each cell of the global grid is owned by exactly one rank, so the distributed reduction counts each cell
   // once.
   auto cellHeat = [ = ] __cuda_callable__( const Cell& cell )
   {
      return uView[ cell.getIndex() ] * cell.getMeasure();
   };
   //! [total heat]

   //! [time loop]
   const PointType h = grid.getSpaceSteps();
   const double tau = 0.2 * h[ 0 ] * h[ 0 ];
   const int steps = 400;
   const int communicatorRank = distributedGrid.getCommunicator().rank();
   for( int step = 0; step <= steps; step++ ) {
      if( step % 100 == 0 ) {
         const double heat = TNL::Meshes::reduce< 2 >( distributedGrid, cellHeat );
         if( communicatorRank == 0 )
            std::cout << "step " << std::setw( 4 ) << step << "  time " << std::setw( 9 ) << step * tau << "  total heat "
                      << heat << '\n';
      }
      if( step == steps )
         break;

      // 1. The fluxes need the values in the ghost cells.
      synchronizer.template synchronize< 2 >( readCell, writeCell );

      // 2. The fluxes are computed on the owned faces which do not lie on the global boundary.
      //    The faces on the lower sides of the subdomain are owned by this rank, the faces on the upper
      //    sides are owned by the upper neighbors.
      TNL::Meshes::forInteriorEntities< 1 >(
         grid,
         [ = ] __cuda_callable__( const Face& face ) mutable
         {
            int closer;
            int remoter;
            face.getAdjacentCells( closer, remoter );
            fluxesView[ face.getIndex() ] = ( uView[ remoter ] - uView[ closer ] ) / h[ face.getOrientationIndex() ];
         } );

      // 3. The cells on the upper sides of the subdomain need the fluxes on the faces owned by the neighbors.
      synchronizer.template synchronize< 1 >( readFace, writeFace );

      // 4. The owned cells which do not lie on the global boundary are updated by the fluxes on their faces.
      //    The fluxes were computed from the old values, so the cells can be updated in place.
      TNL::Meshes::forInteriorEntities< 2 >(
         grid,
         [ = ] __cuda_callable__( const Cell& cell ) mutable
         {
            CoordinatesType closer;
            CoordinatesType remoter;
            cell.getAdjacentFacesIndexes( closer, remoter );
            double divergence = 0;
            for( int i = 0; i < 2; i++ )
               divergence += ( fluxesView[ remoter[ i ] ] - fluxesView[ closer[ i ] ] ) / h[ i ];
            uView[ cell.getIndex() ] += tau * divergence;
         } );
   }
   //! [time loop]
}

int
main( int argc, char* argv[] )
{
   TNL::MPI::ScopedInitializer mpi( argc, argv );

   if( TNL::MPI::GetRank() == 0 )
      std::cout << "Solving the heat equation on CPU ...\n";
   solveHeatEquation< TNL::Devices::Host >();

#if defined( __CUDACC__ ) || defined( __HIP__ )
   if( TNL::MPI::GetRank() == 0 )
      std::cout << "Solving the heat equation on GPU (" << TNL::Backend::getBackendName() << ") ...\n";
   solveHeatEquation< TNL::Devices::GPU >();
#endif
}
