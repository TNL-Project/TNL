#pragma once

#include <TNL/Algorithms/scan.h>
#include "GhostZone.h"

namespace TNL {
namespace ParticleSystem {

template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::assignCells( IndexVectorType firstPointIdx,
                                                         IndexVectorType zoneSizeInCells,
                                                         IndexVectorType gridSize )
{
   if constexpr( ParticleConfig::spaceDimension == 2 )
      this->numberOfCellsInZone = zoneSizeInCells[ 0 ] * zoneSizeInCells[ 1 ];
   if constexpr( ParticleConfig::spaceDimension == 3 )
      this->numberOfCellsInZone = zoneSizeInCells[ 0 ] * zoneSizeInCells[ 1 ] * zoneSizeInCells[ 2 ];

   cellsInZone.resize( this->numberOfCellsInZone );
   numberOfParticlesInCell.resize( this->numberOfCellsInZone );
   particlesInZone.resize( numberOfCellsInZone * numberOfParticlesPerCell );

   auto cellsInZone_view = this->cellsInZone.getView();

   if constexpr( ParticleConfig::spaceDimension == 2 ) {
      auto init = [=] __cuda_callable__ ( const IndexVectorType i ) mutable
      {
         const GlobalIndexType idxLinearized = i[ 0 ] + i[ 1 ] * zoneSizeInCells[ 0 ];
         cellsInZone_view[ idxLinearized ] = CellIndexer::EvaluateCellIndex( firstPointIdx + i, gridSize );
      };
      const IndexVectorType begin = { 0, 0 };
      Algorithms::parallelFor< DeviceType >( begin, zoneSizeInCells, init );
   }

   if constexpr( ParticleConfig::spaceDimension == 3 ) {
      auto init = [=] __cuda_callable__ ( const IndexVectorType i ) mutable
      {
         const GlobalIndexType idxLinearized = i[ 0 ] + i[ 1 ] * zoneSizeInCells[ 0 ] + i[ 2 ] * zoneSizeInCells[ 0 ] * zoneSizeInCells[ 1 ];
         cellsInZone_view[ idxLinearized ] = CellIndexer::EvaluateCellIndex( firstPointIdx + i, gridSize );

         //cellsInZone_view[ idxLinearized ] = i[ 2 ] * gridSize[ 0 ] * gridSize[ 1 ] + i[ 1 ] * gridSize[ 0 ] + i[ 0 ];
         //cellsInZone_view[ idxLinearized ] = i[ 2 ] * gridSize[ 0 ] * gridSize[ 1 ] + i[ 0 ] * gridSize[ 1 ] + i[ 1 ];
      };
      const IndexVectorType begin = { 0, 0, 0 };
      Algorithms::parallelFor< DeviceType >( begin, zoneSizeInCells, init );
   }
}

//TODO: Merge both assign functions together
template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::assignCells( const PointType firstPoint,
                                                         const PointType secondPoint,
                                                         IndexVectorType gridSize,
                                                         PointType gridOrigin,
                                                         RealType searchRadius )
{
   const PointType zoneSize = TNL::abs( secondPoint - firstPoint );
   const IndexVectorType zoneSizeInCells = TNL::ceil( zoneSize / searchRadius );
   const IndexVectorType firstPointIdx = (firstPoint - gridOrigin ) / searchRadius;

   if constexpr( ParticleConfig::spaceDimension == 2 )
      this->numberOfCellsInZone = zoneSizeInCells[ 0 ] * zoneSizeInCells[ 1 ];

   if constexpr( ParticleConfig::spaceDimension == 3 )
      this->numberOfCellsInZone = zoneSizeInCells[ 0 ] * zoneSizeInCells[ 1 ] * zoneSizeInCells[ 2 ];

   cellsInZone.resize( this->numberOfCellsInZone );
   numberOfParticlesInCell.resize( this->numberOfCellsInZone );
   particlesInZone.resize( numberOfCellsInZone * numberOfParticlesPerCell );

   auto cellsInZone_view = this->cellsInZone.getView();

   if constexpr( ParticleConfig::spaceDimension == 2 ) {
      auto init = [=] __cuda_callable__ ( const IndexVectorType i ) mutable
      {
         const GlobalIndexType idxLinearized = i[ 0 ] + i[ 1 ] * zoneSizeInCells[ 0 ];
         cellsInZone_view[ idxLinearized ] = CellIndexer::EvaluateCellIndex( firstPointIdx + i, gridSize );
      };
      const IndexVectorType begin = { 0, 0 };
      Algorithms::parallelFor< DeviceType >( begin, zoneSizeInCells, init );
   }
   if constexpr( ParticleConfig::spaceDimension == 3 ) {
      auto init = [=] __cuda_callable__ ( const IndexVectorType i ) mutable
      {
         const GlobalIndexType idxLinearized = i[ 0 ] + i[ 1 ] * zoneSizeInCells[ 0 ] + i[ 2 ] * zoneSizeInCells[ 0 ] * zoneSizeInCells[ 1 ];
         cellsInZone_view[ idxLinearized ] = CellIndexer::EvaluateCellIndex( firstPointIdx + i, gridSize );
      };
      const IndexVectorType begin = { 0, 0, 0 };
      Algorithms::parallelFor< DeviceType >( begin, zoneSizeInCells, init );
   }
}

template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::assignCellsFrame( const IndexVectorType frameFrontOrigin,
                                                              const IndexVectorType frameFrontDims,
                                                              const int frameWidth,
                                                              const IndexVectorType gridSize )
{
   constexpr int dim = ParticleConfig::spaceDimension;

   // -----------------------------------------------------------------------
   // Per-layer, per-face perpendicular extent (outer extrusion only).
   //
   // Face axis d at expansion layer `layer`:
   //   - on perp axis pd, the slab spans frameFrontOrigin[pd] ± expand
   //   - EXCEPT: axes pd < d extend by 1 extra on each side to fill the
   //     corner gaps left by lower-priority (lower-d) faces at this layer.
   //
   // Corner ownership rule (outward only):
   //   Face d=0 claims the full strip at its expansion level.
   //   Face d=1 sees that d=0 already owns pd=0 at ±expand, so d=1 must
   //   reach ±(expand+1) on pd=0 to grab the corner cell. Hence pd < d → +1.
   // -----------------------------------------------------------------------
   auto perpExtent = [&]( int layer, int faceAxis, int perpAxis ) -> GlobalIndexType
   {
      const int expand = layer;  // outward expansion for this layer

      GlobalIndexType origin = frameFrontOrigin[ perpAxis ] - expand;
      GlobalIndexType end    = frameFrontOrigin[ perpAxis ] + frameFrontDims[ perpAxis ] + expand;

      // Fill corners not covered by lower-priority (lower-d) faces
      if( perpAxis < faceAxis ) {
         origin -= 1;
         end    += 1;
      }

      origin = TNL::max( origin, (GlobalIndexType)0 );
      end    = TNL::min( end,    (GlobalIndexType)gridSize[ perpAxis ] );

      return static_cast< GlobalIndexType >( TNL::max( (GlobalIndexType)0, end - origin ) );
   };

   this->numberOfCellsInZone = 0;
   for( int layer = 0; layer < frameWidth; layer++ ) {
       const int expand = layer;
       for( int d = 0; d < dim; d++ ) {
           for( int s : { -1, +1 } ) {
               const GlobalIndexType ifaceCoord = ( s < 0 )
                   ? frameFrontOrigin[ d ] - expand - 1
                   : frameFrontOrigin[ d ] + frameFrontDims[ d ] + expand;

               if( ifaceCoord < 0 || ifaceCoord >= gridSize[ d ] ) continue;

               GlobalIndexType faceNodes = 1;
               for( int pd = 0; pd < dim; pd++ )
                   if( pd != d )
                       faceNodes *= perpExtent( layer, d, pd );
               this->numberOfCellsInZone += faceNodes;
           }
       }
   }

   std::cout << "NUMBER OF CELLS TO GENERATE: " << numberOfCellsInZone << std::endl;
   cellsInZone.resize( this->numberOfCellsInZone );
   numberOfParticlesInCell.resize( this->numberOfCellsInZone );
   particlesInZone.resize( this->numberOfCellsInZone * numberOfParticlesPerCell );

   auto cellsInZone_view = this->cellsInZone.getView();

   // -----------------------------------------------------------------------
   // Pass 2 — fill cells
   // -----------------------------------------------------------------------
   GlobalIndexType offset = 0;

   for( int layer = 0; layer < frameWidth; layer++ ) {
      const int expand = layer;

      for( int d = 0; d < dim; d++ ) {
         for( int s : { -1, +1 } ) {
            // The face sits one step outside the current expansion shell:
            //   s=-1 face: just below the origin side
            //   s=+1 face: just above the end side
            const GlobalIndexType ifaceCoord = ( s < 0 )
               ? frameFrontOrigin[ d ] - expand - 1
               : frameFrontOrigin[ d ] + frameFrontDims[ d ] + expand;

            // Perpendicular extents and origins
            IndexVectorType end = 0;
            end[ d ] = 1;
            for( int pd = 0; pd < dim; pd++ )
               if( pd != d )
                  end[ pd ] = perpExtent( layer, d, pd );

            // Strides for linearisation over perp axes (row-major, d excluded)
            IndexVectorType stride = 0;
            {
               GlobalIndexType running = 1;
               for( int pd = dim - 1; pd >= 0; pd-- ) {
                  if( pd == d ) continue;
                  stride[ pd ] = running;
                  running     *= end[ pd ];
               }
            }

            // Perpendicular origins (with same corner-ownership extension)
            IndexVectorType perpOrigin = 0;
            perpOrigin[ d ] = ifaceCoord;
            for( int pd = 0; pd < dim; pd++ ) {
               if( pd == d ) continue;
               GlobalIndexType o = frameFrontOrigin[ pd ] - expand;
               if( pd < d ) o -= 1;   // match perpExtent's corner extension
               perpOrigin[ pd ] = TNL::max( o, (GlobalIndexType)0 );
            }

            // Skip face if interface coordinate is outside the domain
            if( ifaceCoord < 0 || ifaceCoord >= gridSize[ d ] ) {
               //GlobalIndexType faceNodes = 1;
               //for( int pd = 0; pd < dim; pd++ )
               //   if( pd != d ) faceNodes *= end[ pd ];
               //offset += faceNodes;
               continue;
            }

            const GlobalIndexType    faceOffset = offset;
            const int                iAxis      = d;
            const GlobalIndexType    iCoord     = ifaceCoord;
            const IndexVectorType    pOrigin    = perpOrigin;
            const IndexVectorType    gs         = gridSize;

            auto fill = [=] __cuda_callable__ ( const IndexVectorType idx ) mutable
            {
               // linearise write index
               GlobalIndexType i = faceOffset;
               for( int pd = 0; pd < dim; pd++ )
                  i += idx[ pd ] * stride[ pd ];

               // absolute cell coordinates
               IndexVectorType c = pOrigin;
               c[ iAxis ] = iCoord;
               for( int pd = 0; pd < dim; pd++ )
                  if( pd != iAxis )
                     c[ pd ] += idx[ pd ];

               // guard: skip out-of-domain cells (shouldn't happen after clamping)
               for( int pd = 0; pd < dim; pd++ )
                  if( c[ pd ] < 0 || c[ pd ] >= gs[ pd ] ) return;

               cellsInZone_view[ i ] = CellIndexer::EvaluateCellIndex( c, gs );
            };

            if constexpr( dim == 2 )
               Algorithms::parallelFor< DeviceType >( IndexVectorType{0,0}, end, fill );
            else
               Algorithms::parallelFor< DeviceType >( IndexVectorType{0,0,0}, end, fill );

            GlobalIndexType faceNodes = 1;
            for( int pd = 0; pd < dim; pd++ )
               if( pd != d ) faceNodes *= end[ pd ];
            offset += faceNodes;
         }
      }
   }
}

template< typename ParticleConfig, typename DeviceType >
template< typename Array >
void
ParticleZone< ParticleConfig, DeviceType >::assignCells( Array& inputCells )
{

}

template< typename ParticleConfig, typename DeviceType >
const typename ParticleZone< ParticleConfig, DeviceType >::IndexArrayType&
ParticleZone< ParticleConfig, DeviceType >::getCellsInZone() const
{
   return cellsInZone;
}

template< typename ParticleConfig, typename DeviceType >
typename ParticleZone< ParticleConfig, DeviceType >::IndexArrayType&
ParticleZone< ParticleConfig, DeviceType >::getCellsInZone()
{
   return cellsInZone;
}

template< typename ParticleConfig, typename DeviceType >
const typename ParticleZone< ParticleConfig, DeviceType >::IndexArrayType&
ParticleZone< ParticleConfig, DeviceType >::getParticlesInZone() const
{
   return particlesInZone;
}

template< typename ParticleConfig, typename DeviceType >
typename ParticleZone< ParticleConfig, DeviceType >::IndexArrayType&
ParticleZone< ParticleConfig, DeviceType >::getParticlesInZone()
{
   return particlesInZone;
}

template< typename ParticleConfig, typename DeviceType >
const typename ParticleZone< ParticleConfig, DeviceType >::GlobalIndexType
ParticleZone< ParticleConfig, DeviceType >::getNumberOfParticles() const
{
   return numberOfParticlesInZone;
}

template< typename ParticleConfig, typename DeviceType >
const typename ParticleZone< ParticleConfig, DeviceType >::GlobalIndexType
ParticleZone< ParticleConfig, DeviceType >::getNumberOfCells() const
{
   return numberOfCellsInZone;
}

template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::setNumberOfParticlesPerCell( const GlobalIndexType numberOfParticlesPerCell )
{
   this->numberOfParticlesPerCell = numberOfParticlesPerCell;
}

template< typename ParticleConfig, typename DeviceType >
const typename ParticleZone< ParticleConfig, DeviceType >::GlobalIndexType
ParticleZone< ParticleConfig, DeviceType >::getNumberOfParticlesPerCell() const
{
   return this->numberOfParticlesPerCell;
}

template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::resetParticles()
{
   numberOfParticlesInZone = 0;
   numberOfParticlesInCell = 0;
   particlesInZone = 0;
}

template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::resetZoneCells()
{
   numberOfParticlesInZone = 0;
   particlesInZone = 0;
   cellsInZone = 0;
}

template< typename ParticleConfig, typename DeviceType >
template< typename ParticlesPointer >
void
ParticleZone< ParticleConfig, DeviceType >::collectNumbersOfParticlesInCells( const ParticlesPointer& particles )
{
   const auto firstLastParticle_view = particles->getCellFirstLastParticleList().getConstView();
   const auto cellsInZone_view = this->cellsInZone.getConstView();
   auto numberOfParticlesInCell_view = this->numberOfParticlesInCell.getView();

   auto collectParticlesCounts = [=] __cuda_callable__ ( int i ) mutable
   {
      const GlobalIndexType cell = cellsInZone_view[ i ];
      const PairIndexType firstAndLastParticleInCell = firstLastParticle_view[ cell ];

      if( firstAndLastParticleInCell[ 0 ] != INT_MAX )
      {
         numberOfParticlesInCell_view[ i ] = firstAndLastParticleInCell[ 1 ] - firstAndLastParticleInCell[ 0 ] + 1;
      }
   };
   Algorithms::parallelFor< DeviceType >( 0, this->numberOfCellsInZone, collectParticlesCounts );
}

template< typename ParticleConfig, typename DeviceType >
template< typename ParticlesPointer >
void
ParticleZone< ParticleConfig, DeviceType >::buildParticleList( const ParticlesPointer& particles )
{

   Algorithms::inplaceExclusiveScan( this->numberOfParticlesInCell );
   this->numberOfParticlesInZone = this->numberOfParticlesInCell.getElement( numberOfCellsInZone - 1 ); //without last cell!

   const auto firstLastCellParticle_view = particles->getCellFirstLastParticleList().getConstView();
   const auto cellsInZone_view = this->cellsInZone.getConstView();
   const auto numberOfParticlesInCell_view = this->numberOfParticlesInCell.getConstView();
   auto particlesInZone_view = this->particlesInZone.getView();

   auto collectParticles = [=] __cuda_callable__ ( int i ) mutable //TODO: This i is cell index, rename it
   {
      const GlobalIndexType cell = cellsInZone_view[ i ];
      const PairIndexType firstLastParticle = firstLastCellParticle_view[ cell ];

      if( firstLastParticle[ 0 ] != INT_MAX )
      {
         const GlobalIndexType numberOfPtcsInThisCell = firstLastParticle[ 1 ] - firstLastParticle[ 0 ] + 1;
         const GlobalIndexType particleListCellPrefix = numberOfParticlesInCell_view[ i ];

         for( int j = 0; j < numberOfPtcsInThisCell; j++ )
         {
            particlesInZone_view[ particleListCellPrefix + j ] = firstLastParticle[ 0 ] + j;
         }
      }
   };
   Algorithms::parallelFor< DeviceType >( 0, this->numberOfCellsInZone, collectParticles );

   //Add the particle from last cell
   const PairIndexType firstLastParticleLastCell = firstLastCellParticle_view.getElement(
         cellsInZone_view.getElement( this->numberOfCellsInZone - 1 ) );
   if( firstLastParticleLastCell[ 0 ] != INT_MAX )
   {
      this->numberOfParticlesInZone += ( firstLastParticleLastCell[ 1 ] - firstLastParticleLastCell[ 0 ] + 1 );
   }

}


template< typename ParticleConfig, typename DeviceType >
template< typename ParticlesPointer >
void
ParticleZone< ParticleConfig, DeviceType >::updateParticlesInZone( const ParticlesPointer& particles )
{
   this->resetParticles();
   this->collectNumbersOfParticlesInCells( particles );
   this->buildParticleList( particles );
}

template< typename ParticleConfig, typename DeviceType >
template< typename ParticlesPointer, typename TimeMeasurement >
void
ParticleZone< ParticleConfig, DeviceType >::updateParticlesInZone( const ParticlesPointer& particles, TimeMeasurement& timeMeasurement )
{
   timeMeasurement.start( "zone-reset" );
   this->resetParticles();
   timeMeasurement.stop( "zone-reset" );
   timeMeasurement.start( "zone-collect" );
   this->collectNumbersOfParticlesInCells( particles );
   timeMeasurement.stop( "zone-collect" );
   timeMeasurement.start( "zone-build" );
   this->buildParticleList( particles );
   timeMeasurement.stop( "zone-build" );
}

template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::writeProlog( TNL::Logger& logger ) const noexcept
{
   logger.writeParameter( "Particle zone information:", "" );
   logger.writeParameter( "Number of particles per cell:", numberOfParticlesPerCell, 1 );
   logger.writeParameter( "Number of cells in zone:", numberOfCellsInZone, 1 );
}

/*
template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::saveZoneToVTK(
   const std::string&    filename,
   const IndexVectorType gridSize,
   const PointType      gridOrigin,
   const RealType        searchRadius ) const
{
   constexpr int dim = ParticleConfig::spaceDimension;

   // Copy cell indices to host for writing
   TNL::Containers::Array< GlobalIndexType, TNL::Devices::Host, GlobalIndexType >
         cellsHost( this->numberOfCellsInZone );
   cellsHost = this->cellsInZone;  // device → host copy

   std::ofstream f( filename );
   if( !f.is_open() )
      throw std::runtime_error( "saveZoneToVTK: cannot open " + filename );

   const GlobalIndexType n = this->numberOfCellsInZone;

   // Each cell is written as a voxel (VTK_VOXEL = 11 in 3D) or
   // a pixel (VTK_PIXEL = 8 in 2D). A cell at grid coords (ix, iy)
   // has its corner at gridOrigin + {ix, iy} * searchRadius.

   const int pointsPerCell = ( dim == 2 ) ? 4 : 8;

   f << "# vtk DataFile Version 3.0\n";
   f << "ParticleZone frame\n";
   f << "ASCII\n";
   f << "DATASET UNSTRUCTURED_GRID\n";

   // --- Points ---
   f << "POINTS " << n * pointsPerCell << " float\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ ) {
      // Decode flat cell index back to grid coords
      GlobalIndexType idx = cellsHost[ ci ];
      IndexVectorType gc;
      if constexpr( dim == 2 ) {
         gc[ 0 ] = idx % gridSize[ 0 ];
         gc[ 1 ] = idx / gridSize[ 0 ];
      } else {
         gc[ 0 ] = idx % gridSize[ 0 ];
         gc[ 1 ] = ( idx / gridSize[ 0 ] ) % gridSize[ 1 ];
         gc[ 2 ] = idx / ( gridSize[ 0 ] * gridSize[ 1 ] );
      }

      // Cell corner in physical coords
      const float ox = gridOrigin[ 0 ] + gc[ 0 ] * searchRadius;
      const float oy = gridOrigin[ 1 ] + gc[ 1 ] * searchRadius;
      const float oz = ( dim == 3 ) ? gridOrigin[ 2 ] + gc[ 2 ] * searchRadius : 0.f;
      const float sr = searchRadius;

      if constexpr( dim == 2 ) {
         // 4 corners of the pixel (z = 0)
         f << ox      << " " << oy      << " 0\n";
         f << ox + sr << " " << oy      << " 0\n";
         f << ox      << " " << oy + sr << " 0\n";
         f << ox + sr << " " << oy + sr << " 0\n";
      } else {
         // 8 corners of the voxel
         f << ox      << " " << oy      << " " << oz      << "\n";
         f << ox + sr << " " << oy      << " " << oz      << "\n";
         f << ox      << " " << oy + sr << " " << oz      << "\n";
         f << ox + sr << " " << oy + sr << " " << oz      << "\n";
         f << ox      << " " << oy      << " " << oz + sr << "\n";
         f << ox + sr << " " << oy      << " " << oz + sr << "\n";
         f << ox      << " " << oy + sr << " " << oz + sr << "\n";
         f << ox + sr << " " << oy + sr << " " << oz + sr << "\n";
      }
   }

   // --- Cells ---
   f << "CELLS " << n << " " << n * ( pointsPerCell + 1 ) << "\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ ) {
      f << pointsPerCell;
      for( int p = 0; p < pointsPerCell; p++ )
         f << " " << ci * pointsPerCell + p;
      f << "\n";
   }

   // --- Cell types ---
   // VTK_PIXEL = 8 (2D),  VTK_VOXEL = 11 (3D)
   const int vtkCellType = ( dim == 2 ) ? 8 : 11;
   f << "CELL_TYPES " << n << "\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ )
      f << vtkCellType << "\n";

   // --- Cell data: flat cell index for debugging ---
   f << "CELL_DATA " << n << "\n";
   f << "SCALARS cell_index int 1\n";
   f << "LOOKUP_TABLE default\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ )
      f << cellsHost[ ci ] << "\n";

   f.close();
   std::cout << "saveZoneToVTK: wrote " << n << " cells to " << filename << std::endl;
}
*/

template< typename ParticleConfig, typename DeviceType >
void
ParticleZone< ParticleConfig, DeviceType >::saveZoneToVTK(
   const std::string&    filename,
   const IndexVectorType gridSize,
   const PointType       gridOrigin,
   const RealType        searchRadius ) const
{
   constexpr int dim = ParticleConfig::spaceDimension;

   // Copy to host
   TNL::Containers::Array< GlobalIndexType, TNL::Devices::Host, GlobalIndexType >
         cells( this->numberOfCellsInZone );
   cells = this->cellsInZone;

   const GlobalIndexType n          = this->numberOfCellsInZone;
   const int             cornersPerCell = ( dim == 2 ) ? 4 : 8;
   const int             vtkType        = ( dim == 2 ) ? 8 : 11;  // VTK_PIXEL / VTK_VOXEL

   std::cout <<  "OUTPUT: Cells in zone: " << this->numberOfCellsInZone << std::endl;
   std::cout <<  "cells: " << cells << std::endl;

   std::ofstream f( filename );
   if( !f )
      throw std::runtime_error( "saveZoneToVTK: cannot open " + filename );

   // -----------------------------------------------------------------------
   // Header
   // -----------------------------------------------------------------------
   f << "# vtk DataFile Version 3.0\n"
     << "ParticleZone\n"
     << "ASCII\n"
     << "DATASET UNSTRUCTURED_GRID\n";

   // -----------------------------------------------------------------------
   // Points — one cell = cornersPerCell points
   // -----------------------------------------------------------------------
   f << "POINTS " << n * cornersPerCell << " float\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ ) {

      //// Decode flat index → integer grid coords
      //GlobalIndexType idx = cells[ ci ];
      //IndexVectorType gc  = 0;
      //if constexpr( dim == 2 ) {
      //   gc[ 0 ] = idx % gridSize[ 0 ];
      //   gc[ 1 ] = idx / gridSize[ 0 ];
      //} else {
      //   gc[ 0 ] = idx % gridSize[ 0 ];
      //   gc[ 1 ] = ( idx / gridSize[ 0 ] ) % gridSize[ 1 ];
      //   gc[ 2 ] = idx / ( gridSize[ 0 ] * gridSize[ 1 ] );
      //}

      // Physical origin of this cell
      const IndexVectorType gc = CellIndexer::GetCellCoordinates( cells[ ci ], gridSize );
      const float x0 = gridOrigin[ 0 ] + gc[ 0 ] * searchRadius;
      const float y0 = gridOrigin[ 1 ] + gc[ 1 ] * searchRadius;
      const float z0 = ( dim == 3 ) ? gridOrigin[ 2 ] + gc[ 2 ] * searchRadius : 0.f;
      const float sr = static_cast< float >( searchRadius );

      // Write corners in VTK_PIXEL / VTK_VOXEL order
      if constexpr( dim == 2 ) {
         f << x0      << " " << y0      << " 0\n";
         f << x0 + sr << " " << y0      << " 0\n";
         f << x0      << " " << y0 + sr << " 0\n";
         f << x0 + sr << " " << y0 + sr << " 0\n";
      } else {
         f << x0      << " " << y0      << " " << z0      << "\n";
         f << x0 + sr << " " << y0      << " " << z0      << "\n";
         f << x0      << " " << y0 + sr << " " << z0      << "\n";
         f << x0 + sr << " " << y0 + sr << " " << z0      << "\n";
         f << x0      << " " << y0      << " " << z0 + sr << "\n";
         f << x0 + sr << " " << y0      << " " << z0 + sr << "\n";
         f << x0      << " " << y0 + sr << " " << z0 + sr << "\n";
         f << x0 + sr << " " << y0 + sr << " " << z0 + sr << "\n";
      }
   }

   // -----------------------------------------------------------------------
   // Cells
   // -----------------------------------------------------------------------
   f << "CELLS " << n << " " << n * ( cornersPerCell + 1 ) << "\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ ) {
      f << cornersPerCell;
      for( int p = 0; p < cornersPerCell; p++ )
         f << " " << ci * cornersPerCell + p;
      f << "\n";
   }

   // -----------------------------------------------------------------------
   // Cell types
   // -----------------------------------------------------------------------
   f << "CELL_TYPES " << n << "\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ )
      f << vtkType << "\n";

   // -----------------------------------------------------------------------
   // Cell data — flat index and grid coords for inspection in ParaView
   // -----------------------------------------------------------------------
   f << "CELL_DATA " << n << "\n";

   f << "SCALARS flat_index int 1\nLOOKUP_TABLE default\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ )
      f << cells[ ci ] << "\n";

   f << "SCALARS gc_x int 1\nLOOKUP_TABLE default\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ )
      f << cells[ ci ] % gridSize[ 0 ] << "\n";

   f << "SCALARS gc_y int 1\nLOOKUP_TABLE default\n";
   for( GlobalIndexType ci = 0; ci < n; ci++ )
      f << ( cells[ ci ] / gridSize[ 0 ] ) % gridSize[ 1 ] << "\n";

   if constexpr( dim == 3 ) {
      f << "SCALARS gc_z int 1\nLOOKUP_TABLE default\n";
      for( GlobalIndexType ci = 0; ci < n; ci++ )
         f << cells[ ci ] / ( gridSize[ 0 ] * gridSize[ 1 ] ) << "\n";
   }

   std::cout << "saveZoneToVTK: " << n << " cells → " << filename << "\n";
}

} // Particles
} // TNL
