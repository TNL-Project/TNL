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

} // Particles
} // TNL
