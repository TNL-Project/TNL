#include "ParticlesLinkedList.h"
#include "details/details.h"
#include <climits>

namespace TNL {
namespace ParticleSystem {

template< typename ParticleConfig, typename DeviceType >
void
ParticlesLinkedList< ParticleConfig, DeviceType >::setGridDimensions( const IndexVectorType& dimensions )
{
   this->gridDimension = dimensions;
   const IndexVectorType dimensionsWithOverlap = dimensions + 2 * this->getOverlapWidth();
   GlobalIndexType numberOfCells;

   if constexpr( ParticleConfig::spaceDimension == 2 )
      numberOfCells = dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ];
   if constexpr( ParticleConfig::spaceDimension == 3 )
      numberOfCells = dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ] * dimensionsWithOverlap[ 2 ];

   firstLastCellParticle.setSize( numberOfCells );
   numberOfParticlesInCells.setSize( numberOfCells );
   this->resetListWithIndices();
}

template< typename ParticleConfig, typename DeviceType >
void
ParticlesLinkedList< ParticleConfig, DeviceType >::setOverlapWidth( const GlobalIndexType width )
{
   this->overlapWidth = width;
   const IndexVectorType dimensionsWithOverlap = this->getGridDimensions() + 2 * width;
   GlobalIndexType numberOfCells;

   if constexpr( ParticleConfig::spaceDimension == 2 )
      numberOfCells = dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ];
   if constexpr( ParticleConfig::spaceDimension == 3 )
      numberOfCells = dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ] * dimensionsWithOverlap[ 2 ];

   firstLastCellParticle.setSize( numberOfCells );
   numberOfParticlesInCells.setSize( numberOfCells );
   this->resetListWithIndices();
}

template< typename ParticleConfig, typename DeviceType >
void
ParticlesLinkedList< ParticleConfig, DeviceType >::setSize( const GlobalIndexType& size )
{
   BaseType::setSize( size );
   this->particleCellInidices.setSize( size );
   this->indicesOfParticlesInCells.setSize( size );

   particleCellInidices = 0;
   indicesOfParticlesInCells = 0;
}

template< typename ParticleConfig, typename Device >
const typename ParticlesLinkedList< ParticleConfig, Device >::PairIndexArrayType&
ParticlesLinkedList< ParticleConfig, Device >::getCellFirstLastParticleList() const
{
   return firstLastCellParticle;
}

template< typename ParticleConfig, typename Device >
typename ParticlesLinkedList< ParticleConfig, Device >::PairIndexArrayType&
ParticlesLinkedList< ParticleConfig, Device >::getCellFirstLastParticleList()
{
   return firstLastCellParticle;
}

template< typename ParticleConfig, typename Device >
const typename ParticlesLinkedList< ParticleConfig, Device >::CellIndexArrayType&
ParticlesLinkedList< ParticleConfig, Device >::getParticleCellIndices() const
{
   return particleCellInidices;
}

template< typename ParticleConfig, typename Device >
typename ParticlesLinkedList< ParticleConfig, Device >::CellIndexArrayType&
ParticlesLinkedList< ParticleConfig, Device >::getParticleCellIndices()
{
   return particleCellInidices;
}

template< typename ParticleConfig, typename Device >
typename ParticlesLinkedList< ParticleConfig, Device >::NeighborsLoopParams
ParticlesLinkedList< ParticleConfig, Device >::getCLLSearchToken()
{
   NeighborsLoopParams searchToken;

   searchToken.numberOfParticles = this->getNumberOfParticles();
   searchToken.gridSize = this->getGridDimensionsWithOverlap();
   searchToken.gridOrigin = this->getGridOriginWithOverlap();
   searchToken.searchRadius = this->getSearchRadius();
   searchToken.view_firstLastCellParticle.bind( this->getCellFirstLastParticleList().getView() );

   return searchToken;
}

template< typename ParticleConfig, typename Device >
template< typename ParticlesPointerType >
typename ParticlesLinkedList< ParticleConfig, Device >::NeighborsLoopParams
ParticlesLinkedList< ParticleConfig, Device >::getCLLSearchToken( ParticlesPointerType& particlesToSearch )
{
   NeighborsLoopParams searchToken;

   searchToken.numberOfParticles = particlesToSearch->getNumberOfParticles();
   searchToken.gridSize = particlesToSearch->getGridDimensionsWithOverlap();
   searchToken.gridOrigin = particlesToSearch->getGridOriginWithOverlap();
   searchToken.searchRadius = particlesToSearch->getSearchRadius();
   searchToken.view_firstLastCellParticle.bind( particlesToSearch->getCellFirstLastParticleList().getView() );

   return searchToken;
}

template< typename ParticleConfig, typename Device >
typename ParticlesLinkedList< ParticleConfig, Device >::NeighborsLoopParams
ParticlesLinkedList< ParticleConfig, Device >::getSearchToken()
{
   return this->getCLLSearchToken();
}

template< typename ParticleConfig, typename Device >
template< typename ParticlesPointerType >
typename ParticlesLinkedList< ParticleConfig, Device >::NeighborsLoopParams
ParticlesLinkedList< ParticleConfig, Device >::getSearchToken( ParticlesPointerType& particlesToSearch )
{
   return this->getCLLSearchToken( particlesToSearch );
}

template< typename ParticleConfig, typename Device >
template< typename UseWithDomainDecomposition, std::enable_if_t< ! UseWithDomainDecomposition::value, bool > Enabled >
void
ParticlesLinkedList< ParticleConfig, Device >::computeParticleCellIndices()
{
   const RealType searchRadius = this->radius;
   const PointType gridOrigin = this->gridOrigin;
   const IndexVectorType gridDimension = this->gridDimension;
   auto view_particeCellIndices = this->particleCellInidices.getView();
   const auto view_points = this->points.getConstView();

   const GlobalIndexType numberOfCells = numberOfParticlesInCells.getSize();
   auto view_numberOfParticlesInCells = numberOfParticlesInCells.getView();
   auto view_indicesOfParticlesInCells = indicesOfParticlesInCells.getView();

   // reset the counts of number of particles in cells
   view_numberOfParticlesInCells = 0;

   auto indexParticles = [ = ] __cuda_callable__( GlobalIndexType i ) mutable
   {
      const PointType point = view_points[ i ];
      if( point[ 0 ] == FLT_MAX || point[ 1 ] == FLT_MAX ) {
         view_particeCellIndices[ i ] = INT_MAX;
      }
      else {
         const IndexVectorType cellCoords = TNL::floor( ( point - gridOrigin ) / searchRadius );
         const CellIndexType cellIndex = CellIndexer::EvaluateCellIndex( cellCoords, gridDimension );

	     // if ( cellIndex < 0 || cellIndex >= numberOfCells ) {
        //    view_particeCellIndices[ i ] = INT_MAX;
        // }
        // else {
            view_particeCellIndices[ i ] = cellIndex;
            view_indicesOfParticlesInCells[ i ] = Algorithms::AtomicOperations< DeviceType >::add(
                  view_numberOfParticlesInCells[ cellIndex ], 1 );
        //}
      }
   };
   Algorithms::parallelFor< DeviceType >( 0, this->numberOfParticles, indexParticles );
   Algorithms::inplaceExclusiveScan( numberOfParticlesInCells, 0, numberOfCells );
}

template< typename ParticleConfig, typename Device >
template< typename UseWithDomainDecomposition, std::enable_if_t< UseWithDomainDecomposition::value, bool > Enabled >
void
ParticlesLinkedList< ParticleConfig, Device >::computeParticleCellIndices()
{
   const RealType searchRadius = this->radius;
   const PointType gridRefOrigin = this->gridReferentialOrigin;
   const IndexVectorType gridOriginGlobalCoordsWithOverlap = this->getGridOriginGlobalCoordsWithOverlap();
   const IndexVectorType gridDimensionWithOverlap = this->getGridDimensionsWithOverlap();

   auto view_particeCellIndices = this->particleCellInidices.getView();
   const auto view_points = this->points.getConstView();

   const GlobalIndexType numberOfCells = numberOfParticlesInCells.getSize();
   auto view_numberOfParticlesInCells = numberOfParticlesInCells.getView();
   auto view_indicesOfParticlesInCells = indicesOfParticlesInCells.getView();

   // reset the counts of number of particles in cells
   view_numberOfParticlesInCells = 0;

   auto indexParticles = [ = ] __cuda_callable__( GlobalIndexType i ) mutable
   {
      const PointType point = view_points[ i ];
      if( point[ 0 ] == FLT_MAX || point[ 1 ] == FLT_MAX ) {
         view_particeCellIndices[ i ] = INT_MAX;
      }
      else {
         const IndexVectorType cellCoordsGlobal = TNL::floor( ( point - gridRefOrigin ) / searchRadius );
         const IndexVectorType cellCoords = cellCoordsGlobal - gridOriginGlobalCoordsWithOverlap;
         const CellIndexType cellIndex = CellIndexer::EvaluateCellIndex( cellCoords, gridDimensionWithOverlap );

         view_particeCellIndices[ i ] = cellIndex;
         view_indicesOfParticlesInCells[ i ] = Algorithms::AtomicOperations< DeviceType >::add(
               view_numberOfParticlesInCells[ cellIndex ], 1 );
      }
   };
   Algorithms::parallelFor< DeviceType >( 0, this->numberOfParticles, indexParticles );
   Algorithms::inplaceExclusiveScan( numberOfParticlesInCells, 0, numberOfCells );
}

//FIXME: Temp. function. Remove after resolving the overlaps.
template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::removeParitclesOutOfDomainPositionBased()
{
   const IndexVectorType gridOrigin = this->getGridOrigin();
   const IndexVectorType gridDimensions = this->getGridDimensions();
   const RealType searchRadius = this->getSearchRadius();
   const PointType domainOrigin = gridOrigin;
   const PointType domainSize = searchRadius * gridDimensions;

   auto view_points = this->points.getView();

   auto checkParticlePosition = [ = ] __cuda_callable__( int i ) mutable
   {
      const PointType point = view_points[ i ];
      // if the particle is already removed, skip
      if( point[ 0 ] == FLT_MAX )
         return 0;

      if( this->isInsideDomain( point, domainOrigin, domainSize ) ) {
         return 0;
      }
      else {
         view_points[ i ] = FLT_MAX;
         return 1;
      }
   };
   const GlobalIndexType numberOfParticlesToRemove =
      Algorithms::reduce< DeviceType >( 0, this->numberOfParticles, checkParticlePosition, TNL::Plus() );
   this->setNumberOfParticlesToRemove( this->getNumberOfParticlesToRemove() + numberOfParticlesToRemove );
}

template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::removeParitclesOutOfDomain()
{
   //FIXME: Global grid origin should be shifted aswell with search radius. Or it should not?
   const PointType gridRefOrigin = this->getGridReferentialOrigin();
   const RealType searchRadius = this->getSearchRadius();
   const IndexVectorType gridOriginGlobalCoordsWithOverlap = this->getGridOriginGlobalCoordsWithOverlap();
   const IndexVectorType gridDimensionsWithOverlap = this->getGridDimensionsWithOverlap();
   auto view_points = this->points.getView();

   auto checkParticlePosition = [ = ] __cuda_callable__( int i ) mutable
   {
      const PointType point = view_points[ i ];
      // if the particle is already removed, skip
      if( point[ 0 ] == FLT_MAX )
         return 0;
      const IndexVectorType cellGlobalCoords = TNL::floor( ( point - gridRefOrigin ) / searchRadius );
      const IndexVectorType cellCoords = cellGlobalCoords - gridOriginGlobalCoordsWithOverlap;

      if( this->isInsideDomain( cellCoords, gridDimensionsWithOverlap ) ) {
         return 0;
      }
      else {
         view_points[ i ] = FLT_MAX;
         return 1;
      }
   };
   const GlobalIndexType numberOfParticlesToRemove =
      Algorithms::reduce< DeviceType >( 0, this->numberOfParticles, checkParticlePosition, TNL::Plus() );
   this->setNumberOfParticlesToRemove( this->getNumberOfParticlesToRemove() + numberOfParticlesToRemove );
}

template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::sortParticles()
{
   const GlobalIndexType numberOfParticle = this->getNumberOfParticles();
   const auto view_particeCellIndices = particleCellInidices.getConstView();
   const auto view_numberOfParticlesInCells = numberOfParticlesInCells.getConstView();
   const auto view_indicesOfParticlesInCells = indicesOfParticlesInCells.getConstView();
   auto view_sortPermutations = this->sortPermutations.getView();

   auto buildPermutations = [ = ] __cuda_callable__ ( GlobalIndexType i ) mutable
   {
      const GlobalIndexType cellIndex = view_particeCellIndices[ i ];
      if( cellIndex != INT_MAX ) {
         const GlobalIndexType newIndex = view_numberOfParticlesInCells[ cellIndex ] + view_indicesOfParticlesInCells[ i ];
         view_sortPermutations[ newIndex ] = i;
      }
   };
   Algorithms::parallelFor< Device >( 0, numberOfParticle, buildPermutations );
}

template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::resetListWithIndices()
{
   this->firstLastCellParticle = INT_MAX;
}

template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::particlesToCells()
{
   const GlobalIndexType numberOfParticles = this->getNumberOfParticles();
   const GlobalIndexType numberOfCells = this->firstLastCellParticle.getSize();
   auto view_firstLastCellParticle = this->firstLastCellParticle.getView();
   const auto view_numberOfParticlesInCells = this->numberOfParticlesInCells.getConstView();

   // no particles, skip
   if( numberOfParticles == 0 ) {
      return;
   }

   auto bucketing = [=] __cuda_callable__ ( GlobalIndexType i ) mutable
   {
      if( view_numberOfParticlesInCells[ i ] != view_numberOfParticlesInCells[ i + 1 ] ){
         view_firstLastCellParticle[ i ][ 0 ] = view_numberOfParticlesInCells[ i ];
         view_firstLastCellParticle[ i ][ 1 ] = view_numberOfParticlesInCells[ i + 1 ] - 1;
      }
   };
   Algorithms::parallelFor< DeviceType >( 0, numberOfCells - 1, bucketing ); //TODO: range?

   // -resolve last cell manually
   const GlobalIndexType numberOfParticlesInLastCell = view_numberOfParticlesInCells.getElement( numberOfCells - 1 );
   if( numberOfParticlesInLastCell != numberOfParticles )
      view_firstLastCellParticle.setElement( numberOfCells - 1, { numberOfParticlesInLastCell, numberOfParticles - 1 } );
   else
      view_firstLastCellParticle.setElement( numberOfCells - 1, { INT_MAX, INT_MAX } );
}

template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::searchForNeighbors()
{
   if( this->getNumberOfParticles() == 0 )
      return;
   resetListWithIndices();
   computeParticleCellIndices();
   sortParticles();
   // update number of particles - removed particles with invalid positions are shifted at the end of the array
   if( this->getNumberOfParticlesToRemove() != 0 ) {
      this->setNumberOfParticles( this->getNumberOfParticles() - this->getNumberOfParticlesToRemove() );
      this->setNumberOfParticlesToRemove( 0 );
   }
   this->reorderParticles();
   particlesToCells();
}

template< typename ParticleConfig, typename Device >
template< typename Function, typename... FunctionArgs >
void
ParticlesLinkedList< ParticleConfig, Device >::neighborsLoop( const GlobalIndexType& i,
                                                              const PointType& r_i,
                                                              NeighborsLoopParams neighborsLoopParams,
                                                              Function f,
                                                              FunctionArgs... args )
{
   NeighborsLoop::exec( i, r_i, neighborsLoopParams, f, args...);
}

template< typename ParticleConfig, typename DeviceType >
void
ParticlesLinkedList< ParticleConfig, DeviceType >::writeProlog( TNL::Logger& logger ) const noexcept
{
   BaseType::writeProlog( logger );
}

}  //namespace ParticleSystem
}  //namespace TNL
