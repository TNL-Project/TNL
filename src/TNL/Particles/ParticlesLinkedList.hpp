#include "ParticlesLinkedList.h"
#include "details/details.h"
#include <climits>
#include <limits>
#include <vector>

namespace TNL {
namespace ParticleSystem {

template< typename ParticleConfig, typename DeviceType >
void
ParticlesLinkedList< ParticleConfig, DeviceType >::setGridDimensions( const IndexVectorType& dimensions )
{
   this->gridDimension = dimensions;
   const IndexVectorType dimensionsWithOverlap = dimensions + 2 * this->getOverlapWidth();
   if constexpr( ParticleConfig::spaceDimension == 2 )
      firstLastCellParticle.setSize( dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ] );
   if constexpr( ParticleConfig::spaceDimension == 3 )
      firstLastCellParticle.setSize( dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ] * dimensionsWithOverlap[ 2 ] );

   this->resetListWithIndices();
}

template< typename ParticleConfig, typename DeviceType >
void
ParticlesLinkedList< ParticleConfig, DeviceType >::setOverlapWidth( const GlobalIndexType width )
{
   this->overlapWidth = width;
   const IndexVectorType dimensionsWithOverlap = this->getGridDimensions() + 2 * width;
   if constexpr( ParticleConfig::spaceDimension == 2 )
      firstLastCellParticle.setSize( dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ] );
   if constexpr( ParticleConfig::spaceDimension == 3 )
      firstLastCellParticle.setSize( dimensionsWithOverlap[ 0 ] * dimensionsWithOverlap[ 1 ] * dimensionsWithOverlap[ 2 ] );

   this->resetListWithIndices();
}

template< typename ParticleConfig, typename DeviceType >
void
ParticlesLinkedList< ParticleConfig, DeviceType >::setSize( const GlobalIndexType& size )
{
   BaseType::setSize( size );
   this->particleCellInidices.setSize( size );
   //TODO: is this necessary or not?
   //particleCellInidices = INT_MAX;
   //firstLastCellParticle = INT_MAX;
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
   const auto points_view = this->points.getConstView();

   auto indexParticles = [ = ] __cuda_callable__( GlobalIndexType i ) mutable
   {
      if( points_view[ i ][ 0 ] == FLT_MAX || points_view[ i ][ 1 ] == FLT_MAX ) {
         view_particeCellIndices[ i ] = INT_MAX;
      }
      else {
         const IndexVectorType cellCoords = TNL::floor( ( points_view[ i ] - gridOrigin ) / searchRadius );
         view_particeCellIndices[ i ] = CellIndexer::EvaluateCellIndex( cellCoords, gridDimension );
      }
   };
   Algorithms::parallelFor< DeviceType >( 0, this->numberOfParticles, indexParticles );
}

template< typename ParticleConfig, typename Device >
template< typename UseWithDomainDecomposition, std::enable_if_t< UseWithDomainDecomposition::value, bool > Enabled >
void
ParticlesLinkedList< ParticleConfig, Device >::computeParticleCellIndices()
{
   //FIXME: Global grid origin should be shifted aswell with search radius. Or it should not?
   const PointType gridRefOrigin = this->gridReferentialOrigin;
   const RealType searchRadius = this->radius;
   const IndexVectorType gridOriginGlobalCoordsWithOverlap = this->getGridOriginGlobalCoordsWithOverlap();
   const IndexVectorType gridDimensionWithOverlap = this->getGridDimensionsWithOverlap();
   const auto view_points = this->points.getConstView();
   auto view_particeCellIndices = this->particleCellInidices.getView();

   auto indexParticles = [ = ] __cuda_callable__( GlobalIndexType i ) mutable
   {
      const PointType point = view_points[ i ];
      if( view_points[ i ][ 0 ] == FLT_MAX || view_points[ i ][ 1 ] == FLT_MAX ) {
         view_particeCellIndices[ i ] = INT_MAX;
      }
      else {
         const IndexVectorType cellGlobalCoords = TNL::floor( ( point - gridRefOrigin ) / searchRadius );
         const IndexVectorType cellCoords = cellGlobalCoords - gridOriginGlobalCoordsWithOverlap;
         view_particeCellIndices[ i ] = CellIndexer::EvaluateCellIndex( cellCoords, gridDimensionWithOverlap );
      }
   };
   Algorithms::parallelFor< DeviceType >( 0, this->numberOfParticles, indexParticles );
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
   this->sortPermutations->forAllElements(
      [] __cuda_callable__( int i, int& value )
      {
         value = i;
      } );
   using ThrustDeviceType = TNL::Thrust::ThrustExecutionPolicy< Device >;
   ThrustDeviceType thrustDevice;
   thrust::sort_by_key( thrustDevice,
                        this->particleCellInidices.getArrayData(),
                        this->particleCellInidices.getArrayData() + numberOfParticle,
                        this->sortPermutations->getArrayData() );
   thrust::gather( thrustDevice,
                   this->sortPermutations->getArrayData(),
                   this->sortPermutations->getArrayData() + numberOfParticle,
                   this->points.getArrayData(),
                   this->points_swap.getArrayData() );
   this->points.swap( this->points_swap );
}

template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::resetListWithIndices()
{
   auto view_firstLastCellParticle = this->firstLastCellParticle.getView();
   auto init = [ = ] __cuda_callable__( int i ) mutable
   {
      view_firstLastCellParticle[ i ] = INT_MAX;
   };
   Algorithms::parallelFor< DeviceType >( 0, this->firstLastCellParticle.getSize(), init );
}

template< typename ParticleConfig, typename Device >
void
ParticlesLinkedList< ParticleConfig, Device >::particlesToCells()
{
   const GlobalIndexType numberOfParticles = this->getNumberOfParticles();
   auto view_firstLastCellParticle = this->firstLastCellParticle.getView();
   const auto view_particleCellIndex = this->particleCellInidices.getView();

   if( numberOfParticles == 0 ) {
      return;
   }
   else if( numberOfParticles == 1 ) {
      view_firstLastCellParticle.setElement( view_particleCellIndex.getElement( 0 ), { 0, 0 } );
      return;
   }

   //resolve first particle
   view_firstLastCellParticle.setElement(
      view_particleCellIndex.getElement( 0 ),
      { 0, ( view_particleCellIndex.getElement( 0 ) != view_particleCellIndex.getElement( 0 + 1 ) ) ? 0 : INT_MAX } );

   //[1, N-1]
   auto init = [ = ] __cuda_callable__( int i ) mutable
   {
      if( view_particleCellIndex[ i ] != view_particleCellIndex[ i - 1 ] )
         view_firstLastCellParticle[ view_particleCellIndex[ i ] ][ 0 ] = i;
      if( view_particleCellIndex[ i ] != view_particleCellIndex[ i + 1 ] )
         view_firstLastCellParticle[ view_particleCellIndex[ i ] ][ 1 ] = i;
   };
   Algorithms::parallelFor< DeviceType >( 0 + 1, numberOfParticles - 1, init );

   //resolve last partile
   const PairIndexType lastActiveCellContains =
      view_firstLastCellParticle.getElement( view_particleCellIndex.getElement( numberOfParticles - 1 ) );
   view_firstLastCellParticle.setElement( view_particleCellIndex.getElement( numberOfParticles - 1 ),
                                          { ( view_particleCellIndex.getElement( numberOfParticles - 1 )
                                              != view_particleCellIndex.getElement( numberOfParticles - 2 ) )
                                               ? ( numberOfParticles - 1 )
                                               : lastActiveCellContains[ 0 ],
                                            ( numberOfParticles - 1 ) } );
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
   this->reorderParticles();
   // update number of particles - removed particles with invalid positions are shifted at the end of the array
   if( this->getNumberOfParticlesToRemove() != 0 ) {
      this->setNumberOfParticles( this->getNumberOfParticles() - this->getNumberOfParticlesToRemove() );
      this->setNumberOfParticlesToRemove( 0 );
   }
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
