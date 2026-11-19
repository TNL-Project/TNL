
// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/String.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/Templates/BooleanOperations.h>
#include <TNL/Meshes/GridDetails/NormalsGetter.h>
#include <TNL/Meshes/GridDetails/Templates/Functions.h>
#include <TNL/Meshes/GridDetails/Templates/ForEachOrientation.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/staticFor.h>

#ifndef DOXYGEN_ONLY

namespace TNL::Meshes {

namespace detail {

/**
 * \brief Kernel of the traversals of the grid entities with the given total orientation index.
 *
 * The orientation is a template parameter, so that it is a compile-time constant in the kernel. With a run-time
 * orientation, the tables of the grid indexed by the orientation (e.g. in \ref Grid::getEntityIndex) are copied
 * to the local memory on the GPU, which made the traversals of faces three times slower.
 */
template< typename GridEntityType, typename Grid, int TotalOrientationIndex, typename Func >
struct OrientedEntitiesKernel
{
   Func func;

   template< typename... FuncArgs >
   __cuda_callable__
   void
   operator()( const typename Grid::CoordinatesType& coordinates, const Grid& grid, FuncArgs... args )
   {
      GridEntityType entity( coordinates );
      entity.setTotalOrientationIndex( TotalOrientationIndex );
      entity.setGrid( grid );
      entity.refresh();
      func( entity, args... );
   }
};

/**
 * \brief Computes the scalar product of the coordinates and the coordinates multiplicators of the grid.
 *
 * The first multiplicator is always one (see \ref Grid::setCoordinatesMultiplicators). The compiler cannot know
 * it, so with the plain scalar product, the step between the neighbouring entities along the first axis is
 * a run-time value and the compilers do not vectorize the loops over the entities (e.g. GCC with `-fno-math-errno`).
 */
template< typename Coordinates, typename Multiplicators >
__cuda_callable__
auto
getLinearIndex( const Coordinates& coordinates, const Multiplicators& multiplicators ) -> typename Coordinates::ValueType
{
   typename Coordinates::ValueType result = coordinates[ 0 ];
   for( int i = 1; i < Coordinates::getSize(); i++ )
      result += coordinates[ i ] * multiplicators[ i ];
   return result;
}

}  // namespace detail

template< int Dimension, typename Real, typename Device, typename Index >
constexpr int
Grid< Dimension, Real, Device, Index >::getMeshDimension()
{
   return Dimension;
}

template< int Dimension, typename Real, typename Device, typename Index >
Grid< Dimension, Real, Device, Index >::Grid( const CoordinatesType& sizes )
{
   setSizes( sizes );

   proportions = 0;
   spaceSteps = 0;
   origin = 0;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto __cuda_callable__
Grid< Dimension, Real, Device, Index >::getEntitiesOrientations() const->const EntitiesOrientations&
{
   return this->entitiesOrientations;
}

template< int Dimension, typename Real, typename Device, typename Index >
constexpr Index
Grid< Dimension, Real, Device, Index >::getEntityOrientationsCount( IndexType entityDimension )
{
   return combinationsCount< Index >( entityDimension, Dimension );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setSizes(
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& sizes )
{
   TNL_ASSERT_GE( sizes, CoordinatesType( 0 ), "Sizes must be positive" );
   this->sizes = sizes;
   setEntitiesIndexesOffsets();
   setCoordinatesMultiplicators();
   fillSpaceSteps();
   this->localBegin = 0;
   this->localEnd = this->getSizes();
   this->interiorBegin = 1;
   this->interiorEnd = this->getSizes() - 1;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::CoordinatesType&
Grid< Dimension, Real, Device, Index >::getSizes() const noexcept
{
   return this->sizes;
}

template< int Dimension, typename Real, typename Device, typename Index >
[[nodiscard]] __cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount( IndexType entityDimension ) const noexcept
{
   TNL_ASSERT_GE( entityDimension, 0, "Entity dimension must be greater than or equal to 0" );
   TNL_ASSERT_LE( entityDimension, Dimension, "Entity dimension must be less than or equal to Dimension" );

   return this->entitiesCounts[ entityDimension ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
[[nodiscard]] __cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount() const noexcept
{
   static_assert( EntityDimension >= 0, "Entity dimension must be greater than or equal to 0" );
   static_assert( EntityDimension <= Dimension, "Entity dimension must be less than or equal to Dimension" );

   return this->entitiesCounts[ EntityDimension ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename EntityType_ >
[[nodiscard]] __cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount() const noexcept
{
   static_assert( EntityType_::getEntityDimension() >= 0, "Entity dimension must be greater than or equal to 0" );
   static_assert( EntityType_::getEntityDimension() <= Dimension, "Entity dimension must be less than or equal to Dimension" );

   return this->entitiesCounts[ EntityType_::getEntityDimension() ];
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntitiesCounts() const noexcept -> const EntitiesCounts&
{
   return this->entitiesCounts;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setOrigin(
   const typename Grid< Dimension, Real, Device, Index >::PointType& origin ) noexcept
{
   this->origin = origin;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientedEntitiesCount( IndexType entityDimension, IndexType orientationIndex ) const
{
   TNL_ASSERT_GE( entityDimension, 0, "dimension must be greater than or equal to 0" );
   TNL_ASSERT_LE( entityDimension, Dimension, "dimension must be less than or equal to Dimension" );

   if( entityDimension == 0 || entityDimension == Dimension )
      return this->getEntitiesCount( entityDimension );

   const Index index = EntitiesOrientations::getTotalOrientationIndex( entityDimension, orientationIndex ) + entityDimension;
   return this->entitiesIndexesOffsets[ index + 1 ] - this->entitiesIndexesOffsets[ index ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, int EntityOrientationIdx >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientedEntitiesCount() const noexcept
{
   static_assert( EntityDimension >= 0 && EntityDimension <= Dimension, "Wrong entity dimension." );
   static_assert(
      EntityOrientationIdx >= 0
         && EntityOrientationIdx <= EntitiesOrientations::template getOrientationsCount< EntityDimension >(),
      "Wrong entity orientation index." );
   if constexpr( EntityDimension == 0 || EntityDimension == Dimension )
      return this->getEntitiesCount( EntityDimension );

   const Index index =
      EntitiesOrientations::getTotalOrientationIndex( EntityDimension, EntityOrientationIdx ) + EntityDimension;
   return this->entitiesIndexesOffsets[ index + 1 ] - this->entitiesIndexesOffsets[ index ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getNormals( Index orientationIdx ) const noexcept -> NormalsType
{
   return this->entitiesOrientations.template getNormals< EntityDimension >( orientationIdx );
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getNormals( Index totalOrientationIdx ) const noexcept -> NormalsType
{
   return this->entitiesOrientations.getNormals( totalOrientationIdx );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
typename Grid< Dimension, Real, Device, Index >::CoordinatesType
Grid< Dimension, Real, Device, Index >::getBasis( Index orientationIndex ) const noexcept
{
   return 1 - this->template getNormals< EntityDimension >( orientationIndex );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
[[nodiscard]] __cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientationIndex( const NormalsType& normals ) const noexcept
{
   return entitiesOrientations.getOrientationIndex( normals );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityCoordinates( IndexType entityIdx, IndexType& totalOrientationIndex )
   const noexcept -> CoordinatesType
{
   if constexpr( EntityDimension != 0 && EntityDimension != getMeshDimension() ) {
      IndexType i = EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( 0 ) + EntityDimension + 1;
      const Index end = i + this->getEntityOrientationsCount( EntityDimension ) + EntityDimension;
      IndexType orientationIdx = 0;
      while( i < end && entityIdx >= this->entitiesIndexesOffsets[ i ] ) {
         i++;
         orientationIdx++;
      }
      entityIdx -= this->entitiesIndexesOffsets[ i - 1 ];
      totalOrientationIndex = EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >(
         orientationIdx );  // TODO: compute directly total orientation index
   }
   else
      totalOrientationIndex = EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( 0 );

   const CoordinatesType dims = this->getSizes() + getNormals( totalOrientationIndex );
   CoordinatesType entityCoordinates( 0 );
   int idx = 0;
   while( idx < getMeshDimension() - 1 ) {
      entityCoordinates[ idx ] = entityIdx % dims[ idx ];
      entityIdx /= dims[ idx++ ];
   }
   entityCoordinates[ idx ] = entityIdx % dims[ idx ];
   return entityCoordinates;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::PointType&
Grid< Dimension, Real, Device, Index >::getOrigin() const noexcept
{
   return this->origin;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setDomain(
   const typename Grid< Dimension, Real, Device, Index >::PointType& origin,
   const typename Grid< Dimension, Real, Device, Index >::PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;

   this->fillSpaceSteps();
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setSpaceSteps(
   const typename Grid< Dimension, Real, Device, Index >::PointType& spaceSteps ) noexcept
{
   this->spaceSteps = spaceSteps;
   fillProportions();
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::PointType&
Grid< Dimension, Real, Device, Index >::getSpaceSteps() const noexcept
{
   return this->spaceSteps;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::PointType&
Grid< Dimension, Real, Device, Index >::getProportions() const noexcept
{
   return this->proportions;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseAll( Func func, FuncArgs... args ) const
{
   this->traverseAll< EntityDimension >( CoordinatesType( 0 ), this->getSizes(), func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseAll(
   const CoordinatesType& from,
   const CoordinatesType& to,
   Func func,
   FuncArgs... args ) const
{
   TNL_ASSERT_ALL_GE( from, 0, "Traverse rect must be in the grid dimensions" );
   TNL_ASSERT_ALL_LE( to, this->getSizes(), "Traverse rect be in the grid dimensions" );
   TNL_ASSERT_ALL_LE( from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor" );

   if constexpr( EntityDimension == getMeshDimension() ) {
      Algorithms::parallelFor< Device >( from, to, func, args... );
   }
   else {
      auto exec = [ & ]( const Index orientation, const NormalsType& normals )
      {
         TNL_ASSERT_EQ( orientation, this->getOrientationIndex( normals ), "Wrong index of entity orientation." );
         Algorithms::parallelFor< Device >( from, to + normals, func, normals, orientation, args... );
      };
      Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseInterior( Func func, FuncArgs... args ) const
{
   this->traverseInterior< EntityDimension >( CoordinatesType( 0 ), this->getSizes(), func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseInterior(
   const CoordinatesType& from,
   const CoordinatesType& to,
   Func func,
   FuncArgs... args ) const
{
   TNL_ASSERT_ALL_GE( from, 0, "Traverse rect must be in the grid dimensions" );
   TNL_ASSERT_ALL_LE( to, this->getSizes(), "Traverse rect be in the grid dimensions" );
   TNL_ASSERT_ALL_LE( from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor" );

   auto exec = [ & ]( const Index orientation, const NormalsType& normals )
   {
      switch( EntityDimension ) {
         case 0:
            {
               const CoordinatesType begin = from + CoordinatesType( 1 );
               Algorithms::parallelFor< Device >( begin, to, func, normals, orientation, args... );
               break;
            }
         case Dimension:
            {
               const CoordinatesType begin = from + CoordinatesType( 1 );
               const CoordinatesType end = to - CoordinatesType( 1 );
               Algorithms::parallelFor< Device >( begin, end, func, normals, orientation, args... );
               break;
            }
         default:
            {
               const CoordinatesType begin = from + normals;
               Algorithms::parallelFor< Device >( begin, to, func, normals, orientation, args... );
               break;
            }
      }
   };

   Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseBoundary( Func func, FuncArgs... args ) const
{
   this->traverseBoundary< EntityDimension >( CoordinatesType( 0 ), this->getSizes(), func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseBoundary(
   const CoordinatesType& from,
   const CoordinatesType& to,
   Func func,
   FuncArgs... args ) const
{
   // Boundaries of the grid are formed by the entities of Dimension - 1.
   // We need to traverse each orientation independently.
   constexpr int orientationsCount = getEntityOrientationsCount( Dimension - 1 );
   constexpr bool isDirectedEntity = EntityDimension != 0 && EntityDimension != Dimension;
   constexpr bool isAnyBoundaryIntersects = EntityDimension != Dimension - 1;

   Containers::StaticVector< orientationsCount, Index > isBoundaryTraversed( 0 );

   auto forBoundary = [ & ]( const auto orthogonalOrientation, const auto orientation, const NormalsType& normals )
   {
      CoordinatesType start = from;
      CoordinatesType end = to + normals;

      if( isAnyBoundaryIntersects ) {
         for( Index i = 0; i < Dimension; i++ ) {
            start[ i ] = ( ! isDirectedEntity || normals[ i ] ) && isBoundaryTraversed[ i ] ? 1 : 0;
            end[ i ] = end[ i ] - ( ( ! isDirectedEntity || normals[ i ] ) && isBoundaryTraversed[ i ] ? 1 : 0 );
         }
      }

      start[ orthogonalOrientation ] = end[ orthogonalOrientation ] - 1;

      Algorithms::parallelFor< Device >( start, end, func, normals, orientation, args... );

      // Skip entities defined only once
      if( ! start[ orthogonalOrientation ] && end[ orthogonalOrientation ] )
         return;

      start[ orthogonalOrientation ] = 0;
      end[ orthogonalOrientation ] = 1;

      Algorithms::parallelFor< Device >( start, end, func, normals, orientation, args... );
   };

   if( ! isAnyBoundaryIntersects ) {
      auto exec = [ & ]( const auto orientation, const NormalsType& normals )
      {
         constexpr int orthogonalOrientation = EntityDimension - orientation;

         forBoundary( orthogonalOrientation, orientation, normals );
      };

      Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
      return;
   }

   auto exec = [ & ]( const auto orthogonalOrientation )
   {
      auto exec = [ & ]( const auto orientation, const NormalsType& normals )
      {
         forBoundary( orthogonalOrientation, orientation, normals );
      };

      if( EntityDimension == 0 || EntityDimension == Dimension ) {
         Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
      }
      else {
         Templates::ForEachOrientation< Index, EntityDimension, Dimension, orthogonalOrientation >::exec( exec );
      }

      isBoundaryTraversed[ orthogonalOrientation ] = 1;
   };

   Algorithms::staticFor< int, 0, orientationsCount >( exec );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::getEntityDimension() <= Dimension && Entity::getEntityDimension() >= 0, "Wrong grid entity sizes." );
   TNL_ASSERT_GE( entity.getTotalOrientationIndex(), 0, "Wrong total orientation index." );
   TNL_ASSERT_LT(
      entity.getTotalOrientationIndex(), EntitiesOrientations::getTotalOrientationsCount(), "Wrong total orientation index." );
   TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "Wrong entity coordinates" );
   TNL_ASSERT_LT( entity.getCoordinates(), this->getSizes() + entity.getNormals(), "Wrong entity coordinates" );

   if constexpr( Entity::getEntityDimension() == Dimension ) {
      return detail::getLinearIndex(
         entity.getCoordinates(), this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] );
   }
   else if constexpr( Entity::getEntityDimension() == 0 ) {
      return detail::getLinearIndex(
         entity.getCoordinates(), this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] );
   }
   else {
      return detail::getLinearIndex(
                entity.getCoordinates(), this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] )
           + this->entitiesIndexesOffsets[ entity.getTotalOrientationIndex() + Entity::getEntityDimension() ];
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename EntityType_ >
__cuda_callable__
EntityType_
Grid< Dimension, Real, Device, Index >::getEntity( const CoordinatesType& coordinates ) const
{
   static_assert(
      EntityType_::getEntityDimension() <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType_( *this, coordinates );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntity( const CoordinatesType& coordinates ) const -> EntityType< EntityDimension >
{
   static_assert( EntityDimension <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType< EntityDimension >( *this, coordinates );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename EntityType_ >
__cuda_callable__
EntityType_
Grid< Dimension, Real, Device, Index >::getEntity( IndexType entityIdx ) const
{
   static_assert(
      EntityType_::getEntityDimension() <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType_( *this, entityIdx );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntity( IndexType entityIdx ) const -> EntityType< EntityDimension >
{
   static_assert( EntityDimension <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType< EntityDimension >( *this, entityIdx );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityIndex( const Entity& entity, const CoordinatesType& offset ) const -> Index
{
   return detail::getLinearIndex( offset, this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] )
        + entity.getIndex();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int OtherEntityDimension, typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityIndex(
   const Entity& entity,
   const CoordinatesType& offset,
   Index otherEntityOrientationIdx ) const -> Index
{
   const IndexType totalOrientationIndex =
      EntitiesOrientations::template getTotalOrientationIndex< OtherEntityDimension >( otherEntityOrientationIdx );
   if constexpr( OtherEntityDimension == getMeshDimension() || OtherEntityDimension == 0 ) {
      return detail::getLinearIndex(
         CoordinatesType( entity.getCoordinates() + offset ), this->coordinatesMultiplicators[ totalOrientationIndex ] );
   }
   else
      return detail::getLinearIndex(
                CoordinatesType( entity.getCoordinates() + offset ), this->coordinatesMultiplicators[ totalOrientationIndex ] )
           + this->entitiesIndexesOffsets[ totalOrientationIndex + OtherEntityDimension ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
Entity
Grid< Dimension, Real, Device, Index >::getEntity( const Entity& entity, const CoordinatesType& offset ) const
{
   return Entity( *this, entity.getCoordinates() + offset, entity.getOrientationIndex() );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int OtherEntityDimension, typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntity(
   const Entity& entity,
   const CoordinatesType& offset,
   const NormalsType& otherEntityOrientation ) const -> EntityType< OtherEntityDimension >
{
   return EntityType< OtherEntityDimension >(
      *this, CoordinatesType( entity.getCoordinates() + offset ), otherEntityOrientation );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int Direction, int Step, typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getAdjacentEntityIndex( const Entity& entity ) const -> IndexType
{
   // the first coordinates multiplicator is always one, see detail::getLinearIndex
   if constexpr( Direction == 0 )
      return entity.getIndex() + Step;
   else
      return entity.getIndex() + Step * this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ][ Direction ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
void
Grid< Dimension, Real, Device, Index >::getAdjacentCells( const Entity& entity, IndexType& closer, IndexType& remoter ) const
{
   static_assert( Entity::getEntityDimension() == Dimension - 1, "This method works only for faces." );
   remoter =
      detail::getLinearIndex( entity.getCoordinates(), this->coordinatesMultiplicators[ getTotalOrientationsCount() - 1 ] );
   TNL_ASSERT_LT( entity.getOrientationIndex(), Dimension, "" );
   closer = remoter - this->coordinatesMultiplicators[ getTotalOrientationsCount() - 1 ][ entity.getOrientationIndex() ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int SuperentitiesDimension, typename Entity >
__cuda_callable__
void
Grid< Dimension, Real, Device, Index >::getSuperentitiesIndexes(
   const Entity& entity,
   SuperentitiesContainer< SuperentitiesDimension, Entity::getDimension() >& closer,
   SuperentitiesContainer< SuperentitiesDimension, Entity::getDimension() >& remoter ) const
{
   static_assert(
      Entity::getEntityDimension() < SuperentitiesDimension,
      "The superentities dimension must be higher the the entity dimension." );
   IndexType i( 0 );
   // TODO: implement
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
void
Grid< Dimension, Real, Device, Index >::getAdjacentFacesIndexes(
   const Entity& entity,
   CoordinatesType& closer,
   CoordinatesType& remoter ) const
{
   constexpr IndexType begin = EntitiesOrientations::template getTotalOrientationIndex< Dimension - 1, 0 >();
   constexpr IndexType end = begin + Dimension;
   Algorithms::staticFor< int, begin, end >(
      [ & ]( IndexType totalOrientationIndex ) mutable
      {
         const IndexType i = totalOrientationIndex - begin;
         closer[ i ] =
            detail::getLinearIndex( entity.getCoordinates(), this->coordinatesMultiplicators[ totalOrientationIndex ] )
            + this->entitiesIndexesOffsets[ totalOrientationIndex + Dimension - 1 ];
         remoter[ i ] = closer[ i ] + this->coordinatesMultiplicators[ totalOrientationIndex ][ i ];
      } );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityOrigin( const Entity& entity ) const -> PointType
{
   return this->getOrigin() + entity.getCoordinates() * this->getSpaceSteps();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityCenter( const Entity& entity ) const -> PointType
{
   return this->getOrigin() + ( entity.getCoordinates() + 0.5 * entity.getBasis() ) * this->getSpaceSteps();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getEntityMeasure( const Entity& entity ) const
{
   if constexpr( Entity::getEntityDimension() != 0 ) {
      // the measure is the product of the space steps along the axes which the entity spans
      const auto basis = entity.getBasis();
      Real measure = 1;
      for( int i = 0; i < Dimension; i++ )
         if( basis[ i ] )
            measure *= this->getSpaceSteps()[ i ];
      return measure;
   }
   return 0.0;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getCellMeasure() const
{
   return product( this->getSpaceSteps() );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
bool
Grid< Dimension, Real, Device, Index >::isBoundaryEntity( const Entity& entity ) const
{
   bool result( false );
   const auto& coordinates = entity.getCoordinates();
   if constexpr( Entity::getEntityDimension() == Dimension ) {
      Algorithms::staticFor< IndexType, 0, Dimension >(
         [ & ]( Index i ) mutable
         {
            if( coordinates[ i ] == 0 || coordinates[ i ] == this->getSizes()[ i ] - 1 )
               result = true;  // return true does not work here
         } );
   }
   else if constexpr( Entity::getEntityDimension() == 0 ) {
      Algorithms::staticFor< IndexType, 0, Dimension >(
         [ & ]( Index i ) mutable
         {
            if( coordinates[ i ] == 0 || coordinates[ i ] == ( this->getSizes()[ i ] ) )
               result = true;  // return true does not work here
         } );
   }
   else {
      const auto& normals = entity.getNormals();
      Algorithms::staticFor< IndexType, 0, Dimension >(
         [ & ]( Index i ) mutable
         {
            if( normals[ i ] && ( coordinates[ i ] == 0 || coordinates[ i ] == this->getSizes()[ i ] ) )
               result = true;  // return true does not work here
         } );
   }
   return result;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setLocalSubdomain( const CoordinatesType& begin, const CoordinatesType& end )
{
   this->localBegin = begin;
   this->localEnd = end;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setLocalBegin( const CoordinatesType& begin )
{
   this->localBegin = begin;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setLocalEnd( const CoordinatesType& end )
{
   this->localEnd = end;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getLocalBegin() const -> const CoordinatesType&
{
   return this->localBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getLocalEnd() const -> const CoordinatesType&
{
   return this->localEnd;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setInteriorBegin( const CoordinatesType& begin )
{
   this->interiorBegin = begin;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setInteriorEnd( const CoordinatesType& end )
{
   this->interiorEnd = end;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getInteriorBegin() const -> const CoordinatesType&
{
   return this->interiorBegin;
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
Grid< Dimension, Real, Device, Index >::getInteriorEnd() const -> const CoordinatesType&
{
   return this->interiorEnd;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::writeProlog( TNL::Logger& logger ) const noexcept
{
   logger.writeParameter( "Sizes:", this->sizes );

   logger.writeParameter( "Origin:", this->origin );
   logger.writeParameter( "Proportions:", this->proportions );
   logger.writeParameter( "Space steps:", this->spaceSteps );

   TNL::Algorithms::staticFor< IndexType, 0, Dimension + 1 >(
      [ & ]( auto entityDim )
      {
         for( IndexType entityOrientationIdx = 0; entityOrientationIdx < this->getEntityOrientationsCount( entityDim() );
              entityOrientationIdx++ )
         {
            auto normals = this->getBasis< entityDim >( entityOrientationIdx );
            TNL::String tmp = TNL::String( "Entities count with basis " ) + TNL::convertToString( normals ) + ":";
            logger.writeParameter( tmp, this->getOrientedEntitiesCount( entityDim, entityOrientationIdx ) );
         }
      } );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setEntitiesIndexesOffsets()
{
   if( getSizes() == 0 ) {
      this->entitiesIndexesOffsets = 0;
      this->entitiesCounts = 0;
      return;
   }

   Index totalOrientationIdx = 0;
   for( int entityDimension = 0; entityDimension <= Dimension; entityDimension++ ) {
      this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension ] = 0;
      for( Index entityOrientationIdx = 0; entityOrientationIdx < EntitiesOrientations::getOrientationsCount( entityDimension );
           entityOrientationIdx++, totalOrientationIdx++ )
      {
         const auto& normals = entitiesOrientations.getNormals( totalOrientationIdx );
         IndexType entitiesCount = 1;
         Algorithms::staticFor< IndexType, 0, Dimension >(
            [ & ]( Index i ) mutable
            {
               entitiesCount *= this->getSizes()[ i ] + normals[ i ];
            } );
         this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension + 1 ] =
            this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension ] + entitiesCount;
      }
      this->entitiesCounts[ entityDimension ] = this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension ];
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setCoordinatesMultiplicators()
{
   for( IndexType totalOrientationIndex = 0; totalOrientationIndex < getTotalOrientationsCount(); totalOrientationIndex++ ) {
      auto& multiplicators = this->coordinatesMultiplicators[ totalOrientationIndex ];
      const auto& normals = getNormals( totalOrientationIndex );
      multiplicators[ 0 ] = 1;
      for( IndexType i = 0; i < getMeshDimension() - 1; i++ )
         multiplicators[ i + 1 ] = multiplicators[ i ] * ( this->getSizes()[ i ] + normals[ i ] );
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillProportions()
{
   this->proportions = this->spaceSteps * this->sizes;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillSpaceSteps()
{
   bool hasAnyInvalidDimension = false;

   for( Index i = 0; i < Dimension; i++ ) {
      if( this->sizes[ i ] <= 0 ) {
         hasAnyInvalidDimension = true;
         break;
      }
   }

   if( ! hasAnyInvalidDimension ) {
      for( Index i = 0; i < Dimension; i++ )
         this->spaceSteps[ i ] = this->proportions[ i ] / this->sizes[ i ];
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forAllEntities( Func function, FuncArgs... args ) const
{
   this->forAllEntities< EntityDimension >( typename Device::LaunchConfiguration{}, function, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forAllEntities(
   typename Device::LaunchConfiguration launchConfiguration,
   Func function,
   FuncArgs... args ) const
{
   this->forEntities< EntityDimension >( CoordinatesType( 0 ), this->getSizes(), launchConfiguration, function, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forEntities(
   const CoordinatesType& begin,
   const CoordinatesType& end,
   Func function,
   FuncArgs... args ) const
{
   this->forEntities< EntityDimension >( begin, end, typename Device::LaunchConfiguration{}, function, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forEntities(
   const CoordinatesType& begin_,
   const CoordinatesType& end_,
   typename Device::LaunchConfiguration launchConfiguration,
   Func function,
   FuncArgs... args ) const
{
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == getMeshDimension() || EntityDimension == 0 ) {
      auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinates, const Grid& grid, FuncArgs... args ) mutable
      {
         GridEntityType entity( grid, coordinates );
         function( entity, args... );
      };
      Algorithms::parallelFor< Device >(
         begin_, CoordinatesType( end_ + CoordinatesType( EntityDimension == 0 ) ), launchConfiguration, exec, *this, args... );
   }
   else {
      constexpr int orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      constexpr int totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );

      // the orientations are traversed at compile time, see detail::OrientedEntitiesKernel
      Algorithms::staticFor< int, 0, orientationsCount >(
         [ & ]( auto orientation )
         {
            constexpr int totalOrientationIdx = totalOrientationsBegin + decltype( orientation )::value;
            const detail::OrientedEntitiesKernel< GridEntityType, Grid, totalOrientationIdx, Func > exec{ function };
            const CoordinatesType end = end_ + entitiesOrientations.getNormals( totalOrientationIdx );
            Algorithms::parallelFor< Device >( begin_, end, launchConfiguration, exec, *this, args... );
         } );
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( Func func, FuncArgs... args ) const
{
   this->forBoundaryEntities< EntityDimension >( typename Device::LaunchConfiguration{}, func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities(
   typename Device::LaunchConfiguration launchConfiguration,
   Func func,
   FuncArgs... args ) const
{
   this->forBoundaryEntities< EntityDimension >( CoordinatesType( 0 ), this->getSizes(), launchConfiguration, func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities(
   const CoordinatesType& begin,
   const CoordinatesType& end,
   Func func,
   FuncArgs... args ) const
{
   this->forBoundaryEntities< EntityDimension >( begin, end, typename Device::LaunchConfiguration{}, func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities(
   const CoordinatesType& begin_,
   const CoordinatesType& end_,
   typename Device::LaunchConfiguration launchConfiguration,
   Func func,
   FuncArgs... args ) const
{
   // The boundary of the region is decomposed into disjoint layers, so that only the boundary entities are
   // traversed. For each orientation, the entities lie in the box [ begin, end + normals ). An entity is on
   // the boundary if its coordinate along some boundary axis is the first or the last one in the box. The
   // boundary axes are all axes for cells and the axes with nonzero normals for the other entities, which
   // agrees with isBoundaryEntity. The layers of the k-th boundary axis exclude the first and the last
   // positions along the previous boundary axes, which were already traversed.
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   constexpr int orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
   constexpr int totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );

   // the orientations are traversed at compile time, see detail::OrientedEntitiesKernel
   Algorithms::staticFor< int, 0, orientationsCount >(
      [ & ]( auto orientation )
      {
         constexpr int totalOrientationIdx = totalOrientationsBegin + decltype( orientation )::value;
         const detail::OrientedEntitiesKernel< GridEntityType, Grid, totalOrientationIdx, Func > exec{ func };
         auto traverseLayer = [ & ]( const CoordinatesType& layerBegin, const CoordinatesType& layerEnd )
         {
            for( int i = 0; i < Dimension; i++ )
               if( layerEnd[ i ] <= layerBegin[ i ] )
                  return;
            Algorithms::parallelFor< Device >( layerBegin, layerEnd, launchConfiguration, exec, *this, args... );
         };

         const NormalsType normals = entitiesOrientations.getNormals( totalOrientationIdx );
         const CoordinatesType regionBegin = begin_;
         const CoordinatesType regionEnd = end_ + normals;
         CoordinatesType rangeBegin = regionBegin;
         CoordinatesType rangeEnd = regionEnd;
         for( int axis = 0; axis < Dimension; axis++ ) {
            if( EntityDimension != Dimension && ! normals[ axis ] )
               continue;
            if( regionEnd[ axis ] <= regionBegin[ axis ] )
               break;

            CoordinatesType layerBegin = rangeBegin;
            CoordinatesType layerEnd = rangeEnd;
            layerBegin[ axis ] = regionBegin[ axis ];
            layerEnd[ axis ] = regionBegin[ axis ] + 1;
            traverseLayer( layerBegin, layerEnd );

            if( regionEnd[ axis ] - 1 > regionBegin[ axis ] ) {
               layerBegin[ axis ] = regionEnd[ axis ] - 1;
               layerEnd[ axis ] = regionEnd[ axis ];
               traverseLayer( layerBegin, layerEnd );
            }

            rangeBegin[ axis ] = regionBegin[ axis ] + 1;
            rangeEnd[ axis ] = regionEnd[ axis ] - 1;
         }
      } );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forInteriorEntities( Func func, FuncArgs... args ) const
{
   this->forInteriorEntities< EntityDimension >( typename Device::LaunchConfiguration{}, func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forInteriorEntities(
   typename Device::LaunchConfiguration launchConfiguration,
   Func func,
   FuncArgs... args ) const
{
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == getMeshDimension() || EntityDimension == 0 ) {
      auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinates, const Grid& grid, FuncArgs... args ) mutable
      {
         GridEntityType entity( grid, coordinates );
         func( entity, args... );
      };
      Algorithms::parallelFor< Device >(
         CoordinatesType( 1 ),
         CoordinatesType( this->getSizes() - CoordinatesType( EntityDimension != 0 ) ),
         launchConfiguration,
         exec,
         *this,
         args... );
   }
   else {
      constexpr int orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      constexpr int totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );

      // the orientations are traversed at compile time, see detail::OrientedEntitiesKernel
      Algorithms::staticFor< int, 0, orientationsCount >(
         [ & ]( auto orientation )
         {
            constexpr int totalOrientationIdx = totalOrientationsBegin + decltype( orientation )::value;
            const detail::OrientedEntitiesKernel< GridEntityType, Grid, totalOrientationIdx, Func > exec{ func };
            const CoordinatesType begin = CoordinatesType( 0 ) + entitiesOrientations.getNormals( totalOrientationIdx );
            Algorithms::parallelFor< Device >( begin, this->getSizes(), launchConfiguration, exec, *this, args... );
         } );
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forLocalEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__(
                  const CoordinatesType& coordinate,
                  const NormalsType& normals,
                  const Index orientationIdx,
                  const Grid& grid,
                  FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, orientationIdx );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( this->localBegin, this->localEnd, exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Vector >
auto
Grid< Dimension, Real, Device, Index >::partitionEntities(
   const Vector& allEntities,
   int entitiesDimension,
   int entitiesOrientationIdx ) const -> typename Vector::ConstViewType
{
   const IndexType totalOrientationIdx =
      EntitiesOrientations::getTotalOrientationIndex( entitiesDimension, entitiesOrientationIdx );
   return allEntities.getConstView(
      this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension ],
      this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension + 1 ] );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Vector >
auto
Grid< Dimension, Real, Device, Index >::partitionEntities(
   Vector& allEntities,
   int entitiesDimension,
   int entitiesOrientationIdx ) const -> typename Vector::ViewType
{
   const IndexType totalOrientationIdx =
      EntitiesOrientations::getTotalOrientationIndex( entitiesDimension, entitiesOrientationIdx );
   return allEntities.getView(
      this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension ],
      this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension + 1 ] );
}

}  // namespace TNL::Meshes

#endif
