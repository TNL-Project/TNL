
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
#include <TNL/Algorithms/ParallelForND.h>

#ifndef DOXYGEN_ONLY

namespace TNL::Meshes {

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

   // dimensions must be set after proportions
   setDimensions( dimensions );
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
   static_assert( EntityOrientationIdx >= 0
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
Grid< Dimension, Real, Device, Index >::getBasis( Index totalOrientationIdx ) const noexcept
{
   return 1 - this->getNormals( totalOrientationIdx );
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
Grid< Dimension, Real, Device, Index >::traverseAll( const CoordinatesType& from,
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
Grid< Dimension, Real, Device, Index >::traverseInterior( const CoordinatesType& from,
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
Grid< Dimension, Real, Device, Index >::traverseBoundary( const CoordinatesType& from,
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
      return ( entity.getCoordinates(), this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] );
   }
   else if constexpr( Entity::getEntityDimension() == 0 ) {
      return ( entity.getCoordinates(), this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] );
   }
   else {
      return ( entity.getCoordinates(), this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] )
           + this->entitiesIndexesOffsets[ entity.getTotalOrientationIndex() + Entity::getEntityDimension() ];
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename EntityType_ >
__cuda_callable__
EntityType_
Grid< Dimension, Real, Device, Index >::getEntity( const CoordinatesType& coordinates ) const
{
   static_assert( EntityType_::getEntityDimension() <= getMeshDimension(),
                  "Entity dimension must be lower or equal to grid dimension." );
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
   static_assert( EntityType_::getEntityDimension() <= getMeshDimension(),
                  "Entity dimension must be lower or equal to grid dimension." );
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
   return ( offset, this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ] ) + entity.getIndex();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int OtherEntityDimension, typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityIndex( const Entity& entity,
                                                        const CoordinatesType& offset,
                                                        Index otherEntityOrientationIdx ) const -> Index
{
   const IndexType totalOrientationIndex =
      EntitiesOrientations::getTotalOrientationIndex( OtherEntityDimension, otherEntityOrientationIdx );
   if constexpr( OtherEntityDimension == getMeshDimension() || OtherEntityDimension == 0 ) {
      return ( entity.getCoordinates() + offset, this->coordinatesMultiplicators[ totalOrientationIndex ] );
   }
   else
      return ( entity.getCoordinates() + offset, this->coordinatesMultiplicators[ totalOrientationIndex ] )
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
Grid< Dimension, Real, Device, Index >::getEntity( const Entity& entity,
                                                   const CoordinatesType& offset,
                                                   const NormalsType& otherEntityOrientation ) const
   -> EntityType< OtherEntityDimension >
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
   return entity.getIndex() + Step * this->coordinatesMultiplicators[ entity.getTotalOrientationIndex() ][ Direction ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
void
Grid< Dimension, Real, Device, Index >::getAdjacentCells( const Entity& entity, IndexType& closer, IndexType& remoter ) const
{
   static_assert( Entity::getEntityDimension() == Dimension - 1, "This method works only for faces." );
   remoter = ( entity.getCoordinates(), this->coordinatesMultiplicators[ getTotalOrientationsCount() - 1 ] );
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
   static_assert( Entity::getEntityDimension() < SuperentitiesDimension,
                  "The superentities dimension must be higher the the entity dimension." );
   IndexType i( 0 );
   // TODO: implement
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
void
Grid< Dimension, Real, Device, Index >::getAdjacentFacesIndexes( const Entity& entity,
                                                                 CoordinatesType& closer,
                                                                 CoordinatesType& remoter ) const
{
   constexpr IndexType begin = EntitiesOrientations::template getTotalOrientationIndex< Dimension - 1, 0 >();
   constexpr IndexType end = begin + Dimension;
   Algorithms::staticFor< int, begin, end >(
      [ & ]( IndexType totalOrientationIndex ) mutable
      {
         const IndexType i = totalOrientationIndex - begin;
         closer[ i ] = ( entity.getCoordinates(), this->coordinatesMultiplicators[ totalOrientationIndex ] )
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
      return product( this->getSpaceSteps() * entity.getBasis() );
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
   this->forEntities< EntityDimension >( CoordinatesType( 0 ), this->getSizes(), function, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forEntities( const CoordinatesType& begin_,
                                                     const CoordinatesType& end_,
                                                     Func function,
                                                     FuncArgs... args ) const
{
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == getMeshDimension() || EntityDimension == 0 ) {
      auto exec = [ = ] __cuda_callable__( GridEntityType & entity, const Grid& grid, FuncArgs... args ) mutable
      {
         entity.setGrid( grid );
         entity.refresh();
         function( entity, args... );
      };
      Algorithms::ParallelForND< Device, false >::exec(
         GridEntityType( begin_ ), GridEntityType( end_ + CoordinatesType( EntityDimension == 0 ) ), exec, *this, args... );
   }
   else {
      /*auto exec = [ = ] __cuda_callable__( GridEntityType& entity, const Grid& grid,
                                           FuncArgs... args ) mutable
      {
         entity.setGrid( grid );
         TNL_ASSERT_LT( entity.getTotalOrientationIndex(), EntitiesOrientations::getTotalOrientationsCount(), "" );

         if( entity.getCoordinates() < grid.getSizes() + entity.getNormals() ) {
            entity.refresh();
            function( entity, args... );
         }
      };
      const Index orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension>();
      const IndexType totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      const IndexType totalOrientationsEnd = EntitiesOrientations::getTotalOrientationIndex( EntityDimension,
      orientationsCount
      ); GridEntityType begin( begin_ ); GridEntityType end( end_ + CoordinatesType( 1 ) ); begin[ Grid::getMeshDimension() ]
      = totalOrientationsBegin; end[ Grid::getMeshDimension() ] = totalOrientationsEnd; Algorithms::ParallelForND< Device,
      false
      >::exec( begin, end, exec, *this, args... );*/

      const Index orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      const IndexType totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      const IndexType totalOrientationsEnd =
         EntitiesOrientations::getTotalOrientationIndex( EntityDimension, orientationsCount );

      for( IndexType totalOrientationIdx = totalOrientationsBegin; totalOrientationIdx < totalOrientationsEnd;
           totalOrientationIdx++ )
      {
         auto exec = [ = ] __cuda_callable__( GridEntityType & entity, const Grid& grid, FuncArgs... args ) mutable
         {
            entity.setTotalOrientationIndex( totalOrientationIdx );
            entity.setGrid( grid );
            entity.refresh();
            function( entity, args... );
         };
         //const Index orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension>();
         //const IndexType totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
         //const IndexType totalOrientationsEnd = EntitiesOrientations::getTotalOrientationIndex( EntityDimension,
         //orientationsCount );
         GridEntityType begin( begin_ );
         GridEntityType end( end_ + entitiesOrientations.getNormals( totalOrientationIdx ) );
         Algorithms::ParallelForND< Device, false >::exec( begin, end, exec, *this, args... );
      }
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( Func func, FuncArgs... args ) const
{
   this->forBoundaryEntities< EntityDimension >( CoordinatesType( 0 ), this->getSizes(), func, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( const CoordinatesType& begin_,
                                                             const CoordinatesType& end_,
                                                             Func func,
                                                             FuncArgs... args ) const
{
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == getMeshDimension() || EntityDimension == 0 ) {
      auto exec = [ = ] __cuda_callable__( GridEntityType & entity, const Grid& grid, FuncArgs... args ) mutable
      {
         entity.setGrid( grid );
         if( entity.isBoundary() ) {
            entity.refresh();
            func( entity, args... );
         }
      };
      Algorithms::ParallelForND< Device, false >::exec(
         GridEntityType( CoordinatesType( begin_ ) ),
         GridEntityType( CoordinatesType( end_ + CoordinatesType( EntityDimension == 0 ) ) ),
         exec,
         *this,
         args... );
   }
   else {
      /*auto exec = [ = ] __cuda_callable__( GridEntityType& entity, const Grid& grid,
                                           FuncArgs... args ) mutable
      {
         entity.setGrid( grid );

         if( entity.getCoordinates() < end_ + entity.getNormals() && entity.isBoundary() ) {
            entity.refresh();
            func( entity, args... );
         }
      };
      const Index orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension>();
      const IndexType totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      const IndexType totalOrientationsEnd = EntitiesOrientations::getTotalOrientationIndex( EntityDimension,
      orientationsCount
      ); GridEntityType begin( begin_ ); GridEntityType end( end_ + CoordinatesType( 1 ) ); begin[ Grid::getMeshDimension() ]
      = totalOrientationsBegin; end[ Grid::getMeshDimension() ] = totalOrientationsEnd; Algorithms::ParallelForND< Device,
      false
      >::exec( begin, end, exec, *this, args... );*/

      const Index orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      const IndexType totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      const IndexType totalOrientationsEnd =
         EntitiesOrientations::getTotalOrientationIndex( EntityDimension, orientationsCount );

      for( IndexType totalOrientationIdx = totalOrientationsBegin; totalOrientationIdx < totalOrientationsEnd;
           totalOrientationIdx++ )
      {
         auto exec = [ = ] __cuda_callable__( GridEntityType & entity, const Grid& grid, FuncArgs... args ) mutable
         {
            entity.setTotalOrientationIndex( totalOrientationIdx );
            entity.setGrid( grid );
            if( entity.isBoundary() ) {
               entity.refresh();
               func( entity, args... );
            }
         };
         GridEntityType begin( begin_ );
         GridEntityType end( end_ + entitiesOrientations.getNormals( totalOrientationIdx ) );
         Algorithms::ParallelForND< Device, false >::exec( begin, end, exec, *this, args... );
      }
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forInteriorEntities( Func func, FuncArgs... args ) const
{
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == getMeshDimension() || EntityDimension == 0 ) {
      auto exec = [ = ] __cuda_callable__( GridEntityType & entity, const Grid& grid, FuncArgs... args ) mutable
      {
         entity.setGrid( grid );
         entity.refresh();
         func( entity, args... );
      };
      Algorithms::ParallelForND< Device, false >::exec(
         GridEntityType( CoordinatesType( 1 ) ),
         GridEntityType( CoordinatesType( this->getSizes() - CoordinatesType( EntityDimension != 0 ) ) ),
         exec,
         *this,
         args... );
   }
   else {
      /*auto exec = [ = ] __cuda_callable__( GridEntityType& entity, const Grid& grid,
                                           FuncArgs... args ) mutable
      {
         TNL_ASSERT_GE( entity.getOrientationIndex(), 0, "" );
         entity.setGrid( grid );
         bool process_entity( true );
         const NormalsType& normals = entity.getNormals();
         const CoordinatesType& coordinates = entity.getCoordinates();
         Algorithms::staticFor< IndexType, 0, getMeshDimension() >(
         [&] ( Index i ) mutable {
            if( coordinates[ i ] == grid.getSizes()[ i ] || ( normals[ i ] && ( coordinates[ i ] == 0 ) ) )
                process_entity = false;
         } );

         if( process_entity ) {
            entity.refresh();
            func( entity, args... );
         }
      };
      const Index orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension>();
      const IndexType totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      const IndexType totalOrientationsEnd = EntitiesOrientations::getTotalOrientationIndex( EntityDimension,
      orientationsCount
      ); GridEntityType begin( CoordinatesType( 0 ) ); GridEntityType end( CoordinatesType( this->getSizes() ) ); begin[
      Grid::getMeshDimension() ] = totalOrientationsBegin; end[ Grid::getMeshDimension() ] = totalOrientationsEnd;
      Algorithms::ParallelForND< Device, false >::exec( begin, end, exec, *this, args... );*/

      const Index orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      const IndexType totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      const IndexType totalOrientationsEnd =
         EntitiesOrientations::getTotalOrientationIndex( EntityDimension, orientationsCount );

      for( IndexType totalOrientationIdx = totalOrientationsBegin; totalOrientationIdx < totalOrientationsEnd;
           totalOrientationIdx++ )
      {
         auto exec = [ = ] __cuda_callable__( GridEntityType & entity, const Grid& grid, FuncArgs... args ) mutable
         {
            entity.setTotalOrientationIndex( totalOrientationIdx );
            entity.setGrid( grid );
            entity.refresh();
            func( entity, args... );
         };
         GridEntityType begin( entitiesOrientations.getNormals( totalOrientationIdx ) );
         GridEntityType end( CoordinatesType( this->getSizes() ) );
         /*if constexpr( Devices::isCuda< Device >() ) {
#ifdef __CUDACC__
            cudaStream_t stream;
            cudaStreamCreate( &stream );
            Devices::Cuda::LaunchConfiguration launch_config;
            launch_config.stream = stream;
            Algorithms::ParallelForND< Device, false >::exec( begin, end, launch_config, exec, *this, args... );
            cudaStreamDestroy( stream );
#endif
         }
         else {
            Algorithms::ParallelForND< Device, false >::exec( begin, end, exec, *this, args... );
         }*/
         Algorithms::ParallelForND< Device, false >::exec( begin, end, exec, *this, args... );
      }
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forLocalEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
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
Grid< Dimension, Real, Device, Index >::partitionEntities( const Vector& allEntities,
                                                           int entitiesDimension,
                                                           int entitiesOrientationIdx ) const -> typename Vector::ConstViewType
{
   const IndexType totalOrientationIdx =
      EntitiesOrientations::getTotalOrientationIndex( entitiesDimension, entitiesOrientationIdx );
   return allEntities.getConstView( this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension ],
                                    this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension + 1 ] );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Vector >
auto
Grid< Dimension, Real, Device, Index >::partitionEntities( Vector& allEntities,
                                                           int entitiesDimension,
                                                           int entitiesOrientationIdx ) const -> typename Vector::ViewType
{
   const IndexType totalOrientationIdx =
      EntitiesOrientations::getTotalOrientationIndex( entitiesDimension, entitiesOrientationIdx );
   return allEntities.getView( this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension ],
                               this->entitiesIndexesOffsets[ totalOrientationIdx + entitiesDimension + 1 ] );
}

}  // namespace TNL::Meshes

#endif
