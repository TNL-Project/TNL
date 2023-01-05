
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
template< typename... Dimensions,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool >,
          std::enable_if_t< sizeof...( Dimensions ) == Dimension, bool > >
Grid< Dimension, Real, Device, Index >::Grid( Dimensions... dimensions )
{
   proportions = 0;
   spaceSteps = 0;
   origin = 0;

   // dimensions must be set after proportions
   setDimensions( dimensions... );
}

template< int Dimension, typename Real, typename Device, typename Index >
Grid< Dimension, Real, Device, Index >::Grid( const CoordinatesType& dimensions )
{
   proportions = 0;
   spaceSteps = 0;
   origin = 0;

   // dimensions must be set after proportions
   setDimensions( dimensions );
}

template< int Dimension, typename Real, typename Device, typename Index >
auto
__cuda_callable__
Grid< Dimension, Real, Device, Index >::
getEntitiesOrientations() const -> const EntitiesOrientations&
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
template< typename... Dimensions,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool >,
          std::enable_if_t< sizeof...( Dimensions ) == Dimension, bool > >
void
Grid< Dimension, Real, Device, Index >::setDimensions( Dimensions... dimensions )
{
   this->dimensions = CoordinatesType( dimensions... );

   TNL_ASSERT_ALL_GE( this->dimensions, 0, "Dimension must be positive" );

   setEntitiesIndexesOffsets();
   fillSpaceSteps();
   this->localBegin = 0;
   this->localEnd = this->getDimensions();
   this->interiorBegin = 1;
   this->interiorEnd = this->getDimensions() - 1;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setDimensions(
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& dimensions )
{
   TNL_ASSERT_GE( this->dimensions, CoordinatesType( 0 ), "Dimension must be positive" );
   this->dimensions = dimensions;
   setEntitiesIndexesOffsets();
   fillSpaceSteps();
   this->localBegin = 0;
   this->localEnd = this->getDimensions();
   this->interiorBegin = 1;
   this->interiorEnd = this->getDimensions() - 1;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension, Real, Device, Index >::CoordinatesType&
Grid< Dimension, Real, Device, Index >::getDimensions() const noexcept
{
   return this->dimensions;
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount( IndexType entityDimension ) const
{
   TNL_ASSERT_GE( entityDimension, 0, "Entity dimension must be greater than or equal to 0" );
   TNL_ASSERT_LE( entityDimension, Dimension, "Entity dimension must be less than or equal to Dimension" );

   return this->entitiesCounts[ entityDimension ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount() const noexcept
{
   static_assert( EntityDimension >= 0, "Entity dimension must be greater than or equal to 0" );
   static_assert( EntityDimension <= Dimension, "Entity dimension must be less than or equal to Dimension" );

   return this->entitiesCounts[ EntityDimension ];
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename EntityType_ >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getEntitiesCount() const
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
template< typename... Coordinates,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Coordinates >... >, bool >,
          std::enable_if_t< sizeof...( Coordinates ) == Dimension, bool > >
void
Grid< Dimension, Real, Device, Index >::setOrigin( Coordinates... coordinates ) noexcept
{
   this->origin = PointType( coordinates... );
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientedEntitiesCount( IndexType entityDimension, IndexType orientation ) const
{
   TNL_ASSERT_GE( entityDimension, 0, "dimension must be greater than or equal to 0" );
   TNL_ASSERT_LE( entityDimension, Dimension, "dimension must be less than or equal to Dimension" );

   if( entityDimension == 0 || entityDimension == Dimension )
      return this->getEntitiesCount( entityDimension );

   const Index index = EntitiesOrientations::getTotalOrientationIndex( entityDimension, orientation ) + entityDimension;
   return this->entitiesIndexesOffsets[ index + 1 ] - this->entitiesIndexesOffsets[ index ];
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension,
          int EntityOrientationIdx,
          std::enable_if_t< Templates::isInClosedInterval( 0, EntityDimension, Dimension ), bool >,
          std::enable_if_t< Templates::isInClosedInterval( 0, EntityOrientationIdx, Dimension ), bool > >
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientedEntitiesCount() const noexcept
{
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
__cuda_callable__
Index
Grid< Dimension, Real, Device, Index >::getOrientationIndex( const NormalsType& normals ) const noexcept
{
   return entitiesOrientations.getOrientationIndex( normals );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getEntityCoordinates( IndexType entityIdx,
                                                              EntityOrientation< EntityDimension >& orientation ) const noexcept
   -> CoordinatesType
{
   if constexpr( EntityDimension != 0 && EntityDimension != getMeshDimension() ) {
      IndexType i = EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( 0 ) + EntityDimension + 1;
      const Index end = i + this->getEntityOrientationsCount( EntityDimension ) + EntityDimension;
      IndexType orientationIdx = 0;
      while( i < end && entityIdx >= this->entitiesIndexesOffsets[ i ] ) {
         i++;
         orientationIdx++;
      }
      entityIdx -= this->entitiesIndexesOffsets[ i-1 ];
      orientation.setTotalOrientationIndex( EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >(orientationIdx) ); // TODO: compute directly total orientation index
   }
   const CoordinatesType dims = this->getDimensions() + getNormals( orientation.getTotalOrientationIndex() );
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

   fillSpaceStepsPowers();
   fillProportions();
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename... Steps,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Steps >... >, bool >,
          std::enable_if_t< sizeof...( Steps ) == Dimension, bool > >
void
Grid< Dimension, Real, Device, Index >::setSpaceSteps( Steps... spaceSteps ) noexcept
{
   this->spaceSteps = PointType( spaceSteps... );

   fillSpaceStepsPowers();
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
template< typename... Powers,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Powers >... >, bool >,
          std::enable_if_t< sizeof...( Powers ) == Dimension, bool > >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSpaceStepsProducts( Powers... powers ) const
{
   int index = Templates::makeCollapsedIndex( spaceStepsPowersSize, CoordinatesType( powers... ) );

   return this->spaceStepsProducts( index );
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSpaceStepsProducts( const CoordinatesType& powers ) const
{
   int index = Templates::makeCollapsedIndex( spaceStepsPowersSize, powers );

   return this->spaceStepsProducts( index );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< Index... Powers, std::enable_if_t< sizeof...( Powers ) == Dimension, bool > >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSpaceStepsProducts() const noexcept
{
   constexpr int index = Templates::makeCollapsedIndex< Index, Powers... >( spaceStepsPowersSize );

   return this->spaceStepsProducts( index );
}

template< int Dimension, typename Real, typename Device, typename Index >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::getSmallestSpaceStep() const noexcept
{
   Real minStep = this->spaceSteps[ 0 ];
   Index i = 1;

   while( i != Dimension )
      minStep = min( minStep, this->spaceSteps[ i++ ] );

   return minStep;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseAll( Func func, FuncArgs... args ) const
{
   this->traverseAll< EntityDimension >( CoordinatesType( 0 ), this->getDimensions(), std::move( func ), std::move( args )... );
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
   TNL_ASSERT_ALL_LE( to, this->getDimensions(), "Traverse rect be in the grid dimensions" );
   TNL_ASSERT_ALL_LE( from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor" );

   if constexpr( EntityDimension == getMeshDimension() ) {
      Templates::ParallelFor< Dimension, Device, Index >::exec( from, to, func, args... );
   }
   else {
      auto exec = [ & ]( const Index orientation, const NormalsType& normals )
      {
         TNL_ASSERT_EQ( orientation, this->getOrientationIndex( normals ), "Wrong index of entity orientation." );
         Templates::ParallelFor< Dimension, Device, Index >::exec( from, to + normals, func, normals, orientation, args... );
      };
      Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::traverseInterior( Func func, FuncArgs... args ) const
{
   this->traverseInterior< EntityDimension >(
      CoordinatesType( 0 ), this->getDimensions(), std::move( func ), std::move( args )... );
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
   TNL_ASSERT_ALL_LE( to, this->getDimensions(), "Traverse rect be in the grid dimensions" );
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
   this->traverseBoundary< EntityDimension >(
      CoordinatesType( 0 ), this->getDimensions(), std::move( func ), std::move( args )... );
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
   static_assert( Entity::getEntityDimension() <= Dimension && Entity::getEntityDimension() >= 0,
                  "Wrong grid entity dimensions." );
   TNL_ASSERT_GE( entity.getCoordinates(), CoordinatesType( 0 ), "Wrong entity coordinates" );
   TNL_ASSERT_LT( entity.getCoordinates(), this->getDimensions() + entity.getNormals(), "Wrong entity coordinates" );

   IndexType idx{ 0 };
   if constexpr( Entity::getEntityDimension() == Dimension )
   {
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += entity.getCoordinates()[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ];
      } );
      return idx + entity.getCoordinates()[ 0 ];
   }
   else if constexpr( Entity::getEntityDimension() == 0  )
   {
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += entity.getCoordinates()[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ] + 1;
      } );
      return idx + entity.getCoordinates()[ 0 ];
   }
   else
   {
      const auto& normals = entity.getNormals();
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += entity.getCoordinates()[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ] + normals[ Dimension - i - 1 ];
      } );
      idx += entity.getCoordinates()[ 0 ];
      return idx + this->entitiesIndexesOffsets[ entity.getOrientation().getTotalOrientationIndex() + Entity::getEntityDimension() ];
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
Grid< Dimension, Real, Device, Index >::getNeighbourEntityIndex( const Entity& entity, const CoordinatesType& offset ) const
   -> Index
{
   IndexType idx{ 0 };
   if constexpr( Entity::getEntityDimension() == getMeshDimension() )
   {
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += offset[ Dimension - i ]; //entity.getCoordinates()[ Dimension - i ] + offset[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ];
      } );
      //return idx + entity.getCoordinates()[ 0 ] + offset[ 0 ];
      return idx + offset[ 0 ] + entity.getIndex();
   }
   else if constexpr( Entity::getEntityDimension() == 0 )
   {
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += offset[ Dimension - i ]; //entity.getCoordinates()[ Dimension - i ] + offset[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ] + 1;
      } );
      //return idx + entity.getCoordinates()[ 0 ] + offset[ 0 ];
      return idx + offset[ 0 ] + entity.getIndex();
   }
   else {
      const auto& normals = entity.getNormals();
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += offset[ Dimension - i ]; //entity.getCoordinates()[ Dimension - i ] + offset[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ] + normals[ Dimension - i - 1 ];
      } );
      //return idx + entity.getCoordinates()[ 0 ] + offset[ 0 ];
      return idx + offset[ 0 ] + entity.getIndex();
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int NeighbourEntityDimension, typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getNeighbourEntityIndex( const Entity& entity,
                                                                 const CoordinatesType& offset,
                                                                 Index neighbourEntityOrientation ) const -> Index
{
   IndexType idx{ 0 };
   if constexpr( NeighbourEntityDimension == getMeshDimension() ) {
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += entity.getCoordinates()[ Dimension - i ] + offset[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ];
      } );
      return idx + entity.getCoordinates()[ 0 ] + offset[ 0 ];
   }
   if constexpr( NeighbourEntityDimension == 0 ) {
      Algorithms::staticFor< IndexType, 1, Dimension >(
         [&] ( Index i ) mutable {
            idx += entity.getCoordinates()[ Dimension - i ] + offset[ Dimension - i ];
            idx *= this->getDimensions()[ Dimension - i - 1 ] + 1;
      } );
      return idx + entity.getCoordinates()[ 0 ] + offset[ 0 ];
   }
   const IndexType totalOrientationIndex = EntitiesOrientations::getTotalOrientationIndex( NeighbourEntityDimension, neighbourEntityOrientation );
   const NormalsType& neighbourEntityNormals = this->getNormals( totalOrientationIndex );
   Algorithms::staticFor< IndexType, 1, Dimension >(
      [&] ( Index i ) mutable {
         idx += entity.getCoordinates()[ Dimension - i ] + offset[ Dimension - i ];
         idx *= this->getDimensions()[ Dimension - i - 1 ] + neighbourEntityNormals[ Dimension - i - 1 ];
      } );
   idx += entity.getCoordinates()[ 0 ] + offset[ 0 ];
   return this->entitiesIndexesOffsets[ totalOrientationIndex + NeighbourEntityDimension ] + idx;
}

template< int Dimension, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
Entity
Grid< Dimension, Real, Device, Index >::getNeighbourEntity( const Entity& entity, const CoordinatesType& offset ) const
{
   return Entity( *this, entity.getCoordinates() + offset, entity.getOrientation().getOrientationIndex() );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int NeighbourEntityDimension, typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::getNeighbourEntity( const Entity& entity,
                                                            const CoordinatesType& offset,
                                                            const NormalsType& neighbourEntityOrientation ) const
   -> EntityType< NeighbourEntityDimension >
{
   return EntityType< NeighbourEntityDimension >(
      *this, CoordinatesType( entity.getCoordinates() + offset ), neighbourEntityOrientation );
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::
getEntityOrigin( const Entity& entity ) const -> PointType
{
   return this->getOrigin() + entity.getCoordinates() * this->getSpaceSteps();
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename Entity >
__cuda_callable__
auto
Grid< Dimension, Real, Device, Index >::
getEntityCenter( const Entity& entity ) const -> PointType
{
   return this->getOrigin() + ( entity.getCoordinates() + 0.5 * entity.getBasis() ) * this->getSpaceSteps();
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename Entity >
__cuda_callable__
Real
Grid< Dimension, Real, Device, Index >::
getEntityMeasure( const Entity& entity ) const
{
   if constexpr( Entity::getEntityDimension() != 0 ) {
      const auto& basis = entity.getBasis();
      RealType measure = ( Real ) 1.0;
      Algorithms::staticFor< IndexType, 0, Dimension >(
         [&] ( Index i ) mutable {
            if( basis[ i ] )
               measure *= this->getSpaceSteps()[ i ];
      } );
      return measure;
   }
   return 0.0;
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename Entity >
__cuda_callable__
bool
Grid< Dimension, Real, Device, Index >::
isBoundaryEntity( const Entity& entity ) const
{
   bool result( false );
   const auto& coordinates = entity.getCoordinates();
   if constexpr( Entity::getEntityDimension() == Dimension ) {
      Algorithms::staticFor< IndexType, 0, Dimension >(
      [&] ( Index i ) mutable {
         if( coordinates[ i ] == 0 ||
             coordinates[ i ] == this->getDimensions()[ i ] - 1 )
             result = true; // return true does not work here
      } );
   } else if constexpr( Entity::getEntityDimension() == 0 ) {
      Algorithms::staticFor< IndexType, 0, Dimension >(
      [&] ( Index i ) mutable {
         if( coordinates[ i ] == 0 ||
             coordinates[ i ] == ( this->getDimensions()[ i ] ) )
             result = true; // return true does not work here
      } );
   } else {
      const auto& normals = entity.getNormals();
      Algorithms::staticFor< IndexType, 0, Dimension >(
         [&] ( Index i ) mutable {
            if( normals[ i ] && ( coordinates[ i ] == 0 || coordinates[ i ] == this->getDimensions()[ i ] ) )
                result = true; // return true does not work here
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
   logger.writeParameter( "Dimensions:", this->dimensions );

   logger.writeParameter( "Origin:", this->origin );
   logger.writeParameter( "Proportions:", this->proportions );
   logger.writeParameter( "Space steps:", this->spaceSteps );

   TNL::Algorithms::staticFor< IndexType, 0, Dimension + 1 >(
      [ & ]( auto entityDim )
      {
         for( IndexType entityOrientation = 0; entityOrientation < getEntityOrientationsCount( entityDim() );
              entityOrientation++ ) {
            auto normals = this->getBasis< entityDim >( entityOrientation );
            TNL::String tmp = TNL::String( "Entities count with basis " ) + TNL::convertToString( normals ) + ":";
            logger.writeParameter( tmp, this->getOrientedEntitiesCount( entityDim, entityOrientation ) );
         }
      } );
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::setEntitiesIndexesOffsets()
{
   if( getDimensions() == 0 ) {
      this->entitiesIndexesOffsets = 0;
      this->entitiesCounts = 0;
      return;
   }

   Index totalOrientationIdx = 0;
   for( int entityDimension = 0; entityDimension <= Dimension; entityDimension++ ) {
      this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension ] = 0;
      for( Index entityOrientation = 0; entityOrientation < EntitiesOrientations::getOrientationsCount( entityDimension );
           entityOrientation++, totalOrientationIdx++ )
      {
         const auto& normals = entitiesOrientations.getNormals( totalOrientationIdx );
         IndexType entitiesCount = 1;
         Algorithms::staticFor< IndexType, 0, Dimension >(
            [ & ]( Index i ) mutable
            {
               entitiesCount *= this->getDimensions()[ i ] + normals[ i ];
            } );
         this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension + 1 ] =
            this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension ] + entitiesCount;
      }
      this->entitiesCounts[ entityDimension ] = this->entitiesIndexesOffsets[ totalOrientationIdx + entityDimension ];
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillProportions()
{
   this->proportions = this->spaceSteps * this->dimensions;
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillSpaceSteps()
{
   bool hasAnyInvalidDimension = false;

   for( Index i = 0; i < Dimension; i++ ) {
      if( this->dimensions[ i ] <= 0 ) {
         hasAnyInvalidDimension = true;
         break;
      }
   }

   if( hasAnyInvalidDimension ) {
      this->spaceSteps = 0;
      this->spaceStepsProducts = 0;
   }
   else {
      this->spaceSteps = this->proportions / this->dimensions;
      fillSpaceStepsPowers();
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
void
Grid< Dimension, Real, Device, Index >::fillSpaceStepsPowers()
{
   Containers::StaticVector< spaceStepsPowersSize * Dimension, Real > powers;

   for( Index i = 0; i < Dimension; i++ ) {
      Index power = -( spaceStepsPowersSize >> 1 );

      for( Index j = 0; j < spaceStepsPowersSize; j++ ) {
         powers[ i * spaceStepsPowersSize + j ] = pow( this->spaceSteps[ i ], power );
         power++;
      }
   }

   for( Index i = 0; i < spaceStepsProducts.getSize(); i++ ) {
      Real product = 1;
      Index index = i;

      for( Index j = 0; j < Dimension; j++ ) {
         Index residual = index % spaceStepsPowersSize;

         index /= spaceStepsPowersSize;

         product *= powers[ j * spaceStepsPowersSize + residual ];
      }

      spaceStepsProducts[ i ] = product;
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forAllEntities( Func func, FuncArgs... args ) const
{
   if constexpr( EntityDimension == getMeshDimension() ) {
      auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate, const Grid& grid, FuncArgs... args ) mutable
      {
         EntityType< EntityDimension > entity( grid, coordinate );

         func( entity, args... );
      };
      this->template traverseAll< EntityDimension >( exec, *this, args... );
   }
   else {
      auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                           const NormalsType& normals,
                                           const Index orientation,
                                           const Grid& grid,
                                           FuncArgs... args ) mutable
      {
         TNL_ASSERT_EQ( normals, grid.template getNormals< EntityDimension >( orientation ), "Wrong index of entity orientation." );
         EntityType< EntityDimension > entity( grid, coordinate, orientation );

         func( entity, args... );
      };
      this->template traverseAll< EntityDimension >( exec, *this, args... );
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forEntities( const CoordinatesType& begin,
                                                     const CoordinatesType& end,
                                                     Func func,
                                                     FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const NormalsType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( begin, end, exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const NormalsType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( const CoordinatesType& begin,
                                                             const CoordinatesType& end,
                                                             Func func,
                                                             FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const NormalsType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( begin, end, exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forInteriorEntities( Func func, FuncArgs... args ) const
{
   if constexpr( EntityDimension == getMeshDimension() || EntityDimension == 0 ) {
      using GridEntityType = typename Grid::template EntityType< EntityDimension >;
      auto exec = [ = ] __cuda_callable__( GridEntityType & entity, const Grid& grid, FuncArgs... args ) mutable
      {
         entity.setGrid( grid );
         entity.refresh();
         func( entity, args... );
      };
      Algorithms::ParallelForND< Device, false >::exec(
         GridEntityType( CoordinatesType( 1 ) ),
         GridEntityType( CoordinatesType( this->getDimensions() - CoordinatesType( EntityDimension != 0 ) ) ),
         exec,
         *this,
         args... );
   }
   else {
      auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                          const NormalsType& normals,
                                           const Index orientation,
                                           const Grid& grid,
                                           FuncArgs... args ) mutable
      {
         EntityType< EntityDimension > entity( grid, coordinate, orientation );

         func( entity, args... );
      };
      this->template traverseInterior< EntityDimension >( exec, *this, args... );
   }
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forLocalEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const NormalsType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( this->localBegin, this->localEnd, exec, *this, args... );
}

template< int Dimension, typename Real, typename Device, typename Index >
   template< typename Vector >
auto
Grid< Dimension, Real, Device, Index >::
partitionEntities( const Vector& allEntities, int entitiesDimension, int entitiesOrientation ) const -> typename Vector::ConstViewType
{
   const IndexType totalOrientationIdx = EntitiesOrientations::getTotalOrientationIndex( entitiesDimension, entitiesOrientation );
   return allEntities.getConstView( this->entitiesIndexesOffsets[ totalOrientationIdx+entitiesDimension ],
                                    this->entitiesIndexesOffsets[ totalOrientationIdx+entitiesDimension+1 ] );
}

}  // namespace TNL::Meshes

#endif
