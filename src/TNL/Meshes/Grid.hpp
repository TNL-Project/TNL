
// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/Templates/BooleanOperations.h>
#include <TNL/Meshes/GridDetails/NormalsGetter.h>
#include <TNL/Meshes/GridDetails/Templates/Functions.h>
#include <TNL/Meshes/GridDetails/Templates/ParallelFor.h>
#include <TNL/Meshes/GridDetails/Templates/DescendingFor.h>
#include <TNL/Meshes/GridDetails/Templates/ForEachOrientation.h>
#include <TNL/Algorithms/staticFor.h>

namespace TNL {
   namespace Meshes {

template< int Dimension_, typename Real, typename Device, typename Index >
constexpr int
Grid< Dimension_, Real, Device, Index >::
getMeshDimension()
{
   return Dimension;
}

template< int Dimension_, typename Real, typename Device, typename Index >
Grid< Dimension_, Real, Device, Index >::
Grid()
{
   CoordinatesType zero = 0;
   setDimensions( zero );

   PointType zeroPoint = 0;
   this->proportions = zeroPoint;
   this->spaceSteps = zeroPoint;
   this->origin = zeroPoint;
   fillNormals();
   fillEntitiesCount();
}

template< int Dimension_, typename Real, typename Device, typename Index >
   template< typename... Dimensions,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool >,
             std::enable_if_t< sizeof...( Dimensions ) == Dimension_, bool > >
Grid< Dimension_, Real, Device, Index >::
Grid( Dimensions... dimensions )
{
   setDimensions( dimensions... );

   PointType zeroPoint = 0;
   proportions = zeroPoint;
   spaceSteps = zeroPoint;
   origin = zeroPoint;
}

template< int Dimension_, typename Real, typename Device, typename Index >
Grid< Dimension_, Real, Device, Index >::
Grid( const CoordinatesType& dimensions )
{
   setDimensions( dimensions );

   PointType zeroPoint = 0;
   proportions = zeroPoint;
   spaceSteps = zeroPoint;
   origin = zeroPoint;
}

template< int Dimension_, typename Real, typename Device, typename Index >
constexpr Index
Grid< Dimension_, Real, Device, Index >::
getEntityOrientationsCount( const Index entityDimension )
{
   const Index dimension = Dimension;

   return Templates::combination< Index >( entityDimension, dimension );
}

template< int Dimension_, typename Real, typename Device, typename Index >
   template< typename... Dimensions,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool >,
             std::enable_if_t< sizeof...( Dimensions ) == Dimension_, bool > >
void
Grid< Dimension_, Real, Device, Index >::setDimensions( Dimensions... dimensions )
{
   this->dimensions = CoordinatesType( dimensions... );

   TNL_ASSERT_GE( this->dimensions, CoordinatesType( 0 ), "Dimension must be positive" );

   fillNormals();
   fillEntitiesCount();
   fillSpaceSteps();
   this->localBegin = 0;
   this->localEnd = this->getDimensions();
   this->interiorBegin = 1;
   this->interiorEnd = this->getDimensions() - 1;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::setDimensions( const typename Grid< Dimension_, Real, Device, Index >::CoordinatesType& dimensions )
{
   this->dimensions = dimensions;

   TNL_ASSERT_GE( this->dimensions, CoordinatesType( 0 ), "Dimension must be positive" );

   fillNormals();
   fillEntitiesCount();
   fillSpaceSteps();
   this->localBegin = 0;
   this->localEnd = this->getDimensions();
   this->interiorBegin = 1;
   this->interiorEnd = this->getDimensions() - 1;
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< typename... DimensionIndex,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, DimensionIndex >... >, bool >,
          std::enable_if_t< ( sizeof...( DimensionIndex ) > 0 ), bool > >
__cuda_callable__
typename Grid< Dimension_, Real, Device, Index >::template Container< sizeof...( DimensionIndex ), Index >
Grid< Dimension_, Real, Device, Index >::getDimensions( DimensionIndex... indices ) const noexcept
{
   Container< sizeof...( DimensionIndex ), Index > result{ indices... };

   for( std::size_t i = 0; i < sizeof...( DimensionIndex ); i++ )
      result[ i ] = this->getDimensions()[ result[ i ] ];

   return result;
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension_, Real, Device, Index >::CoordinatesType&
Grid< Dimension_, Real, Device, Index >::getDimensions() const noexcept
{
   return this->dimensions;
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< Dimension_, Real, Device, Index >::getEntitiesCount( const Index index ) const
{
   TNL_ASSERT_GE( index, 0, "Index must be greater than zero" );
   TNL_ASSERT_LE( index, Dimension_, "Index must be less than or equal to Dimension" );

   return this->cumulativeEntitiesCountAlongNormals( index );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, std::enable_if_t< Templates::isInClosedInterval( 0, EntityDimension, Dimension_ ), bool > >
__cuda_callable__
Index
Grid< Dimension_, Real, Device, Index >::getEntitiesCount() const noexcept
{
   return this->cumulativeEntitiesCountAlongNormals( EntityDimension );
}

template< int Dimension_, typename Real, typename Device, typename Index >
   template< typename EntityType,
             std::enable_if_t< Templates::isInClosedInterval( 0, EntityType::getEntityDimension(), Dimension_ ), bool > >
__cuda_callable__
Index
Grid< Dimension_, Real, Device, Index >::
getEntitiesCount() const noexcept
{
   return this->template getEntitiesCount< EntityType::getEntityDimension() >();
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< typename... DimensionIndex,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, DimensionIndex >... >, bool >,
          std::enable_if_t< ( sizeof...( DimensionIndex ) > 0 ), bool > >
__cuda_callable__
typename Grid< Dimension_, Real, Device, Index >::template Container< sizeof...( DimensionIndex ), Index >
Grid< Dimension_, Real, Device, Index >::getEntitiesCounts( DimensionIndex... indices ) const
{
   Container< sizeof...( DimensionIndex ), Index > result{ indices... };

   for( std::size_t i = 0; i < sizeof...( DimensionIndex ); i++ )
      result[ i ] = this->cumulativeEntitiesCountAlongNormals( result[ i ] );

   return result;
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension_, Real, Device, Index >::EntitiesCounts&
Grid< Dimension_, Real, Device, Index >::getEntitiesCounts() const noexcept
{
   return this->cumulativeEntitiesCountAlongNormals;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::setOrigin( const typename Grid< Dimension_, Real, Device, Index >::PointType& origin ) noexcept
{
   this->origin = origin;
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< typename... Coordinates,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Coordinates >... >, bool >,
          std::enable_if_t< sizeof...( Coordinates ) == Dimension_, bool > >
void
Grid< Dimension_, Real, Device, Index >::setOrigin( Coordinates... coordinates ) noexcept
{
   this->origin = PointType( coordinates... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
Index
Grid< Dimension_, Real, Device, Index >::getOrientedEntitiesCount( const Index dimension, const Index orientation ) const
{
   TNL_ASSERT_GE( dimension, 0, "Dimension must be greater than zero" );
   TNL_ASSERT_LE( dimension, Dimension_, "Requested dimension must be less than or equal to Dimension" );

   if( dimension == 0 || dimension == Dimension )
      return this->getEntitiesCount( dimension );

   Index index = Templates::firstKCombinationSum( dimension, ( Index ) Dimension ) + orientation;

   return this->entitiesCountAlongNormals[ index ];
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
typename Grid< Dimension_, Real, Device, Index >::CoordinatesType
Grid< Dimension_, Real, Device, Index >::getNormals( Index orientation ) const noexcept
{
   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension );

   return this->normals( index + orientation );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
typename Grid< Dimension_, Real, Device, Index >::CoordinatesType
Grid< Dimension_, Real, Device, Index >::getBasis( Index orientation ) const noexcept
{
   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension );
   CoordinatesType ones = 1.0;

   return ones - this->normals( index + orientation );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
Index
Grid< Dimension_, Real, Device, Index >::getOrientation( const CoordinatesType& normals ) const noexcept
{
   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension );
   const Index count = this->getEntityOrientationsCount( EntityDimension );
   for( IndexType orientation = 0; orientation < count; orientation++ )
      if( this->normals( index + orientation ) == normals )
         return orientation;
   return -1;
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension_, Real, Device, Index >::
getEntityCoordinates( IndexType entityIdx, CoordinatesType& entityNormals, Index& orientation ) const noexcept -> CoordinatesType
{
   orientation = Templates::firstKCombinationSum( EntityDimension, Dimension );
   const Index end = orientation + this->getEntityOrientationsCount( EntityDimension );
   auto entityIdx_( entityIdx );
   while( orientation < end && entityIdx_ >= this->entitiesCountAlongNormals[ orientation ] )
   {
      entityIdx_ -= this->entitiesCountAlongNormals[ orientation ];
      orientation++;
   }

   entityNormals = this->normals[ orientation ];
   const CoordinatesType dims = this->getDimensions() + entityNormals;
   CoordinatesType entityCoordinates( 0 );
   int idx = 0;
   while( idx < this->getMeshDimension() - 1 ) {
      entityCoordinates[ idx ] = entityIdx % dims[ idx ];
      entityIdx /= dims[ idx++ ];
   }
   entityCoordinates[ idx ] = entityIdx % dims[ idx ];
   return entityCoordinates;
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension,
          int EntityOrientation,
          std::enable_if_t< Templates::isInClosedInterval( 0, EntityDimension, Dimension_ ), bool >,
          std::enable_if_t< Templates::isInClosedInterval( 0, EntityOrientation, Dimension_ ), bool > >
__cuda_callable__
Index
Grid< Dimension_, Real, Device, Index >::getOrientedEntitiesCount() const noexcept
{
   if( EntityDimension == 0 || EntityDimension == Dimension )
      return this->getEntitiesCount( EntityDimension );

   constexpr Index index = Templates::firstKCombinationSum( EntityDimension, Dimension ) + EntityOrientation;

   return this->entitiesCountAlongNormals[ index ];
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension_, Real, Device, Index >::PointType&
Grid< Dimension_, Real, Device, Index >::getOrigin() const noexcept
{
   return this->origin;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::setDomain( const typename Grid< Dimension_, Real, Device, Index >::PointType& origin, const typename Grid< Dimension_, Real, Device, Index >::PointType& proportions )
{
   this->origin = origin;
   this->proportions = proportions;

   this->fillSpaceSteps();
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::setSpaceSteps( const typename Grid< Dimension_, Real, Device, Index >::PointType& spaceSteps ) noexcept
{
   this->spaceSteps = spaceSteps;

   fillSpaceStepsPowers();
   fillProportions();
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< typename... Steps,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Real, Steps >... >, bool >,
          std::enable_if_t< sizeof...( Steps ) == Dimension_, bool > >
void
Grid< Dimension_, Real, Device, Index >::setSpaceSteps( Steps... spaceSteps ) noexcept
{
   this->spaceSteps = PointType( spaceSteps... );

   fillSpaceStepsPowers();
   fillProportions();
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension_, Real, Device, Index >::PointType&
Grid< Dimension_, Real, Device, Index >::getSpaceSteps() const noexcept
{
   return this->spaceSteps;
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
const typename Grid< Dimension_, Real, Device, Index >::PointType&
Grid< Dimension_, Real, Device, Index >::getProportions() const noexcept
{
   return this->proportions;
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< typename... Powers,
          std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Powers >... >, bool >,
          std::enable_if_t< sizeof...( Powers ) == Dimension_, bool > >
__cuda_callable__
Real
Grid< Dimension_, Real, Device, Index >::getSpaceStepsProducts( Powers... powers ) const
{
   int index = Templates::makeCollapsedIndex( this->spaceStepsPowersSize, CoordinatesType( powers... ) );

   return this->spaceStepsProducts( index );
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
Real
Grid< Dimension_, Real, Device, Index >::getSpaceStepsProducts( const CoordinatesType& powers ) const
{
   int index = Templates::makeCollapsedIndex( this->spaceStepsPowersSize, powers );

   return this->spaceStepsProducts( index );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< Index... Powers, std::enable_if_t< sizeof...( Powers ) == Dimension_, bool > >
__cuda_callable__
Real
Grid< Dimension_, Real, Device, Index >::getSpaceStepsProducts() const noexcept
{
   constexpr int index = Templates::makeCollapsedIndex< Index, Powers... >( spaceStepsPowersSize );

   return this->spaceStepsProducts( index );
}

template< int Dimension_, typename Real, typename Device, typename Index >
__cuda_callable__
Real
Grid< Dimension_, Real, Device, Index >::getSmallestSpaceStep() const noexcept
{
   Real minStep = this->spaceSteps[ 0 ];
   Index i = 1;

   while( i != Dimension )
      minStep = min( minStep, this->spaceSteps[ i++ ] );

   return minStep;
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::traverseAll( Func func, FuncArgs... args ) const
{
   this->traverseAll< EntityDimension >( CoordinatesType( 0 ), this->getDimensions(), func, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::traverseAll( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const
{
   TNL_ASSERT_GE( from, CoordinatesType( 0 ), "Traverse rect must be in the grid dimensions" );
   TNL_ASSERT_LE( to, this->getDimensions(), "Traverse rect be in the grid dimensions" );
   TNL_ASSERT_LE( from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor" );

   auto exec = [ & ]( const Index orientation, const CoordinatesType& normals )
   {
      Templates::ParallelFor< Dimension_, Device, Index >::exec( from, to + normals, func, normals, orientation, args... );
   };
   Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::traverseInterior( Func func, FuncArgs... args ) const
{
   this->traverseInterior< EntityDimension >( CoordinatesType( 0 ), this->getDimensions(), func, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::traverseInterior( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const
{
   TNL_ASSERT_GE( from, CoordinatesType( 0 ), "Traverse rect must be in the grid dimensions" );
   TNL_ASSERT_LE( to, this->getDimensions(), "Traverse rect be in the grid dimensions" );
   TNL_ASSERT_LE( from, to, "Traverse rect must be defined from leading bottom anchor to trailing top anchor" );

   auto exec = [ & ]( const Index orientation, const CoordinatesType& normals )
   {
      switch( EntityDimension ) {
         case 0:
            {
               Templates::ParallelFor< Dimension_, Device, Index >::exec(
                  from + CoordinatesType( 1 ), to, func, normals, orientation, args... );
               break;
            }
         case Dimension:
            {
               Templates::ParallelFor< Dimension_, Device, Index >::exec(
                  from + CoordinatesType( 1 ), to - CoordinatesType( 1 ), func, normals, orientation, args... );
               break;
            }
         default:
            {
               Templates::ParallelFor< Dimension_, Device, Index >::exec( from + normals, to, func, normals, orientation, args... );
               break;
            }
      }
   };

   Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::traverseBoundary( Func func, FuncArgs... args ) const
{
   this->traverseBoundary< EntityDimension >( CoordinatesType( 0 ), this->getDimensions(), func, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::traverseBoundary( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const
{
   // Boundaries of the grid are formed by the entities of Dimension - 1.
   // We need to traverse each orientation independently.
   constexpr int orientationsCount = getEntityOrientationsCount( Dimension - 1 );
   constexpr bool isDirectedEntity = EntityDimension != 0 && EntityDimension != Dimension;
   constexpr bool isAnyBoundaryIntersects = EntityDimension != Dimension - 1;

   Container< orientationsCount, Index > isBoundaryTraversed( 0 );

   auto forBoundary = [ & ]( const auto orthogonalOrientation, const auto orientation, const CoordinatesType& normals )
   {
      CoordinatesType start = from;
      CoordinatesType end = to + normals;

      if( isAnyBoundaryIntersects ) {
#pragma unroll
         for( Index i = 0; i < Dimension; i++ ) {
            start[ i ] = ( ! isDirectedEntity || normals[ i ] ) && isBoundaryTraversed[ i ] ? 1 : 0;
            end[ i ] = end[ i ] - ( ( ! isDirectedEntity || normals[ i ] ) && isBoundaryTraversed[ i ] ? 1 : 0 );
         }
      }

      start[ orthogonalOrientation ] = end[ orthogonalOrientation ] - 1;

      Templates::ParallelFor< Dimension_, Device, Index >::exec( start, end, func, normals, orientation, args... );

      // Skip entities defined only once
      if( ! start[ orthogonalOrientation ] && end[ orthogonalOrientation ] )
         return;

      start[ orthogonalOrientation ] = 0;
      end[ orthogonalOrientation ] = 1;

      Templates::ParallelFor< Dimension_, Device, Index >::exec( start, end, func, normals, orientation, args... );
   };

   if( ! isAnyBoundaryIntersects ) {
      auto exec = [ & ]( const auto orientation, const CoordinatesType& normals )
      {
         constexpr int orthogonalOrientation = EntityDimension - orientation;

         forBoundary( orthogonalOrientation, orientation, normals );
      };

      Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
      return;
   }

   auto exec = [ & ]( const auto orthogonalOrientation )
   {
      auto exec = [ & ]( const auto orientation, const CoordinatesType& normals )
      {
         forBoundary( orthogonalOrientation, orientation, normals );
      };

      if( EntityDimension == 0 || EntityDimension == Dimension ) {
         Templates::ForEachOrientation< Index, EntityDimension, Dimension >::exec( exec );
      }
      else {
         Templates::ForEachOrientation< Index, EntityDimension, Dimension_, orthogonalOrientation >::exec( exec );
      }

      isBoundaryTraversed[ orthogonalOrientation ] = 1;
   };

   Templates::DescendingFor< orientationsCount - 1 >::exec( exec );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< typename Entity >
__cuda_callable__
Index
Grid< Dimension_, Real, Device, Index >::
getEntityIndex( const Entity& entity ) const
{
   static_assert( Entity::entityDimension <= Dimension && Entity::entityDimension >= 0, "Wrong grid entity dimensions." );

   return GridEntityGetter< Grid, Entity::entityDimension >::getEntityIndex( *this, entity );
}

template< int Dimension_, typename Real, typename Device, typename Index >
   template< typename EntityType >
__cuda_callable__
EntityType
Grid< Dimension_, Real, Device, Index >::
getEntity( const CoordinatesType& coordinates ) const
{
   static_assert( EntityType::getEntityDimension() <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType( *this, coordinates );
}

template< int Dimension_, typename Real, typename Device, typename Index >
   template< int EntityDimension >
__cuda_callable__
auto
Grid< Dimension_, Real, Device, Index >::
getEntity( const CoordinatesType& coordinates ) const -> EntityType< EntityDimension >
{
   static_assert( EntityDimension <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType< EntityDimension >( *this, coordinates );
}

template< int Dimension_, typename Real, typename Device, typename Index >
   template< typename EntityType >
EntityType
Grid< Dimension_, Real, Device, Index >::
getEntity( const IndexType& entityIdx ) const
{
   static_assert( EntityType::getEntityDimension() <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType( *this, entityIdx );
};

template< int Dimension_, typename Real, typename Device, typename Index >
   template< int EntityDimension >
auto
Grid< Dimension_, Real, Device, Index >::
getEntity( const IndexType& entityIdx ) const -> EntityType< EntityDimension >
{
   static_assert( EntityDimension <= getMeshDimension(), "Entity dimension must be lower or equal to grid dimension." );
   return EntityType< EntityDimension >( *this, entityIdx );
};

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::
setLocalSubdomain( const CoordinatesType& begin, const CoordinatesType& end )
{
   this->localBegin = begin;
   this->localEnd = end;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::
setLocalBegin( const CoordinatesType& begin )
{
   this->localBegin = begin;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::
setLocalEnd( const CoordinatesType& end )
{
   this->localEnd = end;
}

template< int Dimension_, typename Real, typename Device, typename Index >
auto
Grid< Dimension_, Real, Device, Index >::
getLocalBegin() const -> const CoordinatesType&
{
   return this->localBegin;
}

template< int Dimension_, typename Real, typename Device, typename Index >
auto
Grid< Dimension_, Real, Device, Index >::
getLocalEnd() const ->  const CoordinatesType&
{
   return this->localEnd;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::
setInteriorBegin( const CoordinatesType& begin )
{
   this->interiorBegin = begin;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::
setInteriorEnd( const CoordinatesType& end )
{
   this->interiorBegin = end;
}

template< int Dimension_, typename Real, typename Device, typename Index >
auto
Grid< Dimension_, Real, Device, Index >::
getInteriorBegin() const -> const CoordinatesType&
{
   return this->interiorBegin;
}

template< int Dimension_, typename Real, typename Device, typename Index >
auto
Grid< Dimension_, Real, Device, Index >::
getInteriorEnd() const -> const CoordinatesType&
{
   return this->interiorEnd;
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::writeProlog( TNL::Logger& logger ) const noexcept
{
   logger.writeParameter( "Dimensions:", this->dimensions );

   logger.writeParameter( "Origin:", this->origin );
   logger.writeParameter( "Proportions:", this->proportions );
   logger.writeParameter( "Space steps:", this->spaceSteps );

   TNL::Algorithms::staticFor< IndexType, 0, Dimension + 1 >(
      [&]( auto entityDim ) {
         for( IndexType entityOrientation = 0;
           entityOrientation < this->getEntityOrientationsCount( entityDim() );
           entityOrientation++ ) {
               auto normals = this->getBasis< entityDim >( entityOrientation );
               TNL::String tmp = TNL::String( "Entities count with basis " ) + TNL::convertToString( normals ) + ":";
               logger.writeParameter( tmp, this->getOrientedEntitiesCount( entityDim, entityOrientation) );
      } } );
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::fillEntitiesCount()
{
   for( Index i = 0; i < Dimension + 1; i++ )
      cumulativeEntitiesCountAlongNormals[ i ] = 0;

   // In case, if some dimension is zero. Clear all counts
   for( Index i = 0; i < Dimension; i++ ) {
      if( dimensions[ i ] == 0 ) {
         for( Index k = 0; k < (Index) entitiesCountAlongNormals.getSize(); k++ )
            entitiesCountAlongNormals[ k ] = 0;

         return;
      }
   }

   for( Index i = 0, j = 0; i <= Dimension; i++ ) {
      for( Index n = 0; n < this->getEntityOrientationsCount( i ); n++, j++ ) {
         int result = 1;
         auto normals = this->normals[ j ];


         for( Index k = 0; k < (Index) normals.getSize(); k++ )
            result *= dimensions[ k ] + normals[ k ];

         entitiesCountAlongNormals[ j ] = result;
         cumulativeEntitiesCountAlongNormals[ i ] += result;
      }
   }
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::fillProportions()
{
   Index i = 0;

   while( i != Dimension ) {
      this->proportions[ i ] = this->spaceSteps[ i ] * this->dimensions[ i ];
      i++;
   }
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::fillSpaceSteps()
{
   bool hasAnyInvalidDimension = false;

   for( Index i = 0; i < Dimension; i++ ) {
      if( this->dimensions[ i ] <= 0 ) {
         hasAnyInvalidDimension = true;
         break;
      }
   }

   if( ! hasAnyInvalidDimension ) {
      for( Index i = 0; i < Dimension; i++ )
         this->spaceSteps[ i ] = this->proportions[ i ] / this->dimensions[ i ];

      fillSpaceStepsPowers();
   }
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::fillSpaceStepsPowers()
{
   Container< spaceStepsPowersSize * Dimension_, Real > powers;

   for( Index i = 0; i < Dimension; i++ ) {
      Index power = -( this->spaceStepsPowersSize >> 1 );

      for( Index j = 0; j < spaceStepsPowersSize; j++ ) {
         powers[ i * spaceStepsPowersSize + j ] = pow( this->spaceSteps[ i ], power );
         power++;
      }
   }

   for( Index i = 0; i < this->spaceStepsProducts.getSize(); i++ ) {
      Real product = 1;
      Index index = i;

      for( Index j = 0; j < Dimension; j++ ) {
         Index residual = index % this->spaceStepsPowersSize;

         index /= this->spaceStepsPowersSize;

         product *= powers[ j * spaceStepsPowersSize + residual ];
      }

      spaceStepsProducts[ i ] = product;
   }
}

template< int Dimension_, typename Real, typename Device, typename Index >
void
Grid< Dimension_, Real, Device, Index >::
fillNormals()
{
   OrientationNormalsContainer container;
   for( int i = 0; i < OrientationNormalsContainer::getSize(); i++ )
      for( int j = 0; j < OrientationNormalsContainer::ValueType::getSize(); j++ )
         container[ i ][ j ] = 0;

   int index = container.getSize() - 1;

   auto forEachEntityDimension = [ & ]( const auto entityDimension )
   {
      constexpr Index dimension = entityDimension();
      constexpr Index combinationsCount = getEntityOrientationsCount( dimension );

      auto forEachOrientation = [ & ]( const auto orientation, const auto entityDimension )
      {
         container[ index-- ] = NormalsGetter< Index, entityDimension, Dimension >::template getNormals< orientation >();
      };
      Templates::DescendingFor< combinationsCount - 1 >::exec( forEachOrientation, entityDimension );
   };

   Templates::DescendingFor< Dimension >::exec( forEachEntityDimension );
   this->normals = container;
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::
forAllEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( exec, *this, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::
forEntities( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( from, to, exec, *this, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::
forBoundaryEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( exec, *this, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::
forBoundaryEntities( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseBoundary< EntityDimension >( from, to, exec, *this, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::
forInteriorEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseInterior< EntityDimension >( exec, *this, args... );
}

template< int Dimension_, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension_, Real, Device, Index >::
forLocalEntities( Func func, FuncArgs... args ) const
{
   auto exec = [ = ] __cuda_callable__( const CoordinatesType& coordinate,
                                        const CoordinatesType& normals,
                                        const Index orientation,
                                        const Grid& grid,
                                        FuncArgs... args ) mutable
   {
      EntityType< EntityDimension > entity( grid, coordinate, normals, orientation );

      func( entity, args... );
   };

   this->template traverseAll< EntityDimension >( this->localBegin, this->localEnd, exec, *this, args... );
}

}  // namespace Meshes
}  // namespace TNL
