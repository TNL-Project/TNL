// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/traverse.h>

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

   __cuda_callable__
   void
   operator()( const typename Grid::CoordinatesType& coordinates, const Grid& grid )
   {
      GridEntityType entity( coordinates );
      entity.setTotalOrientationIndex( TotalOrientationIndex );
      entity.setGrid( grid );
      entity.refresh();
      func( entity );
   }
};

/**
 * \brief Traverses the entities of all orientations with the given dimension within the boxes given by `getBox`.
 *
 * The orientations are traversed at compile time, see \ref OrientedEntitiesKernel.
 * For each orientation, `traverse( kernel, normals )` is called,
 * where `kernel` is the kernel for the orientation and `normals` are the normals of the entities.
 */
template< int EntityDimension, typename Grid, typename Func, typename Traverse >
void
forOrientedEntities( const Grid& grid, Func func, Traverse&& traverse )
{
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   using EntitiesOrientations = typename Grid::EntitiesOrientations;
   constexpr int orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
   constexpr int totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );

   Algorithms::staticFor< int, 0, orientationsCount >(
      [ & ]( auto orientation )
      {
         constexpr int totalOrientationIdx = totalOrientationsBegin + decltype( orientation )::value;
         const OrientedEntitiesKernel< GridEntityType, Grid, totalOrientationIdx, Func > kernel{ func };
         traverse( kernel, grid.getNormals( totalOrientationIdx ) );
      } );
}

/**
 * \brief Function object which calls `func( entity, args... )` with the stored arguments `args`.
 *
 * It is used by the deprecated member functions of the grid, which pass additional arguments to the function.
 * The arguments are stored recursively instead of being captured by a lambda function, because nvcc does not
 * allow extended lambda functions to capture the elements of a parameter pack.
 */
template< typename Func, typename... Args >
struct FunctionWithArguments;

template< typename Func >
struct FunctionWithArguments< Func >
{
   Func func;

   template< typename Entity, typename... Prefix >
   __cuda_callable__
   void
   operator()( Entity& entity, Prefix&... prefix )
   {
      func( entity, prefix... );
   }
};

template< typename Func, typename Arg, typename... Args >
struct FunctionWithArguments< Func, Arg, Args... >
{
   FunctionWithArguments< Func, Args... > rest;
   Arg arg;

   template< typename Entity, typename... Prefix >
   __cuda_callable__
   void
   operator()( Entity& entity, Prefix&... prefix )
   {
      rest( entity, prefix..., arg );
   }
};

template< typename Func >
FunctionWithArguments< Func >
bindArguments( Func func )
{
   return { func };
}

template< typename Func, typename Arg, typename... Args >
FunctionWithArguments< Func, Arg, Args... >
bindArguments( Func func, Arg arg, Args... args )
{
   return { bindArguments( func, args... ), arg };
}

}  // namespace detail

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forAllEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   using CoordinatesType = typename Grid< Dimension, Real, Device, Index >::CoordinatesType;
   forEntities< EntityDimension >( grid, CoordinatesType( 0 ), grid.getSizes(), func, launchConfiguration );
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& begin,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& end,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   using GridType = Grid< Dimension, Real, Device, Index >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityType = typename GridType::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == Dimension || EntityDimension == 0 ) {
      auto kernel = [ = ] __cuda_callable__( const CoordinatesType& coordinates, const GridType& kernelGrid ) mutable
      {
         GridEntityType entity( kernelGrid, coordinates );
         func( entity );
      };
      Algorithms::parallelFor< Device >(
         begin, CoordinatesType( end + CoordinatesType( EntityDimension == 0 ) ), launchConfiguration, kernel, grid );
   }
   else {
      detail::forOrientedEntities< EntityDimension >(
         grid,
         func,
         [ & ]( const auto& kernel, const auto& normals )
         {
            const CoordinatesType orientedEnd = end + normals;
            Algorithms::parallelFor< Device >( begin, orientedEnd, launchConfiguration, kernel, grid );
         } );
   }
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forInteriorEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   using GridType = Grid< Dimension, Real, Device, Index >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityType = typename GridType::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == Dimension || EntityDimension == 0 ) {
      auto kernel = [ = ] __cuda_callable__( const CoordinatesType& coordinates, const GridType& kernelGrid ) mutable
      {
         GridEntityType entity( kernelGrid, coordinates );
         func( entity );
      };
      Algorithms::parallelFor< Device >(
         CoordinatesType( 1 ),
         CoordinatesType( grid.getSizes() - CoordinatesType( EntityDimension != 0 ) ),
         launchConfiguration,
         kernel,
         grid );
   }
   else {
      detail::forOrientedEntities< EntityDimension >(
         grid,
         func,
         [ & ]( const auto& kernel, const auto& normals )
         {
            const CoordinatesType orientedBegin = CoordinatesType( 0 ) + normals;
            Algorithms::parallelFor< Device >( orientedBegin, grid.getSizes(), launchConfiguration, kernel, grid );
         } );
   }
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forBoundaryEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   using CoordinatesType = typename Grid< Dimension, Real, Device, Index >::CoordinatesType;
   forBoundaryEntities< EntityDimension >( grid, CoordinatesType( 0 ), grid.getSizes(), func, launchConfiguration );
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forBoundaryEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& begin,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& end,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   // The boundary of the region is decomposed into disjoint layers, so that only the boundary entities are
   // traversed. For each orientation, the entities lie in the box [ begin, end + normals ). An entity is on
   // the boundary if its coordinate along some boundary axis is the first or the last one in the box. The
   // boundary axes are all axes for cells and the axes with nonzero normals for the other entities, which
   // agrees with isBoundaryEntity. The layers of the k-th boundary axis exclude the first and the last
   // positions along the previous boundary axes, which were already traversed.
   using CoordinatesType = typename Grid< Dimension, Real, Device, Index >::CoordinatesType;
   detail::forOrientedEntities< EntityDimension >(
      grid,
      func,
      [ & ]( const auto& kernel, const auto& normals )
      {
         auto traverseLayer = [ & ]( const CoordinatesType& layerBegin, const CoordinatesType& layerEnd )
         {
            for( int i = 0; i < Dimension; i++ )
               if( layerEnd[ i ] <= layerBegin[ i ] )
                  return;
            Algorithms::parallelFor< Device >( layerBegin, layerEnd, launchConfiguration, kernel, grid );
         };

         const CoordinatesType regionBegin = begin;
         const CoordinatesType regionEnd = end + normals;
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

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forLocalEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   forEntities< EntityDimension >( grid, grid.getLocalBegin(), grid.getLocalEnd(), func, launchConfiguration );
}

// Deprecated member functions of Grid. They are defined here, because they call the free functions declared above.

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forAllEntities( Func func, FuncArgs... args ) const
{
   Meshes::forAllEntities< EntityDimension >( *this, detail::bindArguments( func, args... ) );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forEntities(
   const CoordinatesType& begin,
   const CoordinatesType& end,
   Func func,
   FuncArgs... args ) const
{
   Meshes::forEntities< EntityDimension >( *this, begin, end, detail::bindArguments( func, args... ) );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forInteriorEntities( Func func, FuncArgs... args ) const
{
   Meshes::forInteriorEntities< EntityDimension >( *this, detail::bindArguments( func, args... ) );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forBoundaryEntities( Func func, FuncArgs... args ) const
{
   Meshes::forBoundaryEntities< EntityDimension >( *this, detail::bindArguments( func, args... ) );
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
   Meshes::forBoundaryEntities< EntityDimension >( *this, begin, end, detail::bindArguments( func, args... ) );
}

template< int Dimension, typename Real, typename Device, typename Index >
template< int EntityDimension, typename Func, typename... FuncArgs >
void
Grid< Dimension, Real, Device, Index >::forLocalEntities( Func func, FuncArgs... args ) const
{
   Meshes::forLocalEntities< EntityDimension >( *this, detail::bindArguments( func, args... ) );
}

}  // namespace TNL::Meshes
