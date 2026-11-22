// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

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
 * \brief Kernel of the traversals of cells or vertices, which have only one orientation.
 */
template< typename GridEntityType, typename Grid, typename Func >
struct EntitiesKernel
{
   Func func;

   __cuda_callable__
   void
   operator()( const typename Grid::CoordinatesType& coordinates, const Grid& grid )
   {
      GridEntityType entity( grid, coordinates );
      func( entity );
   }
};

/**
 * \brief Traverses the entities with the given dimension within the boxes given by `getBox`.
 *
 * For each orientation of the entities, `getBox( normals )` returns the pair `( begin, end )`
 * of the box `[ begin, end )` with the entities to be traversed,
 * where `normals` are the normals of the entities.
 */
template< int EntityDimension, typename Grid, typename Func, typename GetBox >
void
forEntitiesInBoxes(
   const Grid& grid,
   Func func,
   typename Grid::DeviceType::LaunchConfiguration launchConfiguration,
   GetBox&& getBox )
{
   // The kernels are function objects instead of lambda functions, because nvcc does not allow local types
   // (such as the type of getBox) in the template arguments of the functions with extended lambda functions.
   using CoordinatesType = typename Grid::CoordinatesType;
   using GridEntityType = typename Grid::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == Grid::getMeshDimension() || EntityDimension == 0 ) {
      const EntitiesKernel< GridEntityType, Grid, Func > kernel{ func };
      const auto box = getBox( CoordinatesType( EntityDimension == 0 ) );
      Algorithms::parallelFor< typename Grid::DeviceType >( box.first, box.second, launchConfiguration, kernel, grid );
   }
   else {
      forOrientedEntities< EntityDimension >(
         grid,
         func,
         [ & ]( const auto& kernel, const auto& normals )
         {
            const auto box = getBox( normals );
            Algorithms::parallelFor< typename Grid::DeviceType >( box.first, box.second, launchConfiguration, kernel, grid );
         } );
   }
}

/**
 * \brief Traverses the entities on the boundary of the region `[ begin, end )`.
 *
 * Only the sides of the region given by the flags `lowerSides` and `upperSides` are considered as the boundary,
 * see \ref TNL::Meshes::forBoundaryEntities. The region contains the entities with the normals `normals`
 * and coordinates `c` such that `begin <= c < end + normals * upperSides`, i.e. the entities with nonzero
 * normals at `end` belong to the region only if the upper side is on the boundary. This agrees with the
 * local entities of a subdomain (see \ref TNL::Meshes::forLocalEntities) and, if all sides are on the boundary,
 * with the region of \ref TNL::Meshes::forEntities.
 */
template< int EntityDimension, typename Grid, typename Func >
void
forBoundaryEntitiesOfRegion(
   const Grid& grid,
   const typename Grid::CoordinatesType& begin,
   const typename Grid::CoordinatesType& end,
   const typename Grid::BoundarySidesType& lowerSides,
   const typename Grid::BoundarySidesType& upperSides,
   Func func,
   typename Grid::DeviceType::LaunchConfiguration launchConfiguration )
{
   // The boundary of the region is decomposed into disjoint layers, so that only the boundary entities are
   // traversed. For each orientation, the entities lie in the box [ begin, end + normals * upperSides ). An entity is on
   // the boundary if its coordinate along some boundary axis is the first one in the box and the lower side
   // is on the boundary, or the last one and the upper side is on the boundary. The boundary axes are all axes
   // for cells and the axes with nonzero normals for the other entities, which agrees with isBoundaryEntity.
   // The layers of the k-th boundary axis exclude the boundary layers of the previous boundary axes, which were
   // already traversed.
   constexpr int Dimension = Grid::getMeshDimension();
   using CoordinatesType = typename Grid::CoordinatesType;
   forOrientedEntities< EntityDimension >(
      grid,
      func,
      [ & ]( const auto& kernel, const auto& normals )
      {
         auto traverseLayer = [ & ]( const CoordinatesType& layerBegin, const CoordinatesType& layerEnd )
         {
            for( int i = 0; i < Dimension; i++ )
               if( layerEnd[ i ] <= layerBegin[ i ] )
                  return;
            Algorithms::parallelFor< typename Grid::DeviceType >( layerBegin, layerEnd, launchConfiguration, kernel, grid );
         };

         const CoordinatesType regionBegin = begin;
         CoordinatesType regionEnd = end;
         for( int i = 0; i < Dimension; i++ )
            if( upperSides[ i ] )
               regionEnd[ i ] += normals[ i ];
         CoordinatesType rangeBegin = regionBegin;
         CoordinatesType rangeEnd = regionEnd;
         for( int axis = 0; axis < Dimension; axis++ ) {
            if( EntityDimension != Dimension && ! normals[ axis ] )
               continue;
            if( regionEnd[ axis ] <= regionBegin[ axis ] )
               break;

            CoordinatesType layerBegin = rangeBegin;
            CoordinatesType layerEnd = rangeEnd;
            if( lowerSides[ axis ] ) {
               layerBegin[ axis ] = regionBegin[ axis ];
               layerEnd[ axis ] = regionBegin[ axis ] + 1;
               traverseLayer( layerBegin, layerEnd );
            }

            if( upperSides[ axis ] && ( ! lowerSides[ axis ] || regionEnd[ axis ] - 1 > regionBegin[ axis ] ) ) {
               layerBegin[ axis ] = regionEnd[ axis ] - 1;
               layerEnd[ axis ] = regionEnd[ axis ];
               traverseLayer( layerBegin, layerEnd );
            }

            rangeBegin[ axis ] = regionBegin[ axis ] + ( lowerSides[ axis ] ? 1 : 0 );
            rangeEnd[ axis ] = regionEnd[ axis ] - ( upperSides[ axis ] ? 1 : 0 );
         }
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
   // the grid with zero sizes has no entities, although the boxes of the vertices are not empty
   if( grid.template getEntitiesCount< EntityDimension >() == 0 )
      return;
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
   // the grid with zero sizes has no entities, although the boxes of the vertices are not empty
   if( grid.template getEntitiesCount< EntityDimension >() == 0 )
      return;
   // The interior entities are the local entities (see forLocalEntities) without the entities on the global
   // boundary. Along an axis, the first layer is excluded if the lower side is on the global boundary and it is
   // a boundary axis of the entity (all axes for cells, axes with nonzero normals for the other entities). The
   // last layer of the cells is excluded if the upper side is on the global boundary. The local entities with
   // nonzero normals along the axis do not reach beyond localEnd unless they lie on the global boundary.
   using CoordinatesType = typename Grid< Dimension, Real, Device, Index >::CoordinatesType;
   const CoordinatesType& localBegin = grid.getLocalBegin();
   const CoordinatesType& localEnd = grid.getLocalEnd();
   const auto& lower = grid.getLowerGlobalBoundarySides();
   const auto& upper = grid.getUpperGlobalBoundarySides();
   detail::forEntitiesInBoxes< EntityDimension >(
      grid,
      func,
      launchConfiguration,
      [ & ]( const auto& normals )
      {
         CoordinatesType begin = localBegin;
         CoordinatesType end = localEnd;
         for( int i = 0; i < Dimension; i++ ) {
            if( lower[ i ] && ( EntityDimension == Dimension || normals[ i ] ) )
               begin[ i ]++;
            if( upper[ i ] && EntityDimension == Dimension )
               end[ i ]--;
         }
         return std::make_pair( begin, end );
      } );
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forBoundaryEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   // the grid with zero sizes has no entities, although the boxes of the vertices are not empty
   if( grid.template getEntitiesCount< EntityDimension >() == 0 )
      return;
   // The boundary entities are the local entities (see forLocalEntities) on the sides on the global boundary.
   detail::forBoundaryEntitiesOfRegion< EntityDimension >(
      grid,
      grid.getLocalBegin(),
      grid.getLocalEnd(),
      grid.getLowerGlobalBoundarySides(),
      grid.getUpperGlobalBoundarySides(),
      func,
      launchConfiguration );
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
   // the grid with zero sizes has no entities, although the boxes of the vertices are not empty
   if( grid.template getEntitiesCount< EntityDimension >() == 0 )
      return;
   using BoundarySidesType = typename Grid< Dimension, Real, Device, Index >::BoundarySidesType;
   detail::forBoundaryEntitiesOfRegion< EntityDimension >(
      grid, begin, end, BoundarySidesType( true ), BoundarySidesType( true ), func, launchConfiguration );
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forLocalEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration )
{
   // the grid with zero sizes has no entities, although the boxes of the vertices are not empty
   if( grid.template getEntitiesCount< EntityDimension >() == 0 )
      return;
   // The entities with coordinates c are owned by the subdomain which owns the cell c. The entities with nonzero
   // normals at localEnd have no cell in the subdomain, they are owned by this subdomain only on the global boundary.
   using CoordinatesType = typename Grid< Dimension, Real, Device, Index >::CoordinatesType;
   const CoordinatesType& localBegin = grid.getLocalBegin();
   const CoordinatesType& localEnd = grid.getLocalEnd();
   const auto& upper = grid.getUpperGlobalBoundarySides();
   detail::forEntitiesInBoxes< EntityDimension >(
      grid,
      func,
      launchConfiguration,
      [ & ]( const auto& normals )
      {
         CoordinatesType end = localEnd;
         for( int i = 0; i < Dimension; i++ )
            if( upper[ i ] )
               end[ i ] += normals[ i ];
         return std::make_pair( localBegin, end );
      } );
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
