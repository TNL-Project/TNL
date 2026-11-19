// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL::Meshes {

/**
 * \brief Calls the function `func` for all grid entities with the given dimension.
 *
 * The function is called as `func( entity )`, where `entity` is the grid entity
 * of the type `Grid< Dimension, Real, Device, Index >::EntityType< EntityDimension >`
 * (see \ref TNL::Meshes::GridEntity), for example:
 *
 * \code{.cpp}
 * forAllEntities< Grid::getMeshDimension() >( grid,
 *    [ = ] __cuda_callable__( const typename Grid::Cell& cell ) mutable
 *    {
 *       view[ cell.getIndex() ] = 0;
 *    } );
 * \endcode
 *
 * On GPUs, the function is called in kernels on the device,
 * so it must be annotated as `__cuda_callable__`.
 *
 * The launch configuration is used for all kernels of the traversal.
 * For example, with `blockHostUntilFinished` set to `false`,
 * the function does not wait for the kernels on the GPU to finish.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param func is the function called for each entity.
 * \param launchConfiguration is the configuration of the kernel launches.
 */
template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forAllEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration = typename Device::LaunchConfiguration{} );

/**
 * \brief Calls the function `func` for the grid entities with the given dimension
 * within the region `[ begin, end )`.
 *
 * The entities with the normal vector `normals` (see \ref TNL::Meshes::GridEntity::getNormals)
 * are processed if their coordinates `c` fulfil `begin <= c < end + normals`
 * (vertices if `begin <= c <= end`).
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param begin is the 'lower left' corner of the region.
 * \param end is the 'upper right' corner of the region.
 * \param func is the function called as `func( entity )` for each entity.
 * \param launchConfiguration is the configuration of the kernel launches, see \ref forAllEntities.
 */
template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& begin,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& end,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration = typename Device::LaunchConfiguration{} );

/**
 * \brief Calls the function `func` for the interior grid entities with the given dimension.
 *
 * The interior entities are the entities which do not lie on the boundary of the grid.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param func is the function called as `func( entity )` for each entity.
 * \param launchConfiguration is the configuration of the kernel launches, see \ref forAllEntities.
 */
template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forInteriorEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration = typename Device::LaunchConfiguration{} );

/**
 * \brief Calls the function `func` for the boundary grid entities with the given dimension.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param func is the function called as `func( entity )` for each entity.
 * \param launchConfiguration is the configuration of the kernel launches, see \ref forAllEntities.
 */
template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forBoundaryEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration = typename Device::LaunchConfiguration{} );

/**
 * \brief Calls the function `func` for the grid entities with the given dimension
 * on the boundary of the region `[ begin, end )`.
 *
 * An entity is on the boundary of the region if it is processed by \ref forEntities
 * with the same region and its coordinate along some boundary axis is the first or the last one.
 * The boundary axes are all axes for cells
 * and the axes along which the normal vector of the entity is nonzero for the other entities.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param begin is the 'lower left' corner of the region.
 * \param end is the 'upper right' corner of the region.
 * \param func is the function called as `func( entity )` for each entity.
 * \param launchConfiguration is the configuration of the kernel launches, see \ref forAllEntities.
 */
template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forBoundaryEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& begin,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& end,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration = typename Device::LaunchConfiguration{} );

/**
 * \brief Calls the function `func` for the grid entities with the given dimension
 * within the local subdomain of the grid.
 *
 * The local subdomain is the region `[ grid.getLocalBegin(), grid.getLocalEnd() )`,
 * see \ref forEntities.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param func is the function called as `func( entity )` for each entity.
 * \param launchConfiguration is the configuration of the kernel launches, see \ref forAllEntities.
 */
template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Func >
void
forLocalEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Func func,
   typename Device::LaunchConfiguration launchConfiguration = typename Device::LaunchConfiguration{} );

}  // namespace TNL::Meshes

#include <TNL/Meshes/GridDetails/traverse.hpp>
