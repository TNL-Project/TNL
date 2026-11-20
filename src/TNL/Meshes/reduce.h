// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Functional.h>
#include <TNL/Meshes/Grid.h>

namespace TNL::Meshes {

/**
 * \brief Computes the reduction of values given by the function `fetch` over the grid entities
 * with the given dimension within the region `[ begin, end )`.
 *
 * The region contains the same entities as in \ref forEntities,
 * i.e. the entities of the cells within `[ begin, end )`:
 * the entities with the normal vector `normals` (see \ref TNL::Meshes::GridEntity::getNormals)
 * are processed if their coordinates `c` fulfil `begin <= c < end + normals`
 * (vertices if `begin <= c <= end`).
 *
 * The function `fetch` is called as `fetch( entity )`, where `entity` is the grid entity
 * of the type `Grid< Dimension, Real, Device, Index >::EntityType< EntityDimension >`
 * (see \ref TNL::Meshes::GridEntity), and returns the value of the entity.
 * The values are combined by the function `reduction`, for example:
 *
 * \code{.cpp}
 * const auto sum = reduce< Grid::getMeshDimension() >( grid, begin, end,
 *    [ = ] __cuda_callable__( const typename Grid::Cell& cell )
 *    {
 *       return view[ cell.getIndex() ];
 *    },
 *    TNL::Plus{},
 *    0.0 );
 * \endcode
 *
 * On GPUs, the functions are called in kernels on the device,
 * so they must be annotated as `__cuda_callable__`.
 * The function `reduction` is also called on the host to combine the results
 * of the entities with different orientations.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param begin is the 'lower left' corner of the region.
 * \param end is the 'upper right' corner of the region.
 * \param fetch is the function returning the value of each entity.
 * \param reduction is the function combining two values, for example \ref TNL::Plus.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *        of the reduction operation.
 * \return The result of the reduction, `identity` if the region is empty.
 */
template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename Result >
Result
reduce(
   const Grid< Dimension, Real, Device, Index >& grid,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& begin,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& end,
   Fetch fetch,
   Reduction reduction,
   const Result& identity );

/**
 * \brief Computes the reduction of values given by the function `fetch` over the grid entities
 * with the given dimension within the region `[ begin, end )`,
 * with a reduction function object.
 *
 * The identity element is obtained from the reduction function object
 * (see \ref ReductionFunctionObjects) and the type of the result
 * is the type of the values returned by `fetch`.
 * Otherwise, it is the same as the \ref reduce function with the identity element.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param begin is the 'lower left' corner of the region.
 * \param end is the 'upper right' corner of the region.
 * \param fetch is the function called as `fetch( entity )` returning the value of each entity.
 * \param reduction is the reduction function object, for example \ref TNL::Plus or \ref TNL::Max.
 * \return The result of the reduction, the identity element if the region is empty.
 */
template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction = TNL::Plus >
auto
reduce(
   const Grid< Dimension, Real, Device, Index >& grid,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& begin,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& end,
   Fetch fetch,
   Reduction reduction = TNL::Plus{} );

/**
 * \brief Computes the reduction of values given by the function `fetch` over all grid entities
 * with the given dimension.
 *
 * It is the same as the \ref reduce function for the region `[ 0, grid.getSizes() )`.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param fetch is the function called as `fetch( entity )` returning the value of each entity.
 * \param reduction is the function combining two values, for example \ref TNL::Plus.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *        of the reduction operation.
 * \return The result of the reduction.
 */
template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename Result,
   std::enable_if_t< ! std::is_convertible_v< Fetch, typename Grid< Dimension, Real, Device, Index >::CoordinatesType >, bool > =
      true >
Result
reduce( const Grid< Dimension, Real, Device, Index >& grid, Fetch fetch, Reduction reduction, const Result& identity );

/**
 * \brief Computes the reduction of values given by the function `fetch` over all grid entities
 * with the given dimension, with a reduction function object.
 *
 * It is the same as the \ref reduce function for the region `[ 0, grid.getSizes() )`.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param fetch is the function called as `fetch( entity )` returning the value of each entity.
 * \param reduction is the reduction function object, for example \ref TNL::Plus or \ref TNL::Max.
 * \return The result of the reduction.
 */
template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction = TNL::Plus,
   std::enable_if_t< ! std::is_convertible_v< Fetch, typename Grid< Dimension, Real, Device, Index >::CoordinatesType >, bool > =
      true >
auto
reduce( const Grid< Dimension, Real, Device, Index >& grid, Fetch fetch, Reduction reduction = TNL::Plus{} );

/**
 * \brief Computes the reduction of values given by the function `fetch` over the grid entities
 * with the given dimension owned by the local subdomain of the grid.
 *
 * The entities are the same as in \ref forLocalEntities.
 * For a subdomain of a distributed grid,
 * each entity of the global grid is owned by exactly one subdomain,
 * so the results of all subdomains can be combined to the reduction over the global grid
 * (see \ref TNL::Meshes::reduce for \ref TNL::Meshes::DistributedMeshes::DistributedMesh "distributed grids").
 * For a grid which is not a subdomain of a distributed grid,
 * the reduction is computed over all entities of the grid.
 * The functions are called in the same way as in \ref reduce.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param fetch is the function called as `fetch( entity )` returning the value of each entity.
 * \param reduction is the function combining two values, for example \ref TNL::Plus.
 * \param identity is the [identity element](https://en.wikipedia.org/wiki/Identity_element)
 *        of the reduction operation.
 * \return The result of the reduction, `identity` if there are no local entities.
 */
template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename Result >
Result
reduceLocalEntities(
   const Grid< Dimension, Real, Device, Index >& grid,
   Fetch fetch,
   Reduction reduction,
   const Result& identity );

/**
 * \brief Computes the reduction of values given by the function `fetch` over the grid entities
 * with the given dimension owned by the local subdomain of the grid, with a reduction function object.
 *
 * The identity element is obtained from the reduction function object
 * (see \ref ReductionFunctionObjects).
 * Otherwise, it is the same as the \ref reduceLocalEntities function with the identity element.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param grid is the grid.
 * \param fetch is the function called as `fetch( entity )` returning the value of each entity.
 * \param reduction is the reduction function object, for example \ref TNL::Plus or \ref TNL::Max.
 * \return The result of the reduction.
 */
template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction = TNL::Plus >
auto
reduceLocalEntities( const Grid< Dimension, Real, Device, Index >& grid, Fetch fetch, Reduction reduction = TNL::Plus{} );

}  // namespace TNL::Meshes

#include <TNL/Meshes/GridDetails/reduce.hpp>
