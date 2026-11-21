// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Functional.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/Meshes/reduce.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>

namespace TNL::Meshes {

/**
 * \brief Computes the reduction of values given by the function `fetch` over all entities
 * with the given dimension of a distributed grid.
 *
 * Each rank computes the reduction over the entities owned by its subdomain
 * (see \ref reduceLocalEntities),
 * so each entity of the global grid is counted exactly once.
 * The results of all ranks are gathered and combined by the function `reduction` on the host,
 * in the order of the ranks, so all ranks get the same result.
 * Hence, the reduction can be any function,
 * not only an operation supported by MPI.
 * The result is sent by MPI byte by byte,
 * so its type can be a fundamental type, \ref TNL::Containers::StaticVector
 * or a structure of such types without pointers.
 * Types with a non-trivial destructor (owning resources) are rejected at compile time.
 * The function must be called by all ranks of the communicator of the distributed grid.
 *
 * The functions are called in the same way as in \ref reduce for grids, for example:
 *
 * \code{.cpp}
 * const auto sum = reduce< Grid::getMeshDimension() >( distributedGrid,
 *    [ = ] __cuda_callable__( const typename Grid::Cell& cell )
 *    {
 *       return view[ cell.getIndex() ];
 *    },
 *    TNL::Plus{},
 *    0.0 );
 * \endcode
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param distributedGrid is the distributed grid.
 * \param fetch is the function called as `fetch( entity )` for the entities of the local grid
 *        (see \ref TNL::Meshes::DistributedMeshes::DistributedMesh::getLocalMesh),
 *        returning the value of each entity.
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
   typename Result >
Result
reduce(
   const DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > >& distributedGrid,
   Fetch fetch,
   Reduction reduction,
   const Result& identity )
{
   // StaticVector is not trivially copyable because of its user-provided copy constructor, so at least the types
   // owning resources (with a non-trivial destructor) are rejected
   static_assert(
      std::is_trivially_destructible_v< Result >,
      "The results are sent by MPI byte by byte, so they must not own any resources." );
   const Result localResult =
      reduceLocalEntities< EntityDimension >( distributedGrid.getLocalMesh(), fetch, reduction, identity );
   if( ! distributedGrid.isDistributed() )
      return localResult;

   const MPI::Comm& communicator = distributedGrid.getCommunicator();
   std::vector< Result > results( communicator.size(), identity );
   MPI::Allgather(
      reinterpret_cast< const char* >( &localResult ),
      static_cast< int >( sizeof( Result ) ),
      reinterpret_cast< char* >( results.data() ),
      static_cast< int >( sizeof( Result ) ),
      communicator );
   Result result = identity;
   for( const Result& rankResult : results )
      result = reduction( result, rankResult );
   return result;
}

/**
 * \brief Computes the reduction of values given by the function `fetch` over all entities
 * with the given dimension of a distributed grid, with a reduction function object.
 *
 * The identity element is obtained from the reduction function object
 * (see \ref ReductionFunctionObjects) and the type of the result
 * is the type of the values returned by `fetch`.
 * Otherwise, it is the same as the function with the identity element.
 *
 * \tparam EntityDimension is the dimension of the grid entities.
 * \param distributedGrid is the distributed grid.
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
reduce(
   const DistributedMeshes::DistributedMesh< Grid< Dimension, Real, Device, Index > >& distributedGrid,
   Fetch fetch,
   Reduction reduction = TNL::Plus{} )
{
   using GridEntityType = typename Grid< Dimension, Real, Device, Index >::template EntityType< EntityDimension >;
   using Result = std::decay_t< decltype( fetch( std::declval< GridEntityType& >() ) ) >;
   return reduce< EntityDimension >( distributedGrid, fetch, reduction, reduction.template getIdentity< Result >() );
}

}  // namespace TNL::Meshes
