// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <utility>

#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/reduce.h>

namespace TNL::Meshes {

namespace detail {

/**
 * \brief Function object which fetches the values of the grid entities in the box `[ begin, begin + sizes )`
 * for \ref Algorithms::reduce.
 *
 * The box is mapped to the one-dimensional range of indices, with the x-coordinate changing the fastest
 * (like in the numbering of the grid entities). For `TotalOrientationIndex < 0`, the entities are cells
 * or vertices, which have only one orientation. Otherwise, the entities have the given total orientation
 * index, which is a compile-time constant for the same reason as in \ref OrientedEntitiesKernel.
 *
 * It is a class instead of a lambda function, because nvcc does not allow extended lambda functions
 * to capture variables in `if constexpr` blocks.
 */
template< typename GridEntityType, typename Grid, typename Fetch, typename Result, int TotalOrientationIndex >
struct BoxFetch
{
   using CoordinatesType = typename Grid::CoordinatesType;
   using IndexType = typename Grid::IndexType;

   Grid grid;
   Fetch fetch;
   CoordinatesType begin;
   CoordinatesType sizes;

   __cuda_callable__
   Result
   operator()( IndexType idx ) const
   {
      // the index is nonnegative, so the unsigned division gives the same result and it is faster,
      // and the remainder is computed from the quotient to avoid the second division
      using UnsignedIndex = std::make_unsigned_t< IndexType >;
      constexpr int Dimension = Grid::getMeshDimension();
      CoordinatesType coordinates;
      UnsignedIndex i = idx;
      for( int d = 0; d < Dimension - 1; d++ ) {
         const UnsignedIndex size = sizes[ d ];
         const UnsignedIndex quotient = i / size;
         coordinates[ d ] = begin[ d ] + static_cast< IndexType >( i - quotient * size );
         i = quotient;
      }
      coordinates[ Dimension - 1 ] = begin[ Dimension - 1 ] + static_cast< IndexType >( i );

      if constexpr( TotalOrientationIndex < 0 ) {
         GridEntityType entity( grid, coordinates );
         return fetch( entity );
      }
      else {
         GridEntityType entity( coordinates );
         entity.setTotalOrientationIndex( TotalOrientationIndex );
         entity.setGrid( grid );
         entity.refresh();
         return fetch( entity );
      }
   }
};

/**
 * \brief Computes the reduction over the grid entities in the box `[ begin, end )`, see \ref BoxFetch.
 */
template< typename GridEntityType, int TotalOrientationIndex, typename Grid, typename Fetch, typename Reduction, typename Result >
Result
reduceBox(
   const Grid& grid,
   const typename Grid::CoordinatesType& begin,
   const typename Grid::CoordinatesType& end,
   Fetch fetch,
   Reduction reduction,
   const Result& identity )
{
   using CoordinatesType = typename Grid::CoordinatesType;
   using IndexType = typename Grid::IndexType;
   const CoordinatesType sizes = end - begin;
   IndexType count = 1;
   for( int i = 0; i < Grid::getMeshDimension(); i++ ) {
      if( sizes[ i ] <= 0 )
         return identity;
      count *= sizes[ i ];
   }
   const BoxFetch< GridEntityType, Grid, Fetch, Result, TotalOrientationIndex > boxFetch{ grid, fetch, begin, sizes };
   return Algorithms::reduce< typename Grid::DeviceType >( (IndexType) 0, count, boxFetch, reduction, identity );
}

}  // namespace detail

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
   const Result& identity )
{
   // the grid with zero sizes has no entities, although the boxes of the vertices are not empty
   if( grid.template getEntitiesCount< EntityDimension >() == 0 )
      return identity;
   using GridType = Grid< Dimension, Real, Device, Index >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityType = typename GridType::template EntityType< EntityDimension >;
   if constexpr( EntityDimension == Dimension || EntityDimension == 0 ) {
      // the cells lie in the box [ begin, end ), the vertices in [ begin, end + 1 )
      return detail::reduceBox< GridEntityType, -1 >(
         grid, begin, CoordinatesType( end + CoordinatesType( EntityDimension == 0 ) ), fetch, reduction, identity );
   }
   else {
      // The entities with the normal vector `normals` lie in the box [ begin, end + normals ). The entities
      // with different orientations are reduced separately, so that each reduction reads a contiguous block
      // of the entities, and the results are combined on the host.
      using EntitiesOrientations = typename GridType::EntitiesOrientations;
      constexpr int orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      constexpr int totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      Result result = identity;
      Algorithms::staticFor< int, 0, orientationsCount >(
         [ & ]( auto orientation )
         {
            constexpr int totalOrientationIdx = totalOrientationsBegin + decltype( orientation )::value;
            const CoordinatesType orientedEnd = end + grid.getNormals( totalOrientationIdx );
            result = reduction(
               result,
               detail::reduceBox< GridEntityType, totalOrientationIdx >(
                  grid, begin, orientedEnd, fetch, reduction, identity ) );
         } );
      return result;
   }
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Fetch, typename Reduction >
auto
reduce(
   const Grid< Dimension, Real, Device, Index >& grid,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& begin,
   const typename Grid< Dimension, Real, Device, Index >::CoordinatesType& end,
   Fetch fetch,
   Reduction reduction )
{
   using GridEntityType = typename Grid< Dimension, Real, Device, Index >::template EntityType< EntityDimension >;
   using Result = std::decay_t< decltype( fetch( std::declval< GridEntityType& >() ) ) >;
   return reduce< EntityDimension >( grid, begin, end, fetch, reduction, reduction.template getIdentity< Result >() );
}

template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction,
   typename Result,
   std::enable_if_t< ! std::is_convertible_v< Fetch, typename Grid< Dimension, Real, Device, Index >::CoordinatesType >, bool > >
Result
reduce( const Grid< Dimension, Real, Device, Index >& grid, Fetch fetch, Reduction reduction, const Result& identity )
{
   using CoordinatesType = typename Grid< Dimension, Real, Device, Index >::CoordinatesType;
   return reduce< EntityDimension >( grid, CoordinatesType( 0 ), grid.getSizes(), fetch, reduction, identity );
}

template<
   int EntityDimension,
   int Dimension,
   typename Real,
   typename Device,
   typename Index,
   typename Fetch,
   typename Reduction,
   std::enable_if_t< ! std::is_convertible_v< Fetch, typename Grid< Dimension, Real, Device, Index >::CoordinatesType >, bool > >
auto
reduce( const Grid< Dimension, Real, Device, Index >& grid, Fetch fetch, Reduction reduction )
{
   using CoordinatesType = typename Grid< Dimension, Real, Device, Index >::CoordinatesType;
   return reduce< EntityDimension >( grid, CoordinatesType( 0 ), grid.getSizes(), fetch, reduction );
}

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
   const Result& identity )
{
   // The local entities with the normal vector `normals` lie in the box [ localBegin, localEnd + normals * upper ),
   // where upper are the flags of the upper sides on the global boundary, see forLocalEntities.
   using GridType = Grid< Dimension, Real, Device, Index >;
   using CoordinatesType = typename GridType::CoordinatesType;
   using GridEntityType = typename GridType::template EntityType< EntityDimension >;
   const CoordinatesType& localBegin = grid.getLocalBegin();
   const auto& upper = grid.getUpperGlobalBoundarySides();
   auto getEnd = [ & ]( const auto& normals )
   {
      CoordinatesType end = grid.getLocalEnd();
      for( int i = 0; i < Dimension; i++ )
         if( upper[ i ] )
            end[ i ] += normals[ i ];
      return end;
   };
   if constexpr( EntityDimension == Dimension || EntityDimension == 0 ) {
      return detail::reduceBox< GridEntityType, -1 >(
         grid, localBegin, getEnd( CoordinatesType( EntityDimension == 0 ) ), fetch, reduction, identity );
   }
   else {
      using EntitiesOrientations = typename GridType::EntitiesOrientations;
      constexpr int orientationsCount = EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      constexpr int totalOrientationsBegin = EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );
      Result result = identity;
      Algorithms::staticFor< int, 0, orientationsCount >(
         [ & ]( auto orientation )
         {
            constexpr int totalOrientationIdx = totalOrientationsBegin + decltype( orientation )::value;
            result = reduction(
               result,
               detail::reduceBox< GridEntityType, totalOrientationIdx >(
                  grid, localBegin, getEnd( grid.getNormals( totalOrientationIdx ) ), fetch, reduction, identity ) );
         } );
      return result;
   }
}

template< int EntityDimension, int Dimension, typename Real, typename Device, typename Index, typename Fetch, typename Reduction >
auto
reduceLocalEntities( const Grid< Dimension, Real, Device, Index >& grid, Fetch fetch, Reduction reduction )
{
   using GridEntityType = typename Grid< Dimension, Real, Device, Index >::template EntityType< EntityDimension >;
   using Result = std::decay_t< decltype( fetch( std::declval< GridEntityType& >() ) ) >;
   return reduceLocalEntities< EntityDimension >( grid, fetch, reduction, reduction.template getIdentity< Result >() );
}

}  // namespace TNL::Meshes
