// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridEntitiesOrientations.h>

namespace TNL {
namespace Meshes {

/**
 * \brief Coordinates and orientation of a grid entity.
 *
 * The orientation is stored only for the entities with more orientations,
 * i.e. not for vertices and cells (see the partial specializations).
 *
 * \tparam Grid is the type of the grid.
 * \tparam GridDimension is the dimension of the grid.
 * \tparam EntityDimension is the dimension of the entity.
 */
template< typename Grid, int GridDimension, int EntityDimension >
class GridEntityBase
{
public:
   static constexpr int
   getMeshDimension()
   {
      return GridDimension;
   }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using CoordinatesType = typename Grid::CoordinatesType;

   using EntitiesOrientations = GridEntitiesOrientations< getMeshDimension() >;

   __cuda_callable__
   GridEntityBase() = default;

   template< typename Value >
   __cuda_callable__
   GridEntityBase( const std::initializer_list< Value >& elems )
   : coordinates( elems )
   {}

   __cuda_callable__
   GridEntityBase( IndexType totalOrientationIdx )
   : totalOrientationIndex( totalOrientationIdx )
   {}

   __cuda_callable__
   GridEntityBase( const CoordinatesType& coordinates, IndexType totalOrientationIdx = getDefaultTotalOrientationIndex() )
   : coordinates( coordinates ),
     totalOrientationIndex( totalOrientationIdx )
   {}

   __cuda_callable__
   void
   setCoordinates( const CoordinatesType& coordinates )
   {
      this->coordinates = coordinates;
   }

   __cuda_callable__
   const CoordinatesType&
   getCoordinates() const
   {
      return coordinates;
   }

   __cuda_callable__
   CoordinatesType&
   getCoordinates()
   {
      return coordinates;
   }

   __cuda_callable__
   void
   setTotalOrientationIndex( IndexType idx )
   {
      this->totalOrientationIndex = idx;
   }

   __cuda_callable__
   IndexType
   getTotalOrientationIndex() const
   {
      return this->totalOrientationIndex;
   }

   __cuda_callable__
   IndexType
   getOrientationIndex() const
   {
      return EntitiesOrientations::template getOrientationIndex< EntityDimension >( this->getTotalOrientationIndex() );
   }

protected:
   // the first orientation of entities with given dimension is used when no orientation is given
   static constexpr IndexType
   getDefaultTotalOrientationIndex()
   {
      return EntitiesOrientations::template getTotalOrientationIndex< EntityDimension >( 0 );
   }

   CoordinatesType coordinates;

   IndexType totalOrientationIndex = getDefaultTotalOrientationIndex();
};

template< typename Grid, int GridDimension >
class GridEntityBase< Grid, GridDimension, 0 >
{
public:
   static constexpr int
   getMeshDimension()
   {
      return Grid::getMeshDimension();
   }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using CoordinatesType = typename Grid::CoordinatesType;

   using EntitiesOrientations = GridEntitiesOrientations< getMeshDimension() >;

   __cuda_callable__
   GridEntityBase() = default;

   template< typename Value >
   __cuda_callable__
   GridEntityBase( const std::initializer_list< Value >& elems )
   : coordinates( elems )
   {}

   __cuda_callable__
   GridEntityBase( const CoordinatesType& coordinates, IndexType totalOrientationIdx = 0 )
   : coordinates( coordinates )
   {}

   __cuda_callable__
   GridEntityBase( IndexType orientationIdx ) {}

   __cuda_callable__
   void
   setCoordinates( const CoordinatesType& coordinates )
   {
      this->coordinates = coordinates;
   }

   __cuda_callable__
   const CoordinatesType&
   getCoordinates() const
   {
      return coordinates;
   }

   __cuda_callable__
   CoordinatesType&
   getCoordinates()
   {
      return coordinates;
   }

   __cuda_callable__
   void
   setTotalOrientationIndex( IndexType idx )
   {
      TNL_ASSERT_EQ( idx, 0, "Wrong total orientation index. Only zero value is allowed for vertices." );
   }

   __cuda_callable__
   IndexType
   getTotalOrientationIndex() const
   {
      return 0;
   }

   __cuda_callable__
   IndexType
   getOrientationIndex() const
   {
      return 0;
   }

protected:
   CoordinatesType coordinates;
};

template< typename Grid, int GridDimension >
class GridEntityBase< Grid, GridDimension, GridDimension >
{
public:
   static constexpr int
   getMeshDimension()
   {
      return Grid::getMeshDimension();
   }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using CoordinatesType = typename Grid::CoordinatesType;

   using EntitiesOrientations = GridEntitiesOrientations< getMeshDimension() >;

   __cuda_callable__
   GridEntityBase() = default;

   template< typename Value >
   __cuda_callable__
   GridEntityBase( const std::initializer_list< Value >& elems )
   : coordinates( elems )
   {}

   __cuda_callable__
   GridEntityBase( IndexType orientationIdx ) {}

   __cuda_callable__
   GridEntityBase( const CoordinatesType& coordinates, IndexType totalOrientationIdx = 0 )
   : coordinates( coordinates )
   {}

   __cuda_callable__
   void
   setCoordinates( const CoordinatesType& coordinates )
   {
      this->coordinates = coordinates;
   }

   __cuda_callable__
   const CoordinatesType&
   getCoordinates() const
   {
      return coordinates;
   }

   __cuda_callable__
   CoordinatesType&
   getCoordinates()
   {
      return coordinates;
   }

   __cuda_callable__
   void
   setTotalOrientationIndex( IndexType idx )
   {
      TNL_ASSERT_EQ(
         idx,
         ( 1 << getMeshDimension() ) - 1,
         "Wrong total orientation index. Only 2^GridDimension-1 value is allowed for cells." );
   }

   __cuda_callable__
   IndexType
   getTotalOrientationIndex() const
   {
      return ( 1 << getMeshDimension() ) - 1;
   }

   __cuda_callable__
   IndexType
   getOrientationIndex() const
   {
      return 0;
   }

protected:
   CoordinatesType coordinates;
};

}  // namespace Meshes
}  // namespace TNL
