// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/Grid.h>

namespace TNL {
namespace Meshes {

template< int, int, int >
class NeighbourGridEntityGetter;

template< class >
class BoundaryGridEntityChecker;

template< class >
class GridEntityCenterGetter;

template< class Grid, int EntityDimension >
class GridEntity
{
public:
   using GridType = Grid;
   using Index = typename Grid::IndexType;
   using Device = typename Grid::DeviceType;
   using Real = typename Grid::RealType;

   using CoordinatesType = typename Grid::CoordinatesType;
   using PointType = typename Grid::PointType;

   constexpr static int meshDimension = Grid::getMeshDimension();
   constexpr static int entityDimension = EntityDimension;

   /////////////////////////////
   // Compatability with meshes
   constexpr static int getEntityDimension() {
      return entityDimension;
   }
   /////////////////////////////

   __cuda_callable__
   inline GridEntity( const Grid& grid )
   : grid( grid ), coordinates( 0 )
   {
      this->basis = grid.template getBasis<EntityDimension>(0);
      this->orientation = 0;
      this->refresh();
   }

   __cuda_callable__
   inline GridEntity( const Grid& grid, const CoordinatesType& coordinates )
   : grid( grid ), coordinates( coordinates )
   {
      basis = grid.template getBasis<EntityDimension>(0);
      orientation = 0;
      refresh();
   }

   __cuda_callable__
   inline GridEntity( const Grid& grid, const CoordinatesType& coordinates, const CoordinatesType& basis, const Index orientation )
   : grid( grid ), coordinates( coordinates ), basis( basis ), orientation( orientation )
   {
      refresh();
   }

   __cuda_callable__
   inline const CoordinatesType&
   getCoordinates() const;

   __cuda_callable__
   inline CoordinatesType&
   getCoordinates();

   __cuda_callable__
   inline void
   setCoordinates( const CoordinatesType& coordinates );

   /***
    * @brief - Recalculates entity index.
    *
    * @warning - Call this method every time the coordinates are changed
    */
   __cuda_callable__
   inline void
   refresh();

   /**
    * @brief Get the entity index in global grid
    */
   __cuda_callable__
   inline Index
   getIndex() const;

   /**
    * @brief Tells, if entity is boundary
    */
   __cuda_callable__
   inline bool
   isBoundary() const;

   /**
    * @brief Returns, the center of the entity
    */
   __cuda_callable__
   inline const PointType
   getCenter() const;

   /**
    * @brief Returns, the measure (volume) of the entity
    */
   __cuda_callable__
   inline Real
   getMeasure() const;

   __cuda_callable__
   inline const Grid&
   getMesh() const;

   __cuda_callable__
   inline void
   setBasis( const CoordinatesType& orientation );

   /**
    * @brief Returns, the entity basis
    */
   __cuda_callable__
   inline CoordinatesType
   getBasis() const;

   /**
    * @brief Returns, the entity orientation
    *
    * Orientation is always paired with the basis. In other words, if orientations, entityDimensions and dimensions are equal,
    * then bases are equal also.
    */
   __cuda_callable__
   inline Index
   getOrientation() const;

   __cuda_callable__
   inline void
   setOrientation( const Index orientation );
   /**
    * @brief Returns, the neighbour entity
    *
    * @warning - In case, if the parent entity orientation is greater than possible orientations of neighbour entity,
    *            then orientation is reduces. For example, 3-d cell neighbour of edge with orientaiton 1, will have
    *            orientation 0
    * @warning - You should refresh index manually
    */
   template< int Dimension, int... Steps, std::enable_if_t< sizeof...( Steps ) == Grid::getMeshDimension(), bool > = true >
   __cuda_callable__
   inline GridEntity< Grid, Dimension >
   getNeighbourEntity() const;

   /**
    * @brief Returns, the neighbour entity
    *
    * @warning - You should refresh index manually
    */
   template< int Dimension,
             int Orientation,
             int... Steps,
             std::enable_if_t< sizeof...( Steps ) == Grid::getMeshDimension(), bool > = true >
   __cuda_callable__
   inline GridEntity< Grid, Dimension >
   getNeighbourEntity() const;

   /**
    * @brief Returns, the neighbour entity
    *
    * @warning - In case, if the parent entity orientation is greater than possible orientations of neighbour entity,
    *            then orientation is reduces. For example, 3-d cell neighbour of edge with orientaiton 1, will have
    *            orientation 0
    * @warning - You should refresh index manually
    */
   template< int Dimension >
   __cuda_callable__
   inline GridEntity< Grid, Dimension >
   getNeighbourEntity( const CoordinatesType& offset ) const;

   /**
    * @brief Returns, the neighbour entity
    *
    * @warning - You should refresh index manually
    */
   template< int Dimension, int Orientation >
   __cuda_callable__
   inline GridEntity< Grid, Dimension >
   getNeighbourEntity( const CoordinatesType& offset ) const;

   PointType getPoint() const { return this->grid.getSpaceSteps() * this->getCoordinates(); };

protected:
   const Grid& grid;

   Index index;
   CoordinatesType coordinates;
   CoordinatesType basis;
   Index orientation;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/GridEntity.hpp>
