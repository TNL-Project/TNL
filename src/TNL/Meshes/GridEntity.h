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
   using IndexType = typename Grid::IndexType;
   using DeviceType = typename Grid::DeviceType;
   using RealType = typename Grid::RealType;

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
   GridEntity( const Grid& grid )
   : grid( grid ), coordinates( 0 )
   {
      this->normals = grid.template getNormals<EntityDimension>(0);
      this->orientation = 0;
      this->refresh();
   }

   __cuda_callable__
   GridEntity( const Grid& grid, const CoordinatesType& coordinates )
   : grid( grid ), coordinates( coordinates )
   {
      normals = grid.template getNormals<EntityDimension>(0);
      orientation = 0;
      refresh();
   }

   __cuda_callable__
   GridEntity( const Grid& grid, const CoordinatesType& coordinates, const CoordinatesType& normals )
   : grid( grid ), coordinates( coordinates ), normals( normals ), 
      orientation( grid.template getOrientation< EntityDimension >( normals ) )
   {
      refresh();
   }

   __cuda_callable__
   GridEntity( const Grid& grid, const CoordinatesType& coordinates, const CoordinatesType& normals, 
      const IndexType orientation )
   : grid( grid ), coordinates( coordinates ), normals( normals ), orientation( orientation )
   {
      refresh();
   }

   __cuda_callable__
   const CoordinatesType& getCoordinates() const;

   __cuda_callable__
   CoordinatesType& getCoordinates();

   __cuda_callable__
   void setCoordinates( const CoordinatesType& coordinates );

   /***
    * @brief - Recalculates entity index.
    *
    * @warning - Call this method every time the coordinates are changed
    */
   __cuda_callable__
   void refresh();

   /**
    * @brief Get the entity index in global grid
    */
   __cuda_callable__
   IndexType getIndex() const;

   /**
    * @brief Tells, if entity is boundary
    */
   __cuda_callable__
   bool isBoundary() const;

   /**
    * @brief Returns, the center of the entity
    */
   __cuda_callable__
   const PointType getCenter() const;

   /**
    * @brief Returns, the measure (volume) of the entity
    */
   __cuda_callable__
   RealType getMeasure() const;

   __cuda_callable__
   const Grid& getMesh() const;

   __cuda_callable__
   void setNormals( const CoordinatesType& orientation );

   /**
    * @brief Returns, the entity normals
    */
   __cuda_callable__
   const CoordinatesType& getNormals() const;

   __cuda_callable__
   CoordinatesType getBasis() const;

   /**
    * @brief Returns, the entity orientation
    *
    * Orientation is always paired with the normals. In other words, if orientations, entityDimensions and dimensions are equal,
    * then normals are equal also.
    */
   __cuda_callable__
   IndexType getOrientation() const;

   __cuda_callable__
   inline void
   setOrientation( const IndexType orientation );
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

   __cuda_callable__
   const Grid& getGrid() const;

protected:
   const Grid& grid;

   IndexType index;
   CoordinatesType coordinates;
   CoordinatesType normals;
   IndexType orientation;
};

template< class Grid, int EntityDimension >
std::ostream& operator<<( std::ostream& str, const GridEntity< Grid, EntityDimension >& entity )
{
   str << "Entity dimension = " << EntityDimension << " coordinates = " << entity.getCoordinates() << " normals = " << entity.getNormals()
       << " index = " << entity.getIndex() << " orientation = " << entity.getOrientation();
   return str;
}

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/GridEntity.hpp>
