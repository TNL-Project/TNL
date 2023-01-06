// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityBase.h>
#include <TNL/Meshes/GridEntitiesOrientations.h>

namespace TNL {
namespace Meshes {

template< int, int, int >
class NeighbourGridEntityGetter;

template< class >
class BoundaryGridEntityChecker;

template< class >
class GridEntityCenterGetter;

/**
 * \brief Structure describing a grid entity i.e., grid cells, faces, edges, vertexes and so on.
 *
 * \tparam Grid is a type of grid the entity belongs to.
 * \tparam EntityDimension is a dimensions of the grid entity.
 */
template< class Grid, int EntityDimension >
class GridEntity : public Grid::CoordinatesType, GridEntityBase< Grid, Grid::getMeshDimension(), EntityDimension >
{
public:
   using GridEntityBaseType = GridEntityBase< Grid, Grid::getMeshDimension(), EntityDimension >;

   /**
    * \brief Type of grid the entity belongs to.
    */
   using GridType = Grid;

   /**
    * \brief Type of floating point numbers.
    */
   using RealType = typename Grid::RealType;

   /**
    * \brief Device to be used for execution of operations with the grid.
    */
   using DeviceType = typename Grid::DeviceType;

   /**
    * \brief Type for indexing of the grid entities.
    *
    */
   using IndexType = typename Grid::IndexType;

   /**
    * \brief Type of grid entities coordinates.
    */
   using CoordinatesType = typename Grid::CoordinatesType;

   /**
    * \brief Type of world coordinates.
    */
   using PointType = typename Grid::PointType;

   using EntitiesOrientations = GridEntitiesOrientations< Grid::getMeshDimension() >;

   using NormalsType = typename GridType::NormalsType;

   /**
    * \brief Getter of the dimension of the grid.
    *
    * \return dimension of the grid.
    */
   constexpr static int
   getMeshDimension();

   /**
    * \brief Getter of the dimensions of the grid entity.
    *
    * \return dimensions of the grid entity.
    */
   constexpr static int
   getEntityDimension();

   __cuda_callable__
   GridEntity();

   __cuda_callable__
   GridEntity( const CoordinatesType& c );

   /**
    * brief Constructor with a grid reference.
    *
    * param grid is a reference on a grid the entity belongs to.
    */

   /*template< typename... Indexes, std::enable_if_t< ( Grid::getMeshDimension() > 1 ) && sizeof...( Indexes ) == Grid::getMeshDimension(), bool > = true >
   // NOTE: without __cuda_callable__, nvcc 11.8 would complain that it is __host__ only, even though it is constexpr
   __cuda_callable__
   GridEntity( Indexes&&... indexes );*/

   template< typename Value >
   __cuda_callable__ // TODO: This is only temporary
   GridEntity( const std::initializer_list< Value >& elems );

   /**
    * \brief Constructor with a grid reference and grid entity coordinates.
    *
    * \param grid is a reference on a grid the entity belongs to.
    * \param coordinates are coordinates of the grid entity.
    */
   __cuda_callable__
   GridEntity( const Grid& grid, const CoordinatesType& coordinates );

   /**
    * \brief Constructor with a grid reference, grid entity coordinates and entity normals.
    *
    * Entity normals define the grid entity orientation.
    *
    * \param grid is a reference on a grid the entity belongs to.
    * \param coordinates are coordinates of the grid entity.
    * \param normals is a vector of packed normal vectors to the grid entity.
    */
   __cuda_callable__
   GridEntity( const Grid& grid, const CoordinatesType& coordinates, const NormalsType& normals );

   /**
    * \brief Constructor with a grid reference, grid entity coordinates, entity normals and index of entity orientation.
    *
    * Entity normals define the grid entity orientation.
    * Index of entity orientation is rather internal information. Constructor without this parameter may be used preferably.
    * The index can be computed using the method \ref TNL::Meshes::Grid::getOrientation.
    *
    * \param grid is a reference on a grid the entity belongs to.
    * \param coordinates are coordinates of the grid entity.
    * \param orientation is total ori index of the grid entity orientation.
    */
   __cuda_callable__
   GridEntity( const Grid& grid, const CoordinatesType& coordinates, IndexType orientationIndex );

   /**
    * \brief Constructor with a grid reference and grid entity index.
    *
    * \param grid is a reference on a grid the entity belongs to.
    * \param entityIdx is index of the grid entity.
    */
   __cuda_callable__
   GridEntity( const Grid& grid, IndexType entityIdx );

   /**
    * \brief Getter of the grid entity coordinates for constant instances.
    *
    * \return grid entity coordinates in a form of constant reference.
    */
   __cuda_callable__
   const CoordinatesType&
   getCoordinates() const;

   /**
    * \brief Getter of the grid entity coordinates for non-constant instances.
    *
    * \return grid entity coordinates in a form of a reference.
    */
   __cuda_callable__
   CoordinatesType&
   getCoordinates();

   /**
    * \brief Setter of the grid entity coordinates.
    *
    * \param coordinates are new coordinates of the grid entity.
    */
   __cuda_callable__
   void
   setCoordinates( const CoordinatesType& coordinates );

   /***
    * \brief Recalculates entity index.
    *
    * \warning Call this method every time the coordinates are changed.
    */
   __cuda_callable__
   void
   refresh();

   /**
    * \brief Get the entity index in the grid.
    *
    * \return the grid entity index in the grid.
    */
   __cuda_callable__
   const IndexType&
   getIndex() const;

   /**
    * \brief Tells, if the entity is boundary entity.
    *
    * \return `true` if the entity is a boundary entity and `false` otherwise.
    */
   __cuda_callable__
   bool
   isBoundary() const;

   /**
    * \brief Returns the center of the grid entity.
    *
    * \return the centre of the grid entity.
    */
   __cuda_callable__
   PointType
   getCenter() const;

   /**
    * \brief Returns the measure (length, surface or volume) of the grid entity.
    *
    * \return the measure of the grid entity.
    */
   __cuda_callable__
   RealType
   getMeasure() const;

   /**
    * \brief Returns reference to the grid the grid entity belongs to.
    *
    * \return reference to the grid the grid entity belongs to.
    */
   __cuda_callable__
   const Grid&
   getMesh() const;

   //__cuda_callable__
   //GridEntityOrientationType& getOrientation();

   //__cuda_callable__
   //const GridEntityOrientationType& getOrientation() const;

   /**
    * \brief Setter for the packed normals vector of the grid entity.
    *
    * This vector defines the orienation of the grid entity.
    *
    * \param normals is a vector of packed normal vectors to the grid entity.
    */
   __cuda_callable__
   void
   setNormals( const NormalsType& normals );

   /**
    * \brief Returns the packed normals vector of the grid entity.
    */
   __cuda_callable__
   const NormalsType
   getNormals() const;

   /**
    * \brief Getter of the basis vector.
    *
    * The basis vector has one for each axis along which the grid entity has non-zero length.
    *
    * The basis vector is not stored explicitly in the grid entity and it is computed on the fly.
    *
    * \return basis vector.
    */
   __cuda_callable__
   NormalsType
   getBasis() const;

   /**
    * \brief Returns index of the entity orientation
    *
    * Orientation is always paired with the normals. In other words, if orientations, entity dimensions and dimensions are
    * equal, then normals are equal also.
    */
   __cuda_callable__
   IndexType
   getOrientationIndex() const;

   /**
    * brief Setter of the grid entity orientation index.
    *
    * This is rather internal information. The index can be computed using the method \ref TNL::Meshes::Grid::getOrientation.
    *
    * param orientation is a index of the grid entity orientation.
    */
   __cuda_callable__
   void setOrientationIndex( IndexType orientationIndex );

   __cuda_callable__
   IndexType getTotalOrientationIndex() const;

   __cuda_callable__
   void setTotalOrientationIndex( IndexType totalOrientationIndex );


   /**
    * \brief Returns the neighbour grid entity. TODO: FIX - CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    *
    * \tparam Dimension is a dimension of the neighbour grid entity.
    * \param offset is a offset of coordinates of the neighbour entity relative to this grid entity.
    * \warning In case the parent entity orientation is greater than possible orientations of neighbour entity,
    *            then orientation is reduces. For example, 3-D cell neighbour of edge with orientation 1, will have
    *            orientation 0.
    * \return neighbour grid entity.
    */
   __cuda_callable__
   GridEntity< Grid, EntityDimension >
   getNeighbourEntity( const CoordinatesType& offset ) const;

   __cuda_callable__
   IndexType
   getNeighbourEntityIndex( const CoordinatesType& offset ) const;


   /**
    * brief Returns the neighbour grid entity. TODO: FIX - CHECK !!!!!!!!!!!!!!!!!!!!!!!!!!!!
    *
    * tparam Dimension is a dimension of the neighbour grid entity.
    * tparam Orientation is an orientation index of the grid entity.
    * param offset is a offset of coordinates of the neighbour entity relative to this grid entity.
    * return neighbour grid entity.
    */
   template< int NeighbourEntityDimension >
   __cuda_callable__
   GridEntity< Grid, NeighbourEntityDimension >
   getNeighbourEntity( const CoordinatesType& offset,
                       const NormalsType& neighbourEntityOrientation ) const;

   /**
    * \brief
    *
    * Note: Entity orientation is given by its index for performance reason.
    *
    * \tparam NeighbourEntityDimension
    * \param offset
    * \param neighbourEntityOrientation
    * \return __cuda_callable__
    */
   template< int NeighbourEntityDimension >
   __cuda_callable__
   IndexType
   getNeighbourEntityIndex( const CoordinatesType& offset,
                            IndexType neighbourEntityOrientation ) const;

   /**
    * \brief Returns the point at the origin of the grid entity.
    *
    * \return the point at the origin of the grid entity.
    */
   PointType
   getPoint() const;

   __cuda_callable__
   void
   setGrid( const Grid& );

   __cuda_callable__
   void
   setMesh( const Grid& );


   /**
    * \brief Returns a reference on the grid the grid entity belongs to.
    *
    * \return a reference on the grid the grid entity belongs to.
    */
   __cuda_callable__
   const Grid&
   getGrid() const;

   bool operator>( const GridEntity& e ) const {
      return Containers::Expressions::StaticComparison< GridEntity, GridEntity >::GT( *this, e );
   }

protected:
   const Grid* grid;

   IndexType index;
   //GridEntityOrientationType orientation;
};

/**
 * \brief Overloaded insertion operator for printing a grid entity to output stream.
 *
 * \tparam Grid type of grid the grid entity belongs to.
 * \tparam EntityDimension dimension of the grid entity.
 * \param str insertion operator.
 * \param entity instance of the grid entity.
 * \return std::ostream& reference to the insertion operator.
 */
template< class Grid, int EntityDimension >
std::ostream&
operator<<( std::ostream& str, const GridEntity< Grid, EntityDimension >& entity );

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridEntity.hpp>
