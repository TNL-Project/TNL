// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber, Yury Hayeu

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/NDGrid.h>

namespace TNL {
namespace Meshes {

template< class, int >
class GridEntity;

/**
 * \brief Orthogonal n-dimensional grid.
 *
 * This data structure represents regular orthogonal numerical mesh. It provides indexing of mesh
 * entities like vertexes, edges, faces or cells together with parallel traversing of all, interior
 * or boundary mesh entities.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of the grid.
 */
template< int Dimension_, typename Real = double, typename Device = Devices::Host, typename Index = int >
class Grid : public NDGrid< Dimension_, Real, Device, Index >
{
public:
   using BaseType = NDGrid< Dimension_, Real, Device, Index >;

   /**
    * \brief Dimension of the grid.
    */
   static constexpr int Dimension = Dimension_;

   /**
    * \brief Type of the floating point numbers.
    */
   using RealType = Real;

   /**
    * \brief Device to be used for the execution of grid operations.
    */
   using DeviceType = Device;

   /**
    * \brief Type for indexing of the mesh entities of the grid.
    */
   using IndexType = Index;

   /**
    * \brief Type for indexing of the mesh entities of the grid.
    *
    * This is for compatiblity with unstructured meshes.
    */
   using GlobalIndexType = Index;

   /**
    * \brief Type for mesh entities cordinates within the grid.
    */
   using CoordinatesType = typename BaseType::CoordinatesType;

   /**
    * \brief Type for world coordinates.
    */
   using PointType = typename BaseType::PointType;

   using EntitiesCounts = typename BaseType::EntitiesCounts;

   /**
    * \brief Alias for grid entities with given dimension.
    *
    * \tparam EntityDimension is dimensions of the grid entity.
    */
   template< int EntityDimension >
   using EntityType = GridEntity< Grid, EntityDimension >;

   using Vertex = EntityType< 0 >;
   using Edge = EntityType< 1 >;
   using Face = EntityType< Dimension - 1 >;
   using Cell = EntityType< Dimension >;

   /**
    * \brief Returns the dimension of grid
    */
   static constexpr int
   getMeshDimension()
   {
      return Dimension;
   };

   Grid() = default;

   template< typename... Dimensions,
             std::enable_if_t< Templates::conjunction_v< std::is_convertible< Index, Dimensions >... >, bool > = true,
             std::enable_if_t< sizeof...( Dimensions ) == Dimension, bool > = true >
   Grid( Dimensions... dimensions ) : NDGrid< Dimension, Real, Device, Index >( dimensions... ){};


  // TODO: !!!!!!!!!!!!!!!!!!!!!!!!!
   void setLocalBegin( const CoordinatesType& localBegin ){};

   void setLocalEnd( const CoordinatesType& localEnd ){};

   void setInteriorBegin( const CoordinatesType& localBegin ){};

   void setInteriorEnd( const CoordinatesType& localEnd ){};


   Real getCellMeasure() const { return 0.0; };

   template< typename EntityType >
   Index getEntitiesCount() const { return 0;};


   template< int EntityDimension >
   Index getEntitiesCount() const { return 0;};

   Index getEntitiesCount( IndexType entityDim ) const { return 0;};

   template< typename EntityType >
   EntityType getEntity( const IndexType& entityIdx ) const { return EntityType( *this, entityIdx );};

   template< int EntityDimension >
   EntityType< EntityDimension > getEntity( const IndexType& entityIdx ) const { return EntityType< EntityDimension >( *this, entityIdx );};
   // TODO end



   /**
    * \brief Iterate over all mesh entities with given dimension and perform given lambda function
    *    on each of them.
    *
    * \tparam EntityDimension is dimension of the grid entites.
    * \tparam Func is lambda to be performed on each grid enitty.
    *
    * \param func is an instance of the lambda function to be performed on each grid entity.
    *     It is supposed to have the following form:
    *
    * ```
    * auto func = [=] __cuda_callable__( const typename Grid< Dimension, Real, Device, Index >::template EntityType< EntityDimension >&entity ) mutable {};
    * ```
    * where \ref entity represents given grid entity. See \ref TNL::Meshes::GridEntity.
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forAll( Func func, FuncArgs... args ) const;

   /**
    * \brief Traverser all elements in rect
    * \param from - bottom left anchor of traverse rect
    * \param to - top right anchor of traverse rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forAll( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

   /**
    * \brief Traverser interior elements in rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forInterior( Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser interior elements
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forInterior( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser boundary elements in rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forBoundary( Func func, FuncArgs... args ) const;

   /**
    * @brief Traverser boundary elements in rect
    * @param from - bottom left anchor of traverse rect
    * @param to - top right anchor of traverse rect
    */
   template< int EntityDimension, typename Func, typename... FuncArgs >
   inline void
   forBoundary( const CoordinatesType& from, const CoordinatesType& to, Func func, FuncArgs... args ) const;

};

/**
 * \brief Serialization of the grid structure.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers of grid.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of grid.
 * \param str is output stream.
 * \param grid is an instance of grid.
 * \return std::ostream& is reference on the input stream.
 */
template< int Dimension, typename Real, typename Device, typename Index >
std::ostream&
operator<<( std::ostream& str, const Grid< Dimension, Real, Device, Index >& grid )
{
   TNL::Logger logger( 100, str );

   grid.writeProlog( logger );

   return str;
}

/**
 * \brief Comparison operator for grid.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers of grid.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of grid.
 * \param lhs is an instance of one grid.
 * \param rhs is an instance of another grid.
 * \return true if both grids are equal.
 * \return false if the grids are different.
 */
template< int Dimension, typename Real, typename Device, typename Index >
bool
operator==( const Grid< Dimension, Real, Device, Index >& lhs, const Grid< Dimension, Real, Device, Index >& rhs )
{
   return lhs.getDimensions() == rhs.getDimensions() && lhs.getOrigin() == rhs.getOrigin()
       && lhs.getProportions() == rhs.getProportions();
}

/**
 * \brief Comparison operator for grid.
 *
 * \tparam Dimension is grid dimension.
 * \tparam Real is type of the floating point numbers of grid.
 * \tparam Device is the device to be used for the execution of grid operations.
 * \tparam Index is type for indexing of the mesh entities of grid.
 * \param lhs is an instance of one grid.
 * \param rhs is an instance of another grid.
 * \return true if the grids are different.
 * \return false if both grids are equal.
 */
template< int Dimension, typename Real, typename Device, typename Index >
bool
operator!=( const Grid< Dimension, Real, Device, Index >& lhs, const Grid< Dimension, Real, Device, Index >& rhs )
{
   return ! ( lhs == rhs );
}

}  // namespace Meshes
}  // namespace TNL

//#include <TNL/Meshes/GridDetails/Grid1D.h>
//#include <TNL/Meshes/GridDetails/Grid2D.h>
//#include <TNL/Meshes/GridDetails/Grid3D.h>
#include <TNL/Meshes/GridDetails/Grid.hpp>
