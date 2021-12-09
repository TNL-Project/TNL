// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/StaticVector.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntitiesStorage.h>

namespace TNL {
namespace Meshes {

template< typename GridEntity, int NeighborEntityDimension, typename StencilStorage >
class NeighborGridEntityGetter;

template< typename GridEntityType >
class BoundaryGridEntityChecker;

template< typename GridEntityType >
class GridEntityCenterGetter;

template< typename Grid, int EntityDimension, typename Config >
class GridEntity
{};

template< int Dimension, typename Real, typename Device, typename Index, int EntityDimension, typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >
{
   public:

      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef Config ConfigType;

      constexpr static int getMeshDimension() { return GridType::getMeshDimension(); };

      constexpr static int getEntityDimension() { return EntityDimension; };

      typedef Containers::StaticVector< getMeshDimension(), IndexType > EntityOrientationType;
      typedef typename GridType::PointType PointType;

      template< int NeighborEntityDimension = getEntityDimension() >
      using NeighborEntities =
         NeighborGridEntityGetter<
            GridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           EntityDimension,
                           Config >,
            NeighborEntityDimension >;

   template< int NeighborEntityDimension = getEntityDimension() >
   using NeighborEntities =
      NeighborGridEntityGetter< GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, EntityDimension, Config >,
                                NeighborEntityDimension >;

      __cuda_callable__ inline
      GridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation);

   __cuda_callable__
   inline GridEntity( const GridType& grid,
                      const CoordinatesType& coordinates,
                      const EntityOrientationType& orientation,
                      const EntityBasisType& basis );

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
    * Call this method every time the coordinates are changed
    * to recompute the mesh entity index. The reason for this strange
    * mechanism is a performance.
    */
   __cuda_callable__
   inline
      // void setIndex( IndexType entityIndex );
      void
      refresh();

   __cuda_callable__
   inline Index
   getIndex() const;

   __cuda_callable__
   inline const EntityOrientationType&
   getOrientation() const;

      __cuda_callable__ inline
      bool isBoundaryEntity() const;

   __cuda_callable__
   inline bool
   isBoundaryEntity() const;

   __cuda_callable__
   inline PointType
   getCenter() const;

   __cuda_callable__
   inline const RealType&
   getMeasure() const;

   __cuda_callable__
   inline const GridType&
   getMesh() const;

protected:
   const GridType& grid;

   IndexType entityIndex;

   CoordinatesType coordinates;

   EntityOrientationType orientation;

      //__cuda_callable__ inline
      //GridEntity();

   friend class BoundaryGridEntityChecker< GridEntity >;

   friend class GridEntityCenterGetter< GridEntity >;
};

/****
 * Specializations for cells
 */
template< int Dimension, typename Real, typename Device, typename Index, typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, Dimension, Config >
{
   public:

      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::PointType PointType;
      typedef Config ConfigType;

      constexpr static int getMeshDimension() { return GridType::getMeshDimension(); };

      constexpr static int getEntityDimension() { return getMeshDimension(); };

      typedef Containers::StaticVector< getMeshDimension(), IndexType > EntityOrientationType;

      template< int NeighborEntityDimension = getEntityDimension() >
      using NeighborEntities =
         NeighborGridEntityGetter<
            GridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           Dimension,
                           Config >,
            NeighborEntityDimension >;


      __cuda_callable__ inline
      GridEntity( const GridType& grid );

      __cuda_callable__ inline
      GridEntity( const GridType& grid,
                  const CoordinatesType& coordinates,
                  const EntityOrientationType& orientation = EntityOrientationType( ( Index ) 0 ) );

      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const;

      __cuda_callable__ inline
      CoordinatesType& getCoordinates();

      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates );

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      //void setIndex( IndexType entityIndex );
      void refresh();

      __cuda_callable__ inline
      Index getIndex() const;

      __cuda_callable__ inline
      const EntityOrientationType getOrientation() const;

      __cuda_callable__ inline
      void setOrientation( const EntityOrientationType& orientation ){};

      __cuda_callable__ inline
      bool isBoundaryEntity() const;

      __cuda_callable__ inline
      PointType getCenter() const;

      __cuda_callable__ inline
      const RealType& getMeasure() const;

      __cuda_callable__ inline
      const PointType& getEntityProportions() const;

      __cuda_callable__ inline
      const GridType& getMesh() const;

   CoordinatesType coordinates;

   NeighborGridEntitiesStorageType neighborEntitiesStorage;

   //__cuda_callable__ inline
   // GridEntity();

   friend class BoundaryGridEntityChecker< GridEntity >;

      //__cuda_callable__ inline
      //GridEntity();

      friend class BoundaryGridEntityChecker< GridEntity >;

      friend class GridEntityCenterGetter< GridEntity >;
};

/****
 * Specialization for vertices
 */
template< int Dimension, typename Real, typename Device, typename Index, typename Config >
class GridEntity< Meshes::Grid< Dimension, Real, Device, Index >, 0, Config >
{
   public:

      typedef Meshes::Grid< Dimension, Real, Device, Index > GridType;
      typedef GridType MeshType;
      typedef typename GridType::RealType RealType;
      typedef typename GridType::IndexType IndexType;
      typedef typename GridType::CoordinatesType CoordinatesType;
      typedef typename GridType::PointType PointType;
      typedef Config ConfigType;

      constexpr static int getMeshDimension() { return GridType::getMeshDimension(); };

      constexpr static int getEntityDimension() { return 0; };

      typedef Containers::StaticVector< getMeshDimension(), IndexType > EntityOrientationType;

      template< int NeighborEntityDimension = getEntityDimension() >
      using NeighborEntities =
         NeighborGridEntityGetter<
            GridEntity< Meshes::Grid< Dimension, Real, Device, Index >,
                           0,
                           Config >,
            NeighborEntityDimension >;


      __cuda_callable__ inline
      GridEntity( const GridType& grid );

      __cuda_callable__ inline
      GridEntity( const GridType& grid,
                     const CoordinatesType& coordinates,
                     const EntityOrientationType& orientation = EntityOrientationType( ( Index ) 0 ) );

      __cuda_callable__ inline
      const CoordinatesType& getCoordinates() const;

      __cuda_callable__ inline
      CoordinatesType& getCoordinates();

      __cuda_callable__ inline
      void setCoordinates( const CoordinatesType& coordinates );

      /***
       * Call this method every time the coordinates are changed
       * to recompute the mesh entity index. The reason for this strange
       * mechanism is a performance.
       */
      __cuda_callable__ inline
      //void setIndex( IndexType entityIndex );
      void refresh();

      __cuda_callable__ inline
      Index getIndex() const;

   IndexType entityIndex;

   CoordinatesType coordinates;

      __cuda_callable__ inline
      bool isBoundaryEntity() const;

      __cuda_callable__ inline
      PointType getCenter() const;

      // compatibility with meshes, equivalent to getCenter
      __cuda_callable__ inline
      PointType getPoint() const;

      __cuda_callable__ inline
      const RealType getMeasure() const;

      __cuda_callable__ inline
      PointType getEntityProportions() const;

      __cuda_callable__ inline
      const GridType& getMesh() const;

   protected:

      const GridType& grid;

      IndexType entityIndex;

      CoordinatesType coordinates;

      friend class BoundaryGridEntityChecker< GridEntity >;

      friend class GridEntityCenterGetter< GridEntity >;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/GridEntity_impl.h>
