// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridEntityTopology.h>
#include <TNL/Meshes/GridDetails/GridEntityGetter.h>
#include <TNL/Meshes/GridDetails/NeighborGridEntityGetter.h>
#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridEntityConfig.h>

namespace TNL {
namespace Meshes {

template< typename Real, typename Device, typename Index >
class Grid< 3, Real, Device, Index >
{
public:
   using RealType = Real;
   using DeviceType = Device;
   using GlobalIndexType = Index;
   using PointType = Containers::StaticVector< 3, Real >;
   using CoordinatesType = Containers::StaticVector< 3, Index >;

   // TODO: deprecated and to be removed (GlobalIndexType shall be used instead)
   using IndexType = Index;

   static constexpr int
   getMeshDimension()
   {
      return 3;
   };

   template< int EntityDimension, typename Config = GridEntityCrossStencilStorage< 1 > >
   using EntityType = GridEntity< Grid, EntityDimension, Config >;

   using Cell = EntityType< getMeshDimension(), GridEntityCrossStencilStorage< 1 > >;
   using Face = EntityType< getMeshDimension() - 1 >;
   using Edge = EntityType< 1 >;
   using Vertex = EntityType< 0 >;

   /**
    * \brief See Grid1D::Grid().
    */
   Grid() = default;

   Grid( Index xSize, Index ySize, Index zSize );

   // empty destructor is needed only to avoid crappy nvcc warnings
   ~Grid() = default;

   /**
    * \brief Sets the size of dimensions.
    * \param xSize Size of dimesion x.
    * \param ySize Size of dimesion y.
    * \param zSize Size of dimesion z.
    */
   void
   setDimensions( Index xSize, Index ySize, Index zSize );

   /**
    * \brief See Grid1D::setDimensions( const CoordinatesType& dimensions ).
    */
   void
   setDimensions( const CoordinatesType& dimensions );

   /**
    * \brief See Grid1D::getDimensions().
    */
   __cuda_callable__
   const CoordinatesType&
   getDimensions() const;

   void
   setLocalBegin( const CoordinatesType& begin );

   __cuda_callable__
   const CoordinatesType&
   getLocalBegin() const;

   void
   setLocalEnd( const CoordinatesType& end );

   __cuda_callable__
   const CoordinatesType&
   getLocalEnd() const;

   void
   setInteriorBegin( const CoordinatesType& begin );

   __cuda_callable__
   const CoordinatesType&
   getInteriorBegin() const;

   void
   setInteriorEnd( const CoordinatesType& end );

   __cuda_callable__
   const CoordinatesType&
   getInteriorEnd() const;

   /**
    * \brief See Grid1D::setDomain().
    */
   void
   setDomain( const PointType& origin, const PointType& proportions );

   /**
    * \brief See Grid1D::setOrigin()
    */
   void
   setOrigin( const PointType& origin );

   /**
    * \brief See Grid1D::getOrigin().
    */
   __cuda_callable__
   inline const PointType&
   getOrigin() const;

   /**
    * \brief See Grid1D::getProportions().
    */
   __cuda_callable__
   inline const PointType&
   getProportions() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam EntityDimension Integer specifying dimension of the entity.
    */
   template< int EntityDimension >
   __cuda_callable__
   IndexType
   getEntitiesCount() const;

   /**
    * \brief Gets number of entities in this grid.
    * \tparam Entity Type of the entity.
    */
   template< typename Entity >
   __cuda_callable__
   IndexType
   getEntitiesCount() const;

   /**
    * \brief See Grid1D::getEntity().
    */
   template< typename Entity >
   __cuda_callable__
   inline Entity
   getEntity( const IndexType& entityIndex ) const;

   /**
    * \brief See Grid1D::getEntityIndex().
    */
   template< typename Entity >
   __cuda_callable__
   inline Index
   getEntityIndex( const Entity& entity ) const;

   /**
    * \brief See Grid1D::getSpaceSteps().
    */
   __cuda_callable__
   inline const PointType&
   getSpaceSteps() const;

   /**
    * \brief See Grid1D::setSpaceSteps().
    */
   inline void
   setSpaceSteps( const PointType& steps );

   /**
    * \brief Returns product of space steps to the xPow.
    * \tparam xPow Exponent for dimension x.
    * \tparam yPow Exponent for dimension y.
    * \tparam zPow Exponent for dimension z.
    */
   template< int xPow, int yPow, int zPow >
   __cuda_callable__
   const RealType&
   getSpaceStepsProducts() const;

   /**
    * \brief Returns the number of x-normal faces.
    */
   __cuda_callable__
   IndexType
   getNumberOfNxFaces() const;

   /**
    * \brief Returns the number of x-normal and y-normal faces.
    */
   __cuda_callable__
   IndexType
   getNumberOfNxAndNyFaces() const;

   /**
    * \breif Returns the measure (volume) of a cell in this grid.
    */
   __cuda_callable__
   inline const RealType&
   getCellMeasure() const;

   /**
    * \brief See Grid1D::getSmallestSpaceStep().
    */
   __cuda_callable__
   RealType getSmallestSpaceStep() const;

   /**
    * \brief Traverses all elements
    */
   template<int EntityDimension, typename Func, typename... FuncArgs>
   void forAll(Func func, FuncArgs... args) const;

   /**
    * \brief Traversers interior elements
    */
   template<int EntityDimension, typename Func, typename... FuncArgs>
   void forInterior(Func func, FuncArgs... args) const;

   /**
    * \brief Traversers boundary elements
    */
   template<int EntityDimension, typename Func, typename... FuncArgs>
   void forBoundary(Func func, FuncArgs... args) const;

   void writeProlog( Logger& logger ) const;

   protected:

   void computeProportions();

   void computeSpaceStepPowers();

   void computeSpaceSteps();

   CoordinatesType dimensions, localBegin, localEnd, interiorBegin, interiorEnd;

   IndexType numberOfCells,
          numberOfNxFaces, numberOfNyFaces, numberOfNzFaces, numberOfNxAndNyFaces, numberOfFaces,
          numberOfDxEdges, numberOfDyEdges, numberOfDzEdges, numberOfDxAndDyEdges, numberOfEdges,
          numberOfVertices;

   PointType origin, proportions;

   IndexType cellZNeighborsStep;

   PointType spaceSteps;

   RealType spaceStepsProducts[ 5 ][ 5 ][ 5 ];

   template< typename, typename, int >
   friend class GridEntityGetter;

   template< typename, int, typename >
   friend class NeighborGridEntityGetter;
};

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Grid3D_impl.h>
