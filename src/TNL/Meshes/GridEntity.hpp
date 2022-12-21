// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridEntity.h>
#include <TNL/Meshes/GridDetails/BoundaryGridEntityChecker.h>
#include <TNL/Meshes/GridDetails/GridEntityCenterGetter.h>
#include <TNL/Meshes/GridDetails/GridEntityMeasureGetter.h>
#include <TNL/Meshes/GridDetails/NeighbourGridEntityGetter.h>

namespace TNL {
namespace Meshes {

template< class Grid, int EntityDimension >
constexpr int
GridEntity< Grid, EntityDimension >::getMeshDimension()
{
   return Grid::getMeshDimension();
}

template< class Grid, int EntityDimension >
constexpr int
GridEntity< Grid, EntityDimension >::getEntityDimension()
{
   return EntityDimension;
}

template< class Grid, int EntityDimension >
__cuda_callable__
   GridEntity< Grid, EntityDimension >::GridEntity() : CoordinatesType( 0 ),  grid( nullptr )
   {}

   template< class Grid, int EntityDimension >
   __cuda_callable__
   GridEntity< Grid, EntityDimension >::GridEntity( const CoordinatesType& c ) : CoordinatesType( c ),  grid( nullptr )
   {}

   /*template< class Grid, int EntityDimension >
   template< typename... Indexes, std::enable_if_t< ( Grid::getMeshDimension() > 1 ) && sizeof...( Indexes ) == Grid::getMeshDimension(), bool > >
   __cuda_callable__
   GridEntity< Grid, EntityDimension >::GridEntity( Indexes&&... indexes )
   : CoordinatesType( { IndexType( std::forward< Indexes >( indexes ) )... } )
   {
      this->grid = nullptr;
   }*/

   template< class Grid, int EntityDimension >
      template< typename Value >
   __cuda_callable__
   GridEntity< Grid, EntityDimension >::GridEntity( const std::initializer_list< Value >& elems )
      : CoordinatesType( elems ), grid( 0 )
   {
   }

   template< class Grid, int EntityDimension >
   __cuda_callable__
   GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid, const CoordinatesType& coordinates )
   : CoordinatesType( coordinates ), grid( &grid )
   {
      if constexpr (EntityDimension != 0 && EntityDimension != Grid::getMeshDimension() )
         this->getNormals() = grid.template getNormals< EntityDimension >( 0 );
      TNL_ASSERT_EQ( getNormals(), grid.template getNormals< EntityDimension >( orientation.getIndex() ), "Wrong index of entity orientation." );
      this->refresh();
   }

   template< class Grid, int EntityDimension >
   __cuda_callable__
   GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid,
                                                    const CoordinatesType& coordinates,
                                                    const NormalsType& normals )
   : CoordinatesType( coordinates ),
     grid( &grid ), orientation( normals, grid.template getOrientation< EntityDimension >( normals ) )
   {
      TNL_ASSERT_EQ( getNormals(), grid.template getNormals< EntityDimension >( orientation.getIndex() ), "Wrong index of entity orientation." );
      this->refresh();
   }

   template< class Grid, int EntityDimension >
   __cuda_callable__
   GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid, IndexType entityIdx ) : grid( &grid ), index( entityIdx )
   {
      TNL_ASSERT_NE( this->grid, nullptr, "Grid pointer cannot be initialized with null pointer." );
      this->setCoordinates( grid.template getEntityCoordinates< EntityDimension >( entityIdx, this->getOrientation() ) );
      TNL_ASSERT_EQ( getNormals(), grid.template getNormals< EntityDimension >( orientation.getIndex() ), "Wrong index of entity orientation." );
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >::GridEntity( const Grid& grid,
                                                 const CoordinatesType& coordinates,
                                                 const NormalsType& normals,
                                                 const IndexType orientation )
: CoordinatesType( coordinates ),
  grid( &grid ), orientation( normals, orientation )
{
   TNL_ASSERT_EQ( orientation, grid.template getOrientation< EntityDimension >( normals ), "Wrong index of entity orientation." );
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
const typename GridEntity< Grid, EntityDimension >::CoordinatesType&
GridEntity< Grid, EntityDimension >::getCoordinates() const
{
   return *this;
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::CoordinatesType&
GridEntity< Grid, EntityDimension >::getCoordinates()
{
   return *this;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setCoordinates( const CoordinatesType& coordinates )
{
   Grid::CoordinatesType::operator=( coordinates );
   this->refresh();
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::refresh()
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   this->index = this->grid->getEntityIndex( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getIndex() const -> const IndexType&
{
   TNL_ASSERT_GE( this->index, 0, "Entity index is not non-negative." );
   TNL_ASSERT_LT( this->index, grid->template getEntitiesCount< EntityDimension >(), "Entity index is out of bounds." );
   TNL_ASSERT_EQ( this->index, grid->getEntityIndex( *this ), "Wrong value of stored index." );

   return this->index;
}

template< class Grid, int EntityDimension >
__cuda_callable__
bool
GridEntity< Grid, EntityDimension >::isBoundary() const
{
   return BoundaryGridEntityChecker< GridEntity >::isBoundaryEntity( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::PointType
GridEntity< Grid, EntityDimension >::getCenter() const
{
   return GridEntityCenterGetter< GridEntity >::getEntityCenter( *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
typename GridEntity< Grid, EntityDimension >::RealType
GridEntity< Grid, EntityDimension >::getMeasure() const
{
   return GridEntityMeasureGetter< Grid, EntityDimension >::getMeasure( this->getMesh(), *this );
}

template< class Grid, int EntityDimension >
__cuda_callable__
const Grid&
GridEntity< Grid, EntityDimension >::getMesh() const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   return *this->grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getOrientation() const -> const GridEntityOrientationType&
{
   return this->orientation;
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getOrientation() -> GridEntityOrientationType&
{
   return this->orientation;
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getNormals() const -> const NormalsType
{
   return this->orientation.getNormals();
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setNormals( const NormalsType& normals )
{
   this->setNormals( normals );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::getBasis() const -> NormalsType
{
   return 1 - this->getNormals();
}

template< class Grid, int EntityDimension >
__cuda_callable__
GridEntity< Grid, EntityDimension >
GridEntity< Grid, EntityDimension >::getNeighbourEntity( const CoordinatesType& offset ) const
{
   //using Getter = NeighbourGridEntityGetter< getMeshDimension(), EntityDimension, Dimension >;
   //return Getter::template getEntity< Grid >( *this, offset );

   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   return grid->getNeighbourEntity( *this, offset );
}

template< class Grid, int EntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::
getNeighbourEntityIndex( const CoordinatesType& offset ) const -> IndexType
{
   //using Getter = NeighbourGridEntityGetter< getMeshDimension(), EntityDimension, Dimension >;
   //return Getter::template getEntityIndex< Grid >( *this, offset );

   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   return grid->getNeighbourEntityIndex( *this, offset );
}

template< class Grid, int EntityDimension >
   template< int NeighbourEntityDimension >
__cuda_callable__
GridEntity< Grid, NeighbourEntityDimension >
GridEntity< Grid, EntityDimension >::
getNeighbourEntity( const CoordinatesType& offset,
                    const NormalsType& neighbourEntityOrientation ) const
{
   //using Getter = NeighbourGridEntityGetter< getMeshDimension(), EntityDimension, Dimension >;
   //return Getter::template getEntity< Grid, Orientation >( *this, offset );
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   return grid->template getNeighbourEntity< NeighbourEntityDimension >( *this, offset, neighbourEntityOrientation );
}

template< class Grid, int EntityDimension >
   template< int NeighbourEntityDimension >
__cuda_callable__
auto
GridEntity< Grid, EntityDimension >::
getNeighbourEntityIndex( const CoordinatesType& offset,
                         IndexType neighbourEntityOrientation ) const -> IndexType
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   return grid->template getNeighbourEntityIndex< NeighbourEntityDimension >( *this, offset, neighbourEntityOrientation );
}

template< class Grid, int EntityDimension >
auto
GridEntity< Grid, EntityDimension >::getPoint() const -> PointType
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   return this->grid->getSpaceSteps() * this->getCoordinates();
}

template< class Grid, int EntityDimension >
__cuda_callable__
const Grid&
GridEntity< Grid, EntityDimension >::getGrid() const
{
   TNL_ASSERT_NE( this->grid, nullptr, "Trying to dereference null pointer." );
   return *this->grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setGrid( const Grid& grid )
{
   this->grid = &grid;
}

template< class Grid, int EntityDimension >
__cuda_callable__
void
GridEntity< Grid, EntityDimension >::setMesh( const Grid& grid )
{
   this->grid = &grid;
}

template< class Grid, int EntityDimension >
std::ostream&
operator<<( std::ostream& str, const GridEntity< Grid, EntityDimension >& entity )
{
   str << "Entity dimension = " << EntityDimension << " coordinates = " << entity.getCoordinates()
       << " normals = " << entity.getNormals() << " index = " << entity.getIndex()
       << " orientation = " << entity.getOrientation();
   return str;
}

}  // namespace Meshes
}  // namespace TNL
