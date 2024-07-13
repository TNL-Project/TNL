// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/staticFor.h>
#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridEntitiesOrientations.h>

namespace TNL {
namespace Meshes {

template< typename Grid, int GridDimension, int EntityDimension >
class GridEntityBase : public Containers::StaticVector< Grid::getMeshDimension(), typename Grid::IndexType >
//class GridEntityBase : public Containers::StaticVector< Grid::getMeshDimension()+1, typename Grid::IndexType >
// +1 stands here for total orientation index
{
public:
   //using BaseType = Containers::StaticVector< Grid::getMeshDimension()+1, typename Grid::IndexType >;
   using BaseType = Containers::StaticVector< Grid::getMeshDimension(), typename Grid::IndexType >;

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
   GridEntityBase( const std::initializer_list< Value >& elems ) : BaseType( elems ) {}

   __cuda_callable__
   GridEntityBase( IndexType totalOrientationIdx )
   {
      this->totalOrientationIndex = totalOrientationIdx;
      //BaseType::operator[]( getMeshDimension() ) = totalOrientationIdx;
   }

   __cuda_callable__
   GridEntityBase( const CoordinatesType& coordinates, IndexType totalOrientationIdx = 0 )
   {
      this->assignAt( coordinates );
      //BaseType::operator[]( getMeshDimension() ) = totalOrientationIdx;
      this->totalOrientationIndex = totalOrientationIdx;
   }

   __cuda_callable__
   void
   setCoordinates( const CoordinatesType& coordinates )
   {
      this->assignAt( coordinates );
   }

   __cuda_callable__
   const CoordinatesType&
   getCoordinates() const
   {
      return *static_cast< const CoordinatesType* >( (const void*) this );
   }

   __cuda_callable__
   CoordinatesType&
   getCoordinates()
   {
      return *static_cast< CoordinatesType* >( (void*) this );
   }

   __cuda_callable__
   void
   setTotalOrientationIndex( IndexType idx )
   {
      //BaseType::operator[]( getMeshDimension() ) = idx;
      this->totalOrientationIndex = idx;
   }

   __cuda_callable__
   IndexType
   getTotalOrientationIndex() const
   {
      //return this->operator[]( getMeshDimension() );
      return this->totalOrientationIndex;
   }

   __cuda_callable__
   IndexType
   getOrientationIndex() const
   {
      return EntitiesOrientations::template getOrientationIndex< EntityDimension >( this->getTotalOrientationIndex() );
   }

protected:
   IndexType totalOrientationIndex;
};

template< typename Grid, int GridDimension >
class GridEntityBase< Grid, GridDimension, 0 >
: public Containers::StaticVector< Grid::getMeshDimension(), typename Grid::IndexType >
{
public:
   using BaseType = Containers::StaticVector< Grid::getMeshDimension(), typename Grid::IndexType >;

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
   GridEntityBase( const std::initializer_list< Value >& elems ) : BaseType( elems ) {}

   __cuda_callable__
   GridEntityBase( const CoordinatesType& coordinates, IndexType totalOrientationIdx = 0 )
   {
      Algorithms::staticFor< int, 0, getMeshDimension() >(
         [ & ]( int i )
         {
            BaseType::operator[]( i ) = coordinates[ i ];
         } );
   }

   __cuda_callable__
   GridEntityBase( IndexType orientationIdx ) {}

   __cuda_callable__
   void
   setCoordinates( const CoordinatesType& coordinates )
   {
      *this = coordinates;
   }

   __cuda_callable__
   const CoordinatesType&
   getCoordinates() const
   {
      return *this;
   }

   __cuda_callable__
   CoordinatesType&
   getCoordinates()
   {
      return *this;
   }

   __cuda_callable__
   void
   setTotalOrientationIndex( IndexType idx )
   {
      TNL_ASSERT_EQ( idx, 0, "Wrong total orientation index. Only zero value is allowed for vertexes." );
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
};

template< typename Grid, int GridDimension >
class GridEntityBase< Grid, GridDimension, GridDimension >
: public Containers::StaticVector< Grid::getMeshDimension(), typename Grid::IndexType >
{
public:
   using BaseType = Containers::StaticVector< Grid::getMeshDimension(), typename Grid::IndexType >;

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
   GridEntityBase( const std::initializer_list< Value >& elems ) : BaseType( elems ) {}

   __cuda_callable__
   GridEntityBase( IndexType orientationIdx ) {}

   __cuda_callable__
   GridEntityBase( const CoordinatesType& coordinates, IndexType totalOrientationIdx = 0 )
   {
      Algorithms::staticFor< int, 0, getMeshDimension() >(
         [ & ]( int i )
         {
            this->operator[]( i ) = coordinates[ i ];
         } );
   }

   __cuda_callable__
   void
   setCoordinates( const CoordinatesType& coordinates )
   {
      *this = coordinates;
   }

   __cuda_callable__
   const CoordinatesType&
   getCoordinates() const
   {
      return *static_cast< const CoordinatesType* >( (const void*) this );
   }

   __cuda_callable__
   CoordinatesType&
   getCoordinates()
   {
      return *static_cast< CoordinatesType* >( (void*) this );
   }

   __cuda_callable__
   void
   setTotalOrientationIndex( IndexType idx )
   {
      TNL_ASSERT_EQ( idx,
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
};

}  // namespace Meshes
}  // namespace TNL
