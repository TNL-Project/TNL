// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>

#include <TNL/Meshes/MeshDetails/traits/MeshTraits.h>
#include <TNL/Meshes/Topologies/Polyhedron.h>

namespace TNL::Meshes {

template< typename EntitySeed >
struct EntitySeedHash;
template< typename EntitySeed >
struct EntitySeedEq;

template< typename MeshConfig, typename EntityTopology >
class EntitySeed< MeshConfig, EntityTopology, false >
{
   using MeshTraitsType = MeshTraits< MeshConfig >;
   using SubvertexTraits = typename MeshTraitsType::template SubentityTraits< EntityTopology, 0 >;

public:
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using IdArrayType = Containers::StaticArray< SubvertexTraits::count, GlobalIndexType >;
   using HashType = EntitySeedHash< EntitySeed >;
   using KeyEqual = EntitySeedEq< EntitySeed >;

   // this function is here only for compatibility with MeshReader
   void
   setCornersCount( const LocalIndexType& cornersCount )
   {}

   [[nodiscard]] static constexpr LocalIndexType
   getCornersCount()
   {
      return SubvertexTraits::count;
   }

   void
   setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
   {
      if( cornerIndex < 0 || cornerIndex >= getCornersCount() )
         throw std::out_of_range( "setCornerId: cornerIndex is out of range" );
      if( pointIndex < 0 )
         throw std::out_of_range( "setCornerId: pointIndex is out of range" );

      this->cornerIds[ cornerIndex ] = pointIndex;
   }

   [[nodiscard]] IdArrayType&
   getCornerIds()
   {
      return cornerIds;
   }

   [[nodiscard]] const IdArrayType&
   getCornerIds() const
   {
      return cornerIds;
   }

private:
   IdArrayType cornerIds;
};

template< typename MeshConfig >
class EntitySeed< MeshConfig, Topologies::Vertex, false >
{
   using MeshTraitsType = MeshTraits< MeshConfig >;

public:
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using IdArrayType = Containers::StaticArray< 1, GlobalIndexType >;
   using HashType = EntitySeedHash< EntitySeed >;
   using KeyEqual = EntitySeedEq< EntitySeed >;

   // this function is here only for compatibility with MeshReader
   void
   setCornersCount( const LocalIndexType& cornersCount )
   {}

   [[nodiscard]] static constexpr LocalIndexType
   getCornersCount()
   {
      return 1;
   }

   void
   setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
   {
      if( cornerIndex != 0 )
         throw std::invalid_argument( "setCornerId: cornerIndex must be 0" );
      if( pointIndex < 0 )
         throw std::invalid_argument( "setCornerId: point index must be non-negative" );

      this->cornerIds[ cornerIndex ] = pointIndex;
   }

   [[nodiscard]] IdArrayType&
   getCornerIds()
   {
      return cornerIds;
   }

   [[nodiscard]] const IdArrayType&
   getCornerIds() const
   {
      return cornerIds;
   }

private:
   IdArrayType cornerIds;
};

template< typename MeshConfig, typename EntityTopology >
class EntitySeed< MeshConfig, EntityTopology, true >
{
   using MeshTraitsType = MeshTraits< MeshConfig >;

public:
   using GlobalIndexType = typename MeshTraitsType::GlobalIndexType;
   using LocalIndexType = typename MeshTraitsType::LocalIndexType;
   using IdArrayType = Containers::Array< GlobalIndexType, Devices::Host, LocalIndexType >;
   using HashType = EntitySeedHash< EntitySeed >;
   using KeyEqual = EntitySeedEq< EntitySeed >;

   // this constructor definition is here to avoid default constructor being implicitly declared as __host__ __device__, that
   // causes warning: warning #20011-D: calling a __host__ function("std::allocator<int> ::allocator") from a __host__
   // __device__ function("TNL::Meshes::EntitySeed< ::MeshTest::TestTwoWedgesMeshConfig,
   // ::TNL::Meshes::Topologies::Polygon> ::EntitySeed [subobject]") is not allowed
   EntitySeed() = default;

   void
   setCornersCount( const LocalIndexType& cornersCount )
   {
      if constexpr( std::is_same_v< EntityTopology, Topologies::Polygon > ) {
         if( cornersCount < 3 )
            throw std::invalid_argument( "setCornersCount: polygon must have at least 3 corners" );
      }
      else if constexpr( std::is_same_v< EntityTopology, Topologies::Polyhedron > ) {
         if( cornersCount < 4 )
            throw std::invalid_argument( "setCornersCount: polyhedron must have at least 4 faces" );
      }

      this->cornerIds.setSize( cornersCount );
   }

   [[nodiscard]] LocalIndexType
   getCornersCount() const
   {
      return this->cornerIds.getSize();
   }

   void
   setCornerId( const LocalIndexType& cornerIndex, const GlobalIndexType& pointIndex )
   {
      if( cornerIndex < 0 || cornerIndex >= getCornersCount() )
         throw std::out_of_range( "setCornerId: cornerIndex is out of range" );
      if( pointIndex < 0 )
         throw std::out_of_range( "setCornerId: pointIndex is out of range" );

      this->cornerIds[ cornerIndex ] = pointIndex;
   }

   [[nodiscard]] IdArrayType&
   getCornerIds()
   {
      return cornerIds;
   }

   [[nodiscard]] const IdArrayType&
   getCornerIds() const
   {
      return cornerIds;
   }

private:
   IdArrayType cornerIds;
};

template< typename MeshConfig, typename EntityTopology >
std::ostream&
operator<<( std::ostream& str, const EntitySeed< MeshConfig, EntityTopology >& e )
{
   str << e.getCornerIds();
   return str;
}

template< typename EntitySeed >
struct EntitySeedHash
{
   [[nodiscard]] std::size_t
   operator()( const EntitySeed& seed ) const
   {
      using LocalIndexType = typename EntitySeed::LocalIndexType;
      using GlobalIndexType = typename EntitySeed::GlobalIndexType;

      // Note that we must use an associative function to combine the hashes,
      // because we *want* to ignore the order of the corner IDs.
      std::size_t hash = 0;
      for( LocalIndexType i = 0; i < seed.getCornersCount(); i++ )
         // hash ^= std::hash< GlobalIndexType >{}( seed.getCornerIds()[ i ] );
         hash += std::hash< GlobalIndexType >{}( seed.getCornerIds()[ i ] );
      return hash;
   }
};

template< typename EntitySeed >
struct EntitySeedEq
{
   [[nodiscard]] bool
   operator()( const EntitySeed& left, const EntitySeed& right ) const
   {
      using IdArrayType = typename EntitySeed::IdArrayType;

      IdArrayType sortedLeft( left.getCornerIds() );
      IdArrayType sortedRight( right.getCornerIds() );

      // use std::sort for now, because polygon EntitySeeds use TNL::Containers::Array for cornersIds, that is missing sort
      // function
      std::sort( sortedLeft.getData(), sortedLeft.getData() + sortedLeft.getSize() );
      std::sort( sortedRight.getData(), sortedRight.getData() + sortedRight.getSize() );
      /*sortedLeft.sort();
      sortedRight.sort();*/
      return sortedLeft == sortedRight;
   }
};

template< typename MeshConfig >
struct EntitySeedEq< EntitySeed< MeshConfig, Topologies::Vertex > >
{
   using Seed = EntitySeed< MeshConfig, Topologies::Vertex >;

   [[nodiscard]] bool
   operator()( const Seed& left, const Seed& right ) const
   {
      return left.getCornerIds() == right.getCornerIds();
   }
};

}  // namespace TNL::Meshes
