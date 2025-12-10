// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include <TNL/Meshes/MeshEntity.h>

namespace TNL::Meshes::Writers::detail {

template< typename T, typename Enable = void >
struct has_entity_topology : std::false_type
{};

template< typename T >
struct has_entity_topology< T, typename enable_if_type< typename T::EntityTopology >::type > : std::true_type
{};

template< typename Entity, bool _is_mesh_entity = has_entity_topology< Entity >::value >
struct VerticesPerEntity
{
   static constexpr int count = Topologies::Subtopology< typename Entity::EntityTopology, 0 >::count;
};

template< typename MeshConfig, typename Device >
struct VerticesPerEntity< MeshEntity< MeshConfig, Device, Topologies::Vertex >, true >
{
   static constexpr int count = 1;
};

template< typename GridEntity >
struct VerticesPerEntity< GridEntity, false >
{
public:
   static constexpr int count = []() constexpr
   {
      constexpr int dim = GridEntity::getEntityDimension();
      static_assert( dim >= 0 && dim <= 3, "unexpected dimension of the grid entity" );

      if constexpr( dim == 0 )
         return 1;
      if constexpr( dim == 1 )
         return 2;
      if constexpr( dim == 2 )
         return 4;
      if constexpr( dim == 3 )
         return 8;
   }();
};

}  // namespace TNL::Meshes::Writers::detail
