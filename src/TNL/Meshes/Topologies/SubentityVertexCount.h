// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Topologies/SubentityVertexMap.h>

namespace TNL::Meshes::Topologies {

template< typename EntityTopology, typename SubentityTopology, int SubentityIndex >
struct SubentityVertexCount
{
   static constexpr int count = Subtopology< SubentityTopology, 0 >::count;
};

}  // namespace TNL::Meshes::Topologies
