// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Meshes::Topologies {

template< typename EntityTopology, int SubentityDimension >
struct Subtopology
{};

template< typename EntityTopology, typename SubentityTopology, int SubentityIndex, int SubentityVertexIndex >
struct SubentityVertexMap
{};

}  // namespace TNL::Meshes::Topologies
