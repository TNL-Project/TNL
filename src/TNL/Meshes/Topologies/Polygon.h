// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Topologies/Edge.h>

namespace TNL::Meshes::Topologies {

struct Polygon
{
   static constexpr int dimension = 2;
};

template<>
struct Subtopology< Polygon, 0 >
{
   using Topology = Vertex;
};

template<>
struct Subtopology< Polygon, 1 >
{
   using Topology = Edge;
};

}  // namespace TNL::Meshes::Topologies
