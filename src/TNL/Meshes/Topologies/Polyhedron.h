// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/Topologies/Polygon.h>

namespace TNL::Meshes::Topologies {

struct Polyhedron
{
   static constexpr int dimension = 3;
};

template<>
struct Subtopology< Polyhedron, 0 >
{
   using Topology = Vertex;
};

template<>
struct Subtopology< Polyhedron, 1 >
{
   using Topology = Edge;
};

template<>
struct Subtopology< Polyhedron, 2 >
{
   using Topology = Polygon;
};

}  // namespace TNL::Meshes::Topologies
