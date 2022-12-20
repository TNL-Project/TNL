// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <cstdint>
#include <TNL/Containers/StaticVector.h>

namespace TNL {
   namespace Meshes {

template< int Dimension,
          typename Real,
          typename Index >
struct GridTraits
{
   using RealType = Real;

   using IndexType = Index;

   using PointType = Containers::StaticVector< Dimension, Real >;

   using CoordinatesType = Containers::StaticVector< Dimension, IndexType >;

   using NormalsType = Containers::StaticVector< Dimension, short int >;
};

   } //namespace Meshes
} //namespace TNL

