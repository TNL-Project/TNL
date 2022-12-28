// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridTraits.h>
#include <TNL/Meshes/GridEntitiesOrientations.h>

namespace TNL {
   namespace Meshes {

template< typename Grid, int GridDimension, int EntityDimension >
class GridEntityOrientation
{
public:

   static constexpr int
   getDimension() { return GridDimension; }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using EntitiesOrientations = GridEntitiesOrientations< getDimension() >;

   __cuda_callable__
   GridEntityOrientation() = default;

   __cuda_callable__
   GridEntityOrientation( IndexType totalOrientationIdx )
      : totalOrientationIdx( totalOrientationIdx ) {}

   __cuda_callable__
   void setTotalOrientationIndex( IndexType idx ) { this->totalOrientationIdx = idx; }

   __cuda_callable__
   IndexType getTotalOrientationIndex() const { return this->totalOrientationIdx; }

   __cuda_callable__
   IndexType getOrientationIndex() const {
      return EntitiesOrientations::template getOrientationIndex< EntityDimension >( this->totalOrientationIdx ); }

protected:

   IndexType totalOrientationIdx = 0;
};

template< typename Grid, int GridDimension >
class GridEntityOrientation< Grid, GridDimension, 0 >
{
public:

   static constexpr int
   getDimension() { return Grid::getMeshDimension(); }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using EntitiesOrientations = GridEntitiesOrientations< getDimension() >;

   __cuda_callable__
   GridEntityOrientation() = default;

   __cuda_callable__
   GridEntityOrientation( IndexType orientationIdx ) {}

      __cuda_callable__
   IndexType getTotalOrientationIndex() const { return 0; }

   __cuda_callable__
   IndexType getOrientationIndex() const { return 0; }
};

template< typename Grid, int GridDimension >
class GridEntityOrientation< Grid, GridDimension, GridDimension >
{
public:

   static constexpr int
   getDimension() { return Grid::getMeshDimension(); }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using EntitiesOrientations = GridEntitiesOrientations< getDimension() >;

   __cuda_callable__
   GridEntityOrientation() = default;

   __cuda_callable__
   GridEntityOrientation( IndexType orientationIdx ) {}

      __cuda_callable__
   IndexType getTotalOrientationIndex() const { return ( 1 << getDimension() ) - 1; }

   __cuda_callable__
   IndexType getOrientationIndex() const { return 0; }
};

   } // namespace Meshes
} // namespace TNL
