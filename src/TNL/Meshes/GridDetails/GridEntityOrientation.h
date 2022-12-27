// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <TNL/Meshes/Grid.h>
#include <TNL/Meshes/GridDetails/GridTraits.h>

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

   using GridTraitsType = GridTraits< getDimension(), RealType, IndexType >;

   using NormalsType = typename GridTraitsType::NormalsType;

   __cuda_callable__
   GridEntityOrientation() = default;

   __cuda_callable__
   GridEntityOrientation( const NormalsType& normals,
                          IndexType orientationIdx )
      : normals( normals ), orientationIdx( orientationIdx ) {}

   __cuda_callable__
   void setNormals( const NormalsType& normals ) { this->normals = normals; }

   __cuda_callable__
   const NormalsType& getNormals() const { return this->normals; }

   __cuda_callable__
   void setIndex( IndexType idx ) { this->orientationIdx = idx; }

   __cuda_callable__
   IndexType getIndex() const { return this->orientationIdx; }

protected:

   NormalsType normals;

   IndexType orientationIdx = 0;
};

template< typename Grid, int GridDimension >
class GridEntityOrientation< Grid, GridDimension, 0 >
{
public:

   static constexpr int
   getDimension() { return Grid::getMeshDimension(); }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using GridTraitsType = GridTraits< getDimension(), RealType, IndexType >;

   using NormalsType = typename GridTraitsType::NormalsType;

   __cuda_callable__
   GridEntityOrientation() = default;

   __cuda_callable__
   GridEntityOrientation( const NormalsType& normals,
                          IndexType orientationIdx ) {}

   __cuda_callable__
   NormalsType getNormals() const { NormalsType n; n = 1; return n; }

     __cuda_callable__
   IndexType getIndex() const { return 0; }
};

template< typename Grid, int GridDimension >
class GridEntityOrientation< Grid, GridDimension, GridDimension >
{
public:

   static constexpr int
   getDimension() { return Grid::getMeshDimension(); }

   using RealType = typename Grid::RealType;

   using IndexType = typename Grid::IndexType;

   using GridTraitsType = GridTraits< getDimension(), RealType, IndexType >;

   using NormalsType = typename GridTraitsType::NormalsType;

   __cuda_callable__
   GridEntityOrientation() = default;

   __cuda_callable__
   GridEntityOrientation( const NormalsType& normals,
                          IndexType orientationIdx ) {}

   __cuda_callable__
   const NormalsType getNormals() const { NormalsType n; n = 0; return n; }

   __cuda_callable__
   IndexType getIndex() const { return 0; }
};

   } // namespace Meshes
} // namespace TNL
