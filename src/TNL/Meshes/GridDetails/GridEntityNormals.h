// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

// Implemented by: Tomáš Oberhuber

#pragma once

#include <cstdint>

#include <TNL/Meshes/Grid.h>

template< int GridDimension, int EntityDimension >
class GridEntityNormals
{
public:

   static constexpr int
   getDimension() { return GridDimension; };

   using NormalsType = Containers::StaticVector< getDimension(), std::int8_t >;

   const getNormals() { return this->normals; };

protected:

   NormalsType normals;
};

template< int GridDimension >
class GridEntityNormals< GridDimension, 0 >
{
public:

   static constexpr int
   getDimension() { return  GridDimension; };

   using NormalsType = Containers::StaticVector< getDimension(), std::int8_t >;

   const getNormals() { return this->normals; };

protected:

   static constexpr NormalsType normals( 1 );
};

template< int GridDimension >
class GridEntityNormals< GridDimension, GridDimension >
{
public:

   static constexpr int
   getDimension() { return GridDimension; };

   using NormalsType = Containers::StaticVector< getDimension(), std::int8_t >;

   const getNormals() { return this->normals; };

protected:

   static constexpr NormalsType normals( 0 );
};
