// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace TNL::Meshes {

template< typename ConfigTag, typename Device >
class MeshTypeResolver
{
public:
   template< typename Reader, typename Functor >
   [[nodiscard]] static bool
   run( Reader& reader, Functor&& functor );

protected:
   template< typename Reader, typename Functor >
   struct detail
   {
      [[nodiscard]] static bool
      resolveCellTopology( Reader& reader, Functor&& functor );

      // NOTE: We could disable the meshes only by the MeshTag, but doing the
      //       resolution for all subtypes is more flexible and also pretty
      //       good optimization of compilation times.

      template< typename CellTopology >
      [[nodiscard]] static bool
      resolveSpaceDimension( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension >
      [[nodiscard]] static bool
      resolveReal( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real >
      [[nodiscard]] static bool
      resolveGlobalIndex( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex >
      [[nodiscard]] static bool
      resolveLocalIndex( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex, typename LocalIndex >
      [[nodiscard]] static bool
      resolveMeshType( Reader& reader, Functor&& functor );

      template< typename MeshConfig >
      [[nodiscard]] static bool
      resolveTerminate( Reader& reader, Functor&& functor );
   };
};

}  // namespace TNL::Meshes

#include <TNL/Meshes/TypeResolver/MeshTypeResolver.hpp>
