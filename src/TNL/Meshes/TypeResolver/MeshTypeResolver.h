// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/TypeResolver/BuildConfigTags.h>

namespace TNL::Meshes {

template< typename ConfigTag, typename Device >
class MeshTypeResolver
{
public:
   template< typename Reader, typename Functor >
   static void
   run( Reader& reader, Functor&& functor );

protected:
   template< typename Reader, typename Functor >
   struct detail
   {
      static void
      resolveCellTopology( Reader& reader, Functor&& functor );

      // NOTE: We could disable the meshes only by the MeshTag, but doing the
      //       resolution for all subtypes is more flexible and also pretty
      //       good optimization of compilation times.

      template< typename CellTopology >
      static void
      resolveSpaceDimension( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension >
      static void
      resolveReal( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real >
      static void
      resolveGlobalIndex( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex >
      static void
      resolveLocalIndex( Reader& reader, Functor&& functor );

      template< typename CellTopology, int SpaceDimension, typename Real, typename GlobalIndex, typename LocalIndex >
      static void
      resolveMeshType( Reader& reader, Functor&& functor );

      template< typename MeshConfig >
      static void
      resolveTerminate( Reader& reader, Functor&& functor );
   };
};

}  // namespace TNL::Meshes

#include <TNL/Meshes/TypeResolver/MeshTypeResolver.hpp>
