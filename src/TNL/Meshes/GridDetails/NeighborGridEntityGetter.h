// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/CudaCallable.h>
#include <TNL/Meshes/GridEntityConfig.h>

namespace TNL {
namespace Meshes {

template <typename GridEntity, int NeighborEntityDimension>
class NeighborGridEntityGetter {
  public:
   // TODO: not all specializations are implemented yet

   __cuda_callable__ NeighborGridEntityGetter(const GridEntity& entity) {
      // TNL_ASSERT( false, );
   }

   __cuda_callable__ void refresh(const typename GridEntity::Gride& grid, const typename GridEntity::IndexType& entityIndex) {
      // TNL_ASSERT( false, );
   }
};

}  // namespace Meshes
}  // namespace TNL
