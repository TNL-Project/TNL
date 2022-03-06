// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Assert.h>
#include <TNL/Cuda/CudaCallable.h>

namespace TNL {
namespace Meshes {

template <typename GridEntity, int NeighborEntityDimension>
class NeighborGridEntityGetter;

}  // namespace Meshes
}  // namespace TNL
