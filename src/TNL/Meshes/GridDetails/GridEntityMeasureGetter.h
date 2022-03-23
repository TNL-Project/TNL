// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#include <TNL/Meshes/Grid.h>

#pragma once

namespace TNL {
namespace Meshes {

template<typename Grid, int EntityDimension>
class GridEntityMeasureGetter;

} // namespace Meshes
} // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/GridEntityMeasureGetter.hpp>
