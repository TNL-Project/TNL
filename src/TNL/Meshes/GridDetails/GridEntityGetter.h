// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Meshes/GridEntity.h>

namespace TNL {
namespace Meshes {

template<class, int>
class GridEntity;

template<typename, int>
class GridEntityGetter;

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Implementations/GridEntityGetter.hpp>
