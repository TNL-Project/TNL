// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Logger.h>
#include <TNL/Meshes/GridDetails/NDimGrid.h>

namespace TNL {
namespace Meshes {

template <int Dimension, typename Real = double, typename Device = Devices::Host, typename Index = int>
class Grid: public NDimGrid<Dimension, Real, Device, Index> {};

// template< int Dimension, typename Real, typename Device, typename Index >
// bool operator==( const Grid< Dimension, Real, Device, Index >& lhs,
//                  const Grid< Dimension, Real, Device, Index >& rhs )
// {
//    return lhs.getDimensions() == rhs.getDimensions()
//        && lhs.getOrigin() == rhs.getOrigin()
//        && lhs.getProportions() == rhs.getProportions();
// }

// template< int Dimension, typename Real, typename Device, typename Index >
// bool operator!=( const Grid< Dimension, Real, Device, Index >& lhs,
//                  const Grid< Dimension, Real, Device, Index >& rhs )
// {
//    return ! (lhs == rhs);
// }

}  // namespace Meshes
}  // namespace TNL

#include <TNL/Meshes/GridDetails/Grid1D.h>
#include <TNL/Meshes/GridDetails/Grid2D.h>
#include <TNL/Meshes/GridDetails/Grid3D.h>
