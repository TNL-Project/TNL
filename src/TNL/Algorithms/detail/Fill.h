// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Backend/Macros.h>

namespace TNL::Algorithms::detail {

template< typename DestinationDevice >
struct Fill;

template<>
struct Fill< Devices::Sequential >
{
   template< typename Element, typename Index >
   __cuda_callable__
   static void
   fill( Element* data, const Element& value, Index size );
};

template<>
struct Fill< Devices::Host >
{
   template< typename Element, typename Index >
   static void
   fill( Element* data, const Element& value, Index size );
};

template<>
struct Fill< Devices::Cuda >
{
   template< typename Element, typename Index >
   static void
   fill( Element* data, const Element& value, Index size );
};

}  // namespace TNL::Algorithms::detail

#include "Fill.hpp"
