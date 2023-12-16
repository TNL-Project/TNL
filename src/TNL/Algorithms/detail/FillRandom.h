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
struct FillRandom;

template<>
struct FillRandom< Devices::Sequential >
{
   template< typename Element, typename Index >
   //__cuda_callable__
   static void
   fillRandom( Element* data, Index size, Element min_val, Element max_val );
};

template<>
struct FillRandom< Devices::Host >
{
   template< typename Element, typename Index >
   static void
   fillRandom( Element* data, Index size, Element min_val, Element max_val );
};

template<>
struct FillRandom< Devices::GPU >
{
   template< typename Element, typename Index >
   static void
   fillRandom( Element* data, Index size, Element min_val, Element max_val );
};

}  // namespace TNL::Algorithms::detail

#include "FillRandom.hpp"
