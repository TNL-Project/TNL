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

template< typename DestinationDevice, typename SourceDevice = DestinationDevice >
struct Equal;

template<>
struct Equal< Devices::Sequential >
{
   template< typename Element1, typename Element2, typename Index >
   __cuda_callable__
   static bool
   equal( const Element1* destination, const Element2* source, Index size );
};

template<>
struct Equal< Devices::Host >
{
   template< typename Element1, typename Element2, typename Index >
   static bool
   equal( const Element1* destination, const Element2* source, Index size );
};

template<>
struct Equal< Devices::Cuda >
{
   template< typename Element1, typename Element2, typename Index >
   static bool
   equal( const Element1* destination, const Element2* source, Index size );
};

template<>
struct Equal< Devices::Host, Devices::Sequential > : public Equal< Devices::Host, Devices::Host >
{};

template<>
struct Equal< Devices::Sequential, Devices::Host > : public Equal< Devices::Host, Devices::Host >
{};

template< typename DeviceType >
struct Equal< Devices::Cuda, DeviceType >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static bool
   equal( const DestinationElement* destination, const SourceElement* source, Index size );
};

template< typename DeviceType >
struct Equal< DeviceType, Devices::Cuda >
{
   template< typename Element1, typename Element2, typename Index >
   static bool
   equal( const Element1* destination, const Element2* source, Index size );
};

}  // namespace TNL::Algorithms::detail

#include "Equal.hpp"
