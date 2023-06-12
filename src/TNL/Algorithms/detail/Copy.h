// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Cuda/CudaCallable.h>

namespace TNL::Algorithms::detail {

template< typename DestinationDevice, typename SourceDevice = DestinationDevice >
struct Copy;

template<>
struct Copy< Devices::Sequential >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   __cuda_callable__
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size );
};

template<>
struct Copy< Devices::Host >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size );
};

template<>
struct Copy< Devices::Cuda >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size );
};

template<>
struct Copy< Devices::Host, Devices::Sequential > : public Copy< Devices::Host, Devices::Host >
{};

template<>
struct Copy< Devices::Sequential, Devices::Host > : public Copy< Devices::Host, Devices::Host >
{};

template< typename DeviceType >
struct Copy< Devices::Cuda, DeviceType >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size );
};

template< typename DeviceType >
struct Copy< DeviceType, Devices::Cuda >
{
   template< typename DestinationElement, typename SourceElement, typename Index >
   static void
   copy( DestinationElement* destination, const SourceElement* source, Index size );
};

}  // namespace TNL::Algorithms::detail

#include "Copy.hpp"
