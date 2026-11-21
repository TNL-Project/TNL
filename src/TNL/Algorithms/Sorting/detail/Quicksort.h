// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Devices/GPU.h>

#include "GPUQuicksort.h"

namespace TNL::Algorithms::Sorting::detail {

/**
 * \brief Device dispatcher of the quicksort implementations.
 *
 * \tparam Value type of the sorted elements.
 * \tparam Device device on which the sort runs.
 * \tparam Index index type of the sorted array.
 */
template< typename Value, typename Device, typename Index >
class Quicksort
{
   static_assert( std::is_same_v< Device, Devices::GPU >, "Quicksort is implemented only for the GPU devices." );
};

template< typename Value, typename Index >
class Quicksort< Value, Devices::GPU, Index > : public GPUQuicksort< Value, Index >
{};

}  // namespace TNL::Algorithms::Sorting::detail
