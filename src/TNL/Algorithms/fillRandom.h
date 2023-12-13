// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/FillRandom.h>

namespace TNL::Algorithms {

/**
 * \brief Fills memory between `data` and `data + size` with random Element values in the given range.
 *
 * \tparam Device is the device where the data is allocated.
 * \tparam Element is the type of the data.
 * \tparam Index is the type of the size of the data.
 * \param data is the pointer to the memory where the random values will be set.
 * \param size is the size of the data.
 * \param min_val is the minimum random value
 * \param max_val is the maximum random value
 */
template< typename Device, typename Element, typename Index >
void
fillRandom( Element* data, Index size, Element min_val, Element max_val )
{
   detail::FillRandom< Device >::fillRandom( data, size, min_val, max_val );
}

}  // namespace TNL::Algorithms
