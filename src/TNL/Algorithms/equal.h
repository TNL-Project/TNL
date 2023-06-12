// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/Equal.h>

namespace TNL::Algorithms {

/**
 * \brief Compares memory from \e source with \e destination.
 *
 * The \e source data is allocated on the device specified by \e SourceDevice and the
 * \e destination data is allocated on the device specified by \e DestinationDevice.
 *
 * \tparam DestinationDevice is the device where the \e destination data is allocated.
 * \tparam SourceDevice is the device where the \e source data is allocated.
 * \tparam DestinationElement is the type of the \e destination data.
 * \tparam SourceElement is the type of the \e source data.
 * \tparam Index is the type of the size of the data.
 * \param destination is the pointer to the \e destination data.
 * \param source is the pointer to the \e source data.
 * \param size is the size of the data.
 * \returns `true` if all elements are equal, `false` otherwise.
 */
template< typename DestinationDevice,
          typename SourceDevice = DestinationDevice,
          typename DestinationElement,
          typename SourceElement,
          typename Index >
bool
equal( DestinationElement* destination, const SourceElement* source, Index size )
{
   return detail::Equal< DestinationDevice, SourceDevice >::equal( destination, source, size );
}

}  // namespace TNL::Algorithms
