// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <vector>

#include <TNL/Algorithms/detail/Copy.h>
#include <TNL/TypeTraits.h>

namespace TNL::Algorithms {

/**
 * \brief Copies memory from \e source to \e destination.
 *
 * The \e source data is allocated on the device specified by \e SourceDevice
 * and the \e destination data is allocated on the device specified by
 * \e DestinationDevice.
 *
 * \tparam DestinationDevice is the device where the \e destination data is allocated.
 * \tparam SourceDevice is the device where the \e source data is allocated.
 * \tparam DestinationElement is the type of the \e destination data.
 * \tparam SourceElement is the type of the \e source data.
 * \tparam Index is the type of the size of the data.
 * \param destination is the pointer to the \e destination data.
 * \param source is the pointer to the \e source data.
 * \param size is the size of the data.
 */
template< typename DestinationDevice,
          typename SourceDevice = DestinationDevice,
          typename DestinationElement,
          typename SourceElement,
          typename Index >
void
copy( DestinationElement* destination, const SourceElement* source, Index size )
{
   detail::Copy< DestinationDevice, SourceDevice >::copy( destination, source, size );
}

/**
 * \brief Copies memory from the \e source TNL array-like container to the
 * \e destination STL vector.
 *
 * \tparam Array is the type of array  where the \e source data is stored.
 *         It can be for example \ref TNL::Containers::Array,
 *         \ref TNL::Containers::ArrayView, \ref TNL::Containers::Vector
 *         or \ref TNL::Containers::VectorView.
 * \tparam DestinationElement is the type of the \e destination data stored
 *         in the STL vector.
 * \param destination is the destination STL vector.
 * \param source is the source TNL array.
 */
template< typename Array, typename DestinationElement, typename = std::enable_if_t< IsArrayType< Array >::value > >
void
copy( std::vector< DestinationElement >& destination, const Array& source )
{
   if( (std::size_t) source.getSize() != destination.size() )
      destination.resize( source.getSize() );
   copy< Devices::Host, typename Array::DeviceType >( destination.data(), source.getData(), source.getSize() );
}

}  // namespace TNL::Algorithms
