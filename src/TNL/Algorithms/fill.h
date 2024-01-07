// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/detail/Fill.h>

namespace TNL::Algorithms {

/**
 * \brief Fills memory between `data` and `data + size` with a `value`.
 *
 * \tparam Device is the device where the \e data is allocated.
 * \tparam Element is the type of the \e data.
 * \tparam Index is the type of the size of the data.
 * \param data is the pointer to the memory where the value will be set.
 * \param value is the value to be filled.
 * \param size is the size of the data.
 */
template< typename Device, typename Element, typename Index >
void
fill( Element* data, const Element& value, Index size )
{
   detail::Fill< Device >::fill( data, value, size );
}

}  // namespace TNL::Algorithms
