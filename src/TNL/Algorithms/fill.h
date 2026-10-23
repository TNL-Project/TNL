// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include <TNL/Algorithms/detail/Fill.h>

namespace TNL::Algorithms {

/**
 * \brief Fills memory between `data` and `data + size` with a `value`.
 *
 * On the GPU, the fill does not block the host when
 * `launch_config.blockHostUntilFinished` is false. The caller is then
 * responsible for synchronizing (e.g. via `TNL::Backend::streamSynchronize`)
 * before reading the filled memory on the host.
 *
 * \tparam Device is the device where the `data` is allocated.
 * \tparam Element is the type of the `data`.
 * \tparam Index is the type of the size of the data.
 * \param data is the pointer to the memory where the value will be set.
 * \param value is the value to be filled.
 * \param size is the size of the data.
 * \param launch_config specifies kernel launch parameters.
 */
template< typename Device, typename Element, typename Index >
void
fill( Element* data, const Element& value, Index size, typename Device::LaunchConfiguration launch_config = {} )
{
   detail::Fill< Device >::fill( data, value, size, std::move( launch_config ) );
}

}  // namespace TNL::Algorithms
