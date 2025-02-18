// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include "TypeTraits.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Insertion operator of segments to output stream.
 *
 * \tparam Device is the device type of the source segments.
 * \tparam Index is the index type of the source segments.
 * \tparam IndexAllocator is the index allocator of the source segments.
 * \param str is the output stream.
 * \param segments are the source segments.
 * \return reference to the output stream.
 */
template< typename Segments, typename T = std::enable_if_t< isSegments_v< Segments > > >
std::ostream&
operator<<( std::ostream& str, const Segments& segments );

template< typename SegmentsView, typename Fetch >
struct SegmentsPrinter;

/**
 * \brief Print segments sizes, i.e. the segments setup.
 *
 * \tparam Segments is type of segments.
 * \tparam Fetch is type of the lambda function for fetching data.
 *
 * \param segments is an instance of segments.
 * \param fetch is a lambda function for fetching data.
 *
 * \return reference to the output stream.
 *
 * \par Example
 * \include Algorithms/Segments/printSegmentsExample-1.cpp
 * \par Output
 * \include printSegmentsExample-1.out
 */
template< typename Segments, typename Fetch, typename T = std::enable_if_t< isSegments_v< Segments > > >
SegmentsPrinter< typename Segments::ConstViewType, Fetch >
print( const Segments& segments, Fetch fetch );

}  // namespace TNL::Algorithms::Segments

#include "print.hpp"
