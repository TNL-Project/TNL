// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Algorithms::Segments::detail {

/**
 * \brief Throws an exception if `launchConfig` requests
 * other than the `Default` thread mapping.
 *
 * This is used by the segments formats
 * which have a single native kernel for each operation
 * and thus cannot honor any other thread mapping.
 */
inline void
checkDefaultThreadsToSegmentsMapping( const LaunchConfiguration& launchConfig, const std::string& segmentsName )
{
   if( launchConfig.getThreadsToSegmentsMapping() != ThreadsToSegmentsMapping::Default )
      throw std::invalid_argument(
         "Unsupported threads to segments mapping for " + segmentsName + " segments. Only the Default mapping is supported." );
}

/**
 * \brief Calls `function` with the threads per segment count
 * as a compile-time constant.
 *
 * The runtime count `threadsPerSegment` is looked up in `Counts`
 * and `function` is called with `std::integral_constant< int, Count >`
 * for the matching `Count`,
 * so that it can instantiate the kernel for that count.
 * A count which is not in `Counts` throws `std::invalid_argument`.
 *
 * \tparam Counts The supported threads per segment counts.
 * \param threadsPerSegment The threads per segment count to dispatch.
 * \param function The function to call.
 * \param segmentsName The name of the segments format used in the error message.
 */
template< int... Counts, typename Function >
void
dispatchThreadsPerSegment( int threadsPerSegment, Function&& function, const std::string& segmentsName )
{
   const bool supported =
      ( ( threadsPerSegment == Counts && ( function( std::integral_constant< int, Counts >{} ), true ) ) || ... );
   if( ! supported )
      throw std::invalid_argument(
         "Unsupported threads per segment count ( " + std::to_string( threadsPerSegment ) + " ) for " + segmentsName
         + " segments." );
}

}  // namespace TNL::Algorithms::Segments::detail
