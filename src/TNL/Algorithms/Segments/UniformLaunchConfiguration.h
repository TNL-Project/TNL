// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <type_traits>

#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Math.h>

#include "LaunchConfiguration.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Creates a launch configuration with the `Uniform` mapping and the
 * threads per segment count chosen by the average segment size.
 *
 * The threads per segment count is the smallest power of two which is not
 * smaller than the average number of elements per segment, i.e.
 * `segments.getStorageSize() / segments.getSegmentCount()` rounded up, clamped
 * to the interval `[ minThreadsPerSegment, maxThreadsPerSegment ]`.
 * The block size is 256 threads.
 *
 * With `maxThreadsPerSegment` equal to the warp size, this is the choice of the
 * vector size in the LightSpMV kernel described in the paper:
 *
 * Y. Liu and B. Schmidt,
 * "LightSpMV: Faster CSR-based sparse matrix-vector multiplication
 * on CUDA-enabled GPUs,"
 * 2015 IEEE 26th International Conference on Application-specific Systems,
 * Architectures and Processors (ASAP), Toronto, ON, Canada, 2015, pp. 82-89.
 *
 * A larger `maxThreadsPerSegment` allows to map more
 * than one warp to each segment.
 * The segments format may still reject the resulting configuration,
 * see \ref ThreadsToSegmentsMapping.
 *
 * \tparam Segments The type of the segments.
 * \param segments The segments for which the launch configuration is created.
 * \param minThreadsPerSegment The minimal threads per segment count,
 *        which must be a positive power of two.
 * \param maxThreadsPerSegment The maximal threads per segment count,
 *        which must be a power of two not smaller than `minThreadsPerSegment`.
 * \return The launch configuration.
 * \throws std::invalid_argument if the bounds are not powers of two or
 *         `minThreadsPerSegment` is larger than `maxThreadsPerSegment`.
 */
template< typename Segments >
[[nodiscard]] LaunchConfiguration
uniformLaunchConfigurationBySegmentSize( const Segments& segments, int minThreadsPerSegment, int maxThreadsPerSegment )
{
   using Index = typename Segments::IndexType;

   // The constructor checks that the bounds are powers of two.
   LaunchConfiguration launchConfig( ThreadsToSegmentsMapping::Uniform, minThreadsPerSegment, 256 );
   launchConfig.setThreadsPerSegmentCount( maxThreadsPerSegment );
   if( minThreadsPerSegment > maxThreadsPerSegment )
      throw std::invalid_argument( "The minimal threads per segment count must not be larger than the maximal one." );

   const Index segmentsCount = segments.getSegmentCount();
   const Index averageSegmentSize = segmentsCount > 0 ? roundUpDivision( segments.getStorageSize(), segmentsCount ) : 0;
   int threadsPerSegment = minThreadsPerSegment;
   while( threadsPerSegment < maxThreadsPerSegment && threadsPerSegment < averageSegmentSize )
      threadsPerSegment *= 2;
   launchConfig.setThreadsPerSegmentCount( threadsPerSegment );
   return launchConfig;
}

/**
 * \brief Creates a launch configuration with the `Uniform` mapping and the
 * threads per segment count chosen by the average segment size.
 *
 * The maximal threads per segment count is the warp size of the current GPU
 * for segments on GPU. For other devices, where the launch configuration is
 * ignored, it is 32, so that no GPU runtime call is made.
 * See the overload with `maxThreadsPerSegment` for details.
 *
 * \tparam Segments The type of the segments.
 * \param segments The segments for which the launch configuration is created.
 * \param minThreadsPerSegment The minimal threads per segment count,
 *        which must be a positive power of two.
 * \return The launch configuration.
 * \throws std::invalid_argument if `minThreadsPerSegment` is not a power of two
 *         or it is larger than the maximal threads per segment count.
 */
template< typename Segments >
[[nodiscard]] LaunchConfiguration
uniformLaunchConfigurationBySegmentSize( const Segments& segments, int minThreadsPerSegment = 1 )
{
   int maxThreadsPerSegment = 32;
   if constexpr( std::is_same_v< typename Segments::DeviceType, Devices::GPU > )
      maxThreadsPerSegment = Backend::getWarpSize( Backend::getDevice() );
   return uniformLaunchConfigurationBySegmentSize( segments, minThreadsPerSegment, maxThreadsPerSegment );
}

}  // namespace TNL::Algorithms::Segments
