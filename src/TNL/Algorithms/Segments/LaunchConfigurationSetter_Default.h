// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

#include "CSRView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Creates default launch configuration for segments.
 *
 * \tparam Segments The type of segments for which the launch configuration is created.
 */
template< typename Segments >
struct LaunchConfigurationSetter_Default
{
   static LaunchConfiguration
   create( const Segments& segments )
   {
      if constexpr( isCSRSegments_v< Segments >
                    && (std::is_same_v< typename Segments::DeviceType, Devices::Cuda >
                        || std::is_same_v< typename Segments::DeviceType, Devices::Hip >) )
      {
         return { ThreadsToSegmentsMapping::Warp, 1 };
      }
      return { ThreadsToSegmentsMapping::Fixed, 1 };
   }
};

}  // namespace TNL::Algorithms::Segments
