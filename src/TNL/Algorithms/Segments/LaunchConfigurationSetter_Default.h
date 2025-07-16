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
      if constexpr( isCSRSegments_v< Segments > ) {
         return LaunchConfiguration( ThreadsToSegmentsMapping::WarpPerSegment, 1 );
      }
      return LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 );
   }
};

}  // namespace TNL::Algorithms::Segments
