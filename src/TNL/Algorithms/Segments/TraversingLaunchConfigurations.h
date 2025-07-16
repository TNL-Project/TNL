// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Traversing launch configurations for segments.
 *
 * This struct creates a list of available launch configurations for given segments type.
 *
 * \tparam Segments The type of segments for which the launch configurations are created.
 */
template< typename Segments >
static auto
traversingLaunchConfigurations( const Segments& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
{
   using Device = typename Segments::DeviceType;

   if constexpr( isCSRSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::WarpPerSegment, 1 ), "WarpPerSegment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMergedSegments, 1 ),
              "BlockMergedSegments 1 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMergedSegments, 2 ),
              "BlockMergedSegments 2 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMergedSegments, 4 ),
              "BlockMergedSegments 4 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMergedSegments, 8 ),
              "BlockMergedSegments 8 thread per segment" }
         };
   }
   if constexpr( isSlicedEllpackSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::WarpPerSegment, 1 ), "WarpPerSegment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMergedSegments, 1 ),
              "BlockMergedSegments 1 thread per segment" }

         };
   }
   if constexpr( isEllpackSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::WarpPerSegment, 1 ), "WarpPerSegment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMergedSegments, 1 ),
              "BlockMergedSegments 1 thread per segment" }
         };
   }

   return std::list< std::pair< LaunchConfiguration, std::string > >{
      { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" }
   };
}

}  // namespace TNL::Algorithms::Segments
