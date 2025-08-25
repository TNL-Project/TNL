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
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Warp, 1 ), "Warp per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMerged, 1 ), "BlockMerged 1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMerged, 2 ), "BlockMerged 2 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMerged, 4 ), "BlockMerged 4 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMerged, 8 ), "BlockMerged 8 TPS" }
         };
   }
   if constexpr( isSlicedEllpackSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Warp, 1 ), "Warp per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMerged, 1 ), "BlockMerged 1 TPS" }

         };
   }
   if constexpr( isEllpackSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Warp, 1 ), "Warp per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::BlockMerged, 1 ), "BlockMerged 1 TPS" }
         };
   }

   return std::list< std::pair< LaunchConfiguration, std::string > >{
      { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
   };
}

}  // namespace TNL::Algorithms::Segments
