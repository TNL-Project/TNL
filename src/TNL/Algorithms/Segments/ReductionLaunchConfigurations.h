// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/LaunchConfigurationSetter_Default.h>
#include <TNL/Algorithms/Segments/LaunchConfigurationSetter_LightCSR.h>
#include <TNL/Algorithms/Segments/LaunchConfigurationSetter_HybridCSR.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Returns a list of launch configurations for reduction operations on segments.
 *
 * This function generates a list of launch configurations suitable for reduction operations
 * on segments, based on the type of segments provided.
 *
 * \tparam Segments The type of segments for which the launch configurations are generated.
 * \param segments The segments for which the launch configurations are created.
 * \return A list of pairs containing launch configurations and their descriptions.
 */
template< typename Segments >
static auto
reductionLaunchConfigurations( const Segments& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
{
   using Device = typename Segments::DeviceType;

   if constexpr( isSortedSegments_v< Segments > ) {
      return reductionLaunchConfigurations( segments.getEmbeddedSegmentsView() );
   }

   if constexpr( isCSRSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 2 ), "2 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 4 ), "4 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 8 ), "8 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 16 ), "16 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 32 ), "32 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 64 ), "64 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 128 ), "128 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::DynamicGrouping, 256 ), "DynamicGrouping" },
            { LaunchConfigurationSetter_LightCSR< Segments >::create( segments ), "Light CSR" },
            { LaunchConfigurationSetter_HybridCSR< Segments >::create( segments ), "Hybrid CSR" }
         };
   }
   else if constexpr( isSlicedEllpackSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 2 ), "2 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 4 ), "4 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 8 ), "8 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 16 ), "16 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 32 ), "32 TPS" }
         };
   }
   else if constexpr( isEllpackSegments_v< Segments > ) {
      if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
         };
      else
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 2 ), "2 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 4 ), "4 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 8 ), "8 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 16 ), "16 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 32 ), "32 TPS" }
         };
   }
   else {
      return std::list< std::pair< LaunchConfiguration, std::string > >{
         { LaunchConfiguration( ThreadsToSegmentsMapping::Fixed, 1 ), "1 TPS" }
      };
   }
}

}  // namespace TNL::Algorithms::Segments
