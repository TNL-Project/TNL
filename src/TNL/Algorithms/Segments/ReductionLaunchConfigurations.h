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

template< typename Segments >
static auto
reductionLaunchConfigurations( const Segments& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
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
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 1 ), "UserDefined 1 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 2 ), "UserDefined 2 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 4 ), "UserDefined 4 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 8 ), "UserDefined 8 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 16 ), "UserDefined 16 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 32 ), "UserDefined 32 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 64 ), "UserDefined 64 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 128 ), "UserDefined 128 thread per segment" },
            { LaunchConfigurationSetter_Default< Segments >::create( segments ), "DefaultLaunchConfiguration" },
            { LaunchConfigurationSetter_LightCSR< Segments >::create( segments ), "CSRLightLaunchConfiguration" },
            { LaunchConfigurationSetter_HybridCSR< Segments >::create( segments ), "CSRHybridLaunchConfiguration" }
         };
   }
   else {
      return std::list< std::pair< LaunchConfiguration, std::string > >{
         { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" },
         { LaunchConfigurationSetter_Default< Segments >::create( segments ), "DefaultLaunchConfiguration" }
      };
   }
}

}  // namespace TNL::Algorithms::Segments
