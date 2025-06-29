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
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 2 ), "2 threads per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 4 ), "4 threads per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 8 ), "8 threads per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 16 ), "16 threads per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 32 ), "32 threads per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 64 ), "64 threads per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::UserDefined, 128 ), "128 thread per segment" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::WarpPerSegment, 1 ), "WarpPerSegment" },
            { LaunchConfigurationSetter_Default< Segments >::create( segments ), "Default" },
            { LaunchConfigurationSetter_LightCSR< Segments >::create( segments ), "CSRLight" },
            { LaunchConfigurationSetter_HybridCSR< Segments >::create( segments ), "CSRHybrid" }
         };
   }
   else {
      return std::list< std::pair< LaunchConfiguration, std::string > >{
         { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" },
         { LaunchConfigurationSetter_Default< Segments >::create( segments ), "Default" }
      };
   }
}

}  // namespace TNL::Algorithms::Segments
