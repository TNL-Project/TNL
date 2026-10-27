// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Benchmarks::Graphs {

using LaunchConfiguration = TNL::Algorithms::Segments::LaunchConfiguration;

template< typename Segments >
struct LaunchConfigurationsSetup
{
   static auto
   create() -> std::list< std::pair< LaunchConfiguration, std::string > >
   {
      return std::list< std::pair< LaunchConfiguration, std::string > >{
         { LaunchConfiguration( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" }
      };
   }
};

template< typename Device, typename Index, typename IndexAllocator >
struct LaunchConfigurationsSetup< TNL::Algorithms::Segments::CSR< Device, Index, IndexAllocator > >
{
   static auto
   create() -> std::list< std::pair< LaunchConfiguration, std::string > >
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::Host > || std::is_same_v< Device, TNL::Devices::Sequential > )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" }
         };
      else {
         LaunchConfiguration warpPerSegment( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Uniform, 1 );
         warpPerSegment.setThreadsPerSegmentCountToWarpSize();
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" },
            { warpPerSegment, "Warp per segment" },
            { LaunchConfiguration( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::PerElement, 1 ), "PerElement 1 TPS" },
            { LaunchConfiguration( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::PerElement, 2 ), "PerElement 2 TPS" },
            { LaunchConfiguration( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::PerElement, 4 ), "PerElement 4 TPS" },
            { LaunchConfiguration( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::PerElement, 8 ), "PerElement 8 TPS" }
         };
      }
   }
};

}  // namespace TNL::Benchmarks::Graphs
