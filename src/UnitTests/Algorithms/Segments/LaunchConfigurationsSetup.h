#pragma once

#include <TNL/Algorithms/Segments/CSR.h>

using LaunchConfigurationType = TNL::Algorithms::Segments::LaunchConfiguration;

template< typename Segments >
struct LaunchConfigurationsSetup
{};

template< typename Device, typename Index, typename IndexAllocator >
struct LaunchConfigurationsSetup< TNL::Algorithms::Segments::CSR< Device, Index, IndexAllocator > >
{
   static auto
   create() -> std::list< std::pair< LaunchConfigurationType, std::string > >
   {
      if constexpr( std::is_same_v< Device, TNL::Devices::Host > || std::is_same_v< Device, TNL::Devices::Sequential > )
         return std::list< std::pair< LaunchConfigurationType, std::string > >{
            { LaunchConfigurationType( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::ThreadPerSegment, 1, 1 ),
              "ThreadPerSegment" }
         };
      else
         return std::list< std::pair< LaunchConfigurationType, std::string > >{
            { LaunchConfigurationType( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::ThreadPerSegment, 1, 1 ),
              "ThreadPerSegment" },
            { LaunchConfigurationType( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::WarpPerSegment, 1, 1 ),
              "WarpPerSegment" },
            { LaunchConfigurationType( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::BlockMergedSegments, 256, 1 ),
              "BlockMergedSegments 1 thread per segment" },
            { LaunchConfigurationType( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::BlockMergedSegments, 256, 2 ),
              "BlockMergedSegments 2 thread per segment" },
            { LaunchConfigurationType( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::BlockMergedSegments, 256, 4 ),
              "BlockMergedSegments 4 thread per segment" },
            { LaunchConfigurationType( TNL::Algorithms::Segments::ThreadsToSegmentsMapping::BlockMergedSegments, 256, 8 ),
              "BlockMergedSegments 8 thread per segment" }
         };
   }
};
