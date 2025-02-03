// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

namespace TNL::Algorithms::Segments {

template< typename Segments >
struct TraversingLaunchConfigurations
{
   template< typename Segments_ >
   static auto
   create( const Segments_& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
   {
      return std::list< std::pair< LaunchConfiguration, std::string > >{
         { LaunchConfiguration( ThreadsToSegmentsMapping::ThreadPerSegment, 1 ), "ThreadPerSegment" }
      };
   }
};

template< typename Device, typename Index, typename IndexAllocator >
struct TraversingLaunchConfigurations< CSR< Device, Index, IndexAllocator > >
{
   template< typename Segments_ >
   static auto
   create( const Segments_& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
   {
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
};

template< typename Device, typename Index, typename IndexAllocator, ElementsOrganization Organization, int SliceSize >
struct TraversingLaunchConfigurations< SlicedEllpack< Device, Index, IndexAllocator, Organization, SliceSize > >
{
   template< typename Segments_ >
   static auto
   create( const Segments_& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
   {
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
};

template< typename Device, typename Index, typename IndexAllocator, Segments::ElementsOrganization Organization >
struct TraversingLaunchConfigurations< Ellpack< Device, Index, IndexAllocator, Organization > >
{
   template< typename Segments_ >
   static auto
   create( const Segments_& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
   {
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
};

}  // namespace TNL::Algorithms::Segments