// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/CSRView.h>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpackView.h>
#include <TNL/Algorithms/Segments/UniformLaunchConfiguration.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Returns a list of launch configurations for
 * reduction operations on segments.
 *
 * This function generates a list of launch configurations suitable for
 * reduction operations on segments, based on the type of segments provided.
 * The first one is always the Default launch configuration.
 * On the host, the thread mapping is not used, so the Default
 * launch configuration is the only one.
 *
 * \tparam Segments The type of segments for which the launch
 *         configurations are generated.
 * \param segments The segments for which the launch configurations are created.
 * \return A list of pairs containing launch
 *         configurations and their descriptions.
 */
template< typename Segments >
static auto
reductionLaunchConfigurations( const Segments& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
{
   using Device = typename Segments::DeviceType;

   if constexpr( isSortedSegments_v< Segments > ) {
      return reductionLaunchConfigurations( segments.getEmbeddedSegmentsView() );
   }
   else if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > ) {
      return std::list< std::pair< LaunchConfiguration, std::string > >{ { LaunchConfiguration(), "Default" } };
   }
   // For AdaptiveCSR, the Default launch configuration is its native Adaptive mapping.
   else if constexpr( isCSRSegments_v< Segments > || isAdaptiveCSRSegments_v< Segments > ) {
      return std::list< std::pair< LaunchConfiguration, std::string > >{
         { LaunchConfiguration(), "Default" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 2 ), "2 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 4 ), "4 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 8 ), "8 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 16 ), "16 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 32 ), "32 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 64 ), "64 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 128 ), "128 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 1 ), "Tiered 1 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 2 ), "Tiered 2 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 4 ), "Tiered 4 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 8 ), "Tiered 8 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 16 ), "Tiered 16 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 32 ), "Tiered 32 TPS" },
         { uniformLaunchConfigurationBySegmentSize( segments, 2 ), "Uniform by size 2-warp TPS" },
         { uniformLaunchConfigurationBySegmentSize( segments, 2, 128 ), "Uniform by size 2-128 TPS" }
      };
   }
   else if constexpr( isSlicedEllpackSegments_v< Segments > ) {
      if constexpr( Segments::getOrganization() == RowMajorOrder ) {
         std::list< std::pair< LaunchConfiguration, std::string > > launchConfigs{
            { LaunchConfiguration(), "Default" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 2 ), "2 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 4 ), "4 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 8 ), "8 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 16 ), "16 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 32 ), "32 TPS" }
         };
         if constexpr( Backend::getMaxWarpSize() == 64 )
            if( Backend::getWarpSize( Backend::getDevice() ) == 64 )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 64 ), "64 TPS" );
         return launchConfigs;
      }
      else {
         // For the column major ordering we have to ensure that:
         // 1. there are enough threads to cover the slice size
         // 2. TPS * SliceSize >= warp size (checked at runtime below)
         const int warpSize = Backend::getWarpSize( Backend::getDevice() );
         std::list< std::pair< LaunchConfiguration, std::string > > launchConfigs{ { LaunchConfiguration(), "Default" } };
         if constexpr( Segments::getSliceSize() * 1 <= 256 && Segments::getSliceSize() * 1 >= Backend::getMinWarpSize() ) {
            if( Segments::getSliceSize() * 1 >= warpSize )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" );
         }
         if constexpr( Segments::getSliceSize() * 2 <= 256 && Segments::getSliceSize() * 2 >= Backend::getMinWarpSize() ) {
            if( Segments::getSliceSize() * 2 >= warpSize )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 2 ), "2 TPS" );
         }
         if constexpr( Segments::getSliceSize() * 4 <= 256 && Segments::getSliceSize() * 4 >= Backend::getMinWarpSize() ) {
            if( Segments::getSliceSize() * 4 >= warpSize )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 4 ), "4 TPS" );
         }
         if constexpr( Segments::getSliceSize() * 8 <= 256 && Segments::getSliceSize() * 8 >= Backend::getMinWarpSize() ) {
            if( Segments::getSliceSize() * 8 >= warpSize )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 8 ), "8 TPS" );
         }
         if constexpr( Segments::getSliceSize() * 16 <= 256 && Segments::getSliceSize() * 16 >= Backend::getMinWarpSize() ) {
            if( Segments::getSliceSize() * 16 >= warpSize )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 16 ), "16 TPS" );
         }
         if constexpr( Segments::getSliceSize() * 32 <= 256 && Segments::getSliceSize() * 32 >= Backend::getMinWarpSize() ) {
            if( Segments::getSliceSize() * 32 >= warpSize )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 32 ), "32 TPS" );
         }
         if constexpr( Backend::getMaxWarpSize() == 64 && Segments::getSliceSize() * 64 <= 256
                       && Segments::getSliceSize() * 64 >= Backend::getMinWarpSize() )
         {
            if( warpSize == 64 )
               launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 64 ), "64 TPS" );
         }
         return launchConfigs;
      }
   }
   else if constexpr( isEllpackSegments_v< Segments > ) {
      std::list< std::pair< LaunchConfiguration, std::string > > launchConfigs{
         { LaunchConfiguration(), "Default" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 2 ), "2 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 4 ), "4 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 8 ), "8 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 16 ), "16 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 32 ), "32 TPS" }
      };
      if constexpr( Backend::getMaxWarpSize() == 64 )
         if( Backend::getWarpSize( Backend::getDevice() ) == 64 )
            launchConfigs.emplace_back( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 64 ), "64 TPS" );
      return launchConfigs;
   }
   else {
      // The segments formats with a single native kernel support only the Default launch configuration.
      return std::list< std::pair< LaunchConfiguration, std::string > >{ { LaunchConfiguration(), "Default" } };
   }
}

}  // namespace TNL::Algorithms::Segments
