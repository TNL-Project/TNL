// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Traversing launch configurations for segments.
 *
 * This struct creates a list of available launch configurations for given segments type.
 * The first one is always the Default launch configuration. On the host, the thread mapping
 * is not used, so the Default launch configuration is the only one.
 *
 * \tparam Segments The type of segments for which the launch configurations are created.
 */
template< typename Segments >
static auto
traversingLaunchConfigurations( const Segments& segments ) -> std::list< std::pair< LaunchConfiguration, std::string > >
{
   using Device = typename Segments::DeviceType;

   if constexpr( isSortedSegments_v< Segments > ) {
      return traversingLaunchConfigurations( segments.getEmbeddedSegmentsView() );
   }
   else if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > ) {
      return std::list< std::pair< LaunchConfiguration, std::string > >{ { LaunchConfiguration(), "Default" } };
   }
   // The traversal of AdaptiveCSR is the one of CSR.
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
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 256 ), "256 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 1 ), "PerElement 1 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 2 ), "PerElement 2 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 4 ), "PerElement 4 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 8 ), "PerElement 8 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 1 ), "Tiered 1 TPS" },
      };
   }
   else if constexpr( isSlicedEllpackSegments_v< Segments > ) {
      if constexpr( Segments::getOrganization() == ColumnMajorOrder )
         return std::list< std::pair< LaunchConfiguration, std::string > >{
            { LaunchConfiguration(), "Default" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 1 ), "1 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 1 ), "PerElement 1 TPS" }
         };
      else
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
            { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 256 ), "256 TPS" },
            { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 1 ), "PerElement 1 TPS" }
         };
   }
   else if constexpr( isEllpackSegments_v< Segments > ) {
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
         { LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, 256 ), "256 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 1 ), "PerElement 1 TPS" }
      };
   }
   else {
      // The segments formats with a single native kernel support only the Default launch configuration.
      return std::list< std::pair< LaunchConfiguration, std::string > >{ { LaunchConfiguration(), "Default" } };
   }
}

}  // namespace TNL::Algorithms::Segments
