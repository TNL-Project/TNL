// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Returns a list of launch configurations
 * for traversing operations on segments.
 *
 * This function generates a list of launch configurations
 * suitable for traversing operations on segments,
 * based on the type of segments provided.
 * The first one is always the Default launch configuration.
 * On the host, the thread mapping is not used,
 * so the Default launch configuration is the only one.
 *
 * \tparam Segments The type of segments
 *         for which the launch configurations are generated.
 * \param segments The segments for which the launch configurations are created.
 * \return A list of pairs containing launch configurations and their descriptions.
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
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 16 ), "PerElement 16 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 32 ), "PerElement 32 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 1 ), "Tiered 1 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 2 ), "Tiered 2 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 4 ), "Tiered 4 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 8 ), "Tiered 8 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 16 ), "Tiered 16 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::Tiered, 32 ), "Tiered 32 TPS" },
      };
   }
   else if constexpr( isSlicedEllpackSegments_v< Segments > ) {
      std::list< std::pair< LaunchConfiguration, std::string > > configurations;
      configurations.emplace_back( LaunchConfiguration(), "Default" );
      // the column-major organization supports only one thread per segment in the Uniform mapping
      const int maxUniformThreadsPerSegment = Segments::getOrganization() == ColumnMajorOrder ? 1 : 256;
      for( int threadsPerSegment = 1; threadsPerSegment <= maxUniformThreadsPerSegment; threadsPerSegment *= 2 )
         configurations.emplace_back(
            LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, threadsPerSegment ),
            std::to_string( threadsPerSegment ) + " TPS" );
      // every block of 256 threads of the PerElement traversal processes whole slices
      for( int threadsPerSegment = 1; threadsPerSegment <= 32; threadsPerSegment *= 2 )
         if( 256 / threadsPerSegment >= Segments::getSliceSize() )
            configurations.emplace_back(
               LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, threadsPerSegment ),
               "PerElement " + std::to_string( threadsPerSegment ) + " TPS" );
      return configurations;
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
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 1 ), "PerElement 1 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 2 ), "PerElement 2 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 4 ), "PerElement 4 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 8 ), "PerElement 8 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 16 ), "PerElement 16 TPS" },
         { LaunchConfiguration( ThreadsToSegmentsMapping::PerElement, 32 ), "PerElement 32 TPS" }
      };
   }
   else {
      // The segments formats with a single native kernel support only the Default launch configuration.
      return std::list< std::pair< LaunchConfiguration, std::string > >{ { LaunchConfiguration(), "Default" } };
   }
}

}  // namespace TNL::Algorithms::Segments
