// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "CSRView.h"
#include "CSRView.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Launch configuration setter for CSR segments.
 *
 * The mapping of threads is inspired by paper:
 *
 * Y. Liu and B. Schmidt, "LightSpMV: Faster CSR-based sparse matrix-vector multiplication on CUDA-enabled GPUs," 2015 IEEE 26th
 * International Conference on Application-specific Systems, Architectures and Processors (ASAP), Toronto, ON, Canada, 2015, pp.
 * 82-89.
 *
 * but it allows to map more than one warp of threads to each segment.
 *
 * \tparam Segments The type of segments for which the launch configuration is created.
 */
template< typename Segments >
struct LaunchConfigurationSetter_HybridCSR
{
   static_assert( isCSRSegments_v< Segments >, "Segments must be of CSR type." );
   using Index = typename Segments::IndexType;
   static LaunchConfiguration
   create( const Segments& segments )
   {
      LaunchConfiguration launchConfig;
      launchConfig.blockSize = 256;
      const Index segmentsCount = segments.getSegmentCount();
      if( segmentsCount <= 0 )
         return launchConfig;

      launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::Fixed );
      const Index elementsInSegment = roundUpDivision( segments.getStorageSize(), segmentsCount );
      if( elementsInSegment <= 2 )
         launchConfig.setThreadsPerSegmentCount( 2 );
      else if( elementsInSegment <= 4 )
         launchConfig.setThreadsPerSegmentCount( 4 );
      else if( elementsInSegment <= 8 )
         launchConfig.setThreadsPerSegmentCount( 8 );
      else if( elementsInSegment <= 16 )
         launchConfig.setThreadsPerSegmentCount( 16 );
      else if( elementsInSegment <= 32 )
         launchConfig.setThreadsPerSegmentCount( 32 );
      else if( elementsInSegment <= 64 )
         launchConfig.setThreadsPerSegmentCount( 64 );
      else
         launchConfig.setThreadsPerSegmentCount( 128 );
      return launchConfig;
   }
};

}  // namespace TNL::Algorithms::Segments
