// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Containers/Vector.h>

#include "CSRView.h"
#include "CSRView.h"

namespace TNL::Algorithms::Segments {

template< typename Segments, typename T = std::enable_if< isCSRSegments_v< Segments > > >
struct LaunchConfigurationSetter_HybridCSR
{
   using Index = typename Segments::IndexType;
   static LaunchConfiguration
   create( const Segments& segments )
   {
      LaunchConfiguration launchConfig;
      launchConfig.blockSize = 256;
      const Index segmentsCount = segments.getSegmentsCount();
      if( segmentsCount <= 0 )
         return launchConfig;

      launchConfig.setThreadsToSegmentsMapping( ThreadsToSegmentsMapping::UserDefined );
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