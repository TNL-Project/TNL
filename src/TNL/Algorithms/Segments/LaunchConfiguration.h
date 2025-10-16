// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once
#include <TNL/Backend/KernelLaunch.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Enumeration for mapping threads to segments.
 *
 * This enumeration defines how threads are mapped to segments during parallel operations.
 * It includes options for mapping one thread per segment, one warp per segment, and user-defined mappings.
 */
enum class ThreadsToSegmentsMapping
{
   Fixed,
   Warp,
   Block,
   BlockMerged
};

/**
 * \brief Launch configuration for segment operations.
 *
 * This class encapsulates the configuration for launching segment operations,
 * including the mapping of threads to segments and the number of threads per segment.
 */
struct LaunchConfiguration : public Backend::LaunchConfiguration
{
   LaunchConfiguration() = default;

   LaunchConfiguration( ThreadsToSegmentsMapping threadsToSegmentsMapping, int threadsPerSegmentCount, int blockSize = 1 )
   : Backend::LaunchConfiguration( dim3{}, blockSize ),
     threadsToSegmentsMapping( threadsToSegmentsMapping )
   {
      this->setThreadsPerSegmentCount( threadsPerSegmentCount );
   }

   ThreadsToSegmentsMapping
   getThreadsToSegmentsMapping() const
   {
      return threadsToSegmentsMapping;
   }

   void
   setThreadsPerSegmentCount( int threadsPerSegmentCount )
   {
      if( threadsPerSegmentCount > 0 && ( threadsPerSegmentCount & ( threadsPerSegmentCount - 1 ) ) == 0 )
         this->threadsPerSegmentCount = threadsPerSegmentCount;
      else
         throw std::invalid_argument( "Threads per segment count must be a positive power of two." );
   }

   void
   setThreadsPerSegmentCountToWarpSize()
   {
      this->threadsPerSegmentCount = Backend::getWarpSize();
   }

   int
   getThreadsPerSegmentCount() const
   {
      return threadsPerSegmentCount;
   }

   void
   setThreadsToSegmentsMapping( ThreadsToSegmentsMapping threadsToSegmentsMapping )
   {
      this->threadsToSegmentsMapping = threadsToSegmentsMapping;
   }

   const Backend::LaunchConfiguration&
   getBackendLaunchConfiguration() const
   {
      return *this;
   }

protected:
   ThreadsToSegmentsMapping threadsToSegmentsMapping = ThreadsToSegmentsMapping::Fixed;
   int threadsPerSegmentCount = 1;
};

}  // namespace TNL::Algorithms::Segments
