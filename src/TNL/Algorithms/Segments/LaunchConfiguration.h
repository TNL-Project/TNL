// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once
#include <TNL/Backend/KernelLaunch.h>

namespace TNL::Algorithms::Segments {

enum class ThreadsToSegmentsMapping
{
   ThreadPerSegment,
   WarpPerSegment,
   BlockPerSegment,
   BlockMergedSegments,
   UserDefined
};

struct LaunchConfiguration : public Backend::LaunchConfiguration
{
   LaunchConfiguration() = default;

   LaunchConfiguration( ThreadsToSegmentsMapping threadsToSegmentsMapping, int threadsPerSegmentCount, int blockSize = 1 )
   : Backend::LaunchConfiguration( blockSize ), threadsToSegmentsMapping( threadsToSegmentsMapping ),
     threadsPerSegmentCount( threadsPerSegmentCount )
   {}

   ThreadsToSegmentsMapping
   getThreadsToSegmentsMapping()
   {
      return threadsToSegmentsMapping;
   }

   int
   getThreadsPerSegmentCount()
   {
      return threadsPerSegmentCount;
   }

   const Backend::LaunchConfiguration&
   getBackendLaunchConfiguration() const
   {
      return *this;
   }

protected:
   ThreadsToSegmentsMapping threadsToSegmentsMapping = ThreadsToSegmentsMapping::WarpPerSegment;
   int threadsPerSegmentCount = 1;
};

}  // namespace TNL::Algorithms::Segments
