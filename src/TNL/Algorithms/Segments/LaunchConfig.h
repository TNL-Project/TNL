// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Algorithms::Segments {

enum class ThreadsToSegmentsMapping
{
   ThreadToSegmentMapping,
   WarpToSegmentMapping,
   BlockToSegmentMapping,
   MergedSegmentsMapping,
   UserDefinedThreadsToSegmentsMapping
};

template< typename Segments >
struct LaunchConfig
{
   using IndexType = typename Segments::IndexType;
   ThreadsToSegmentsMapping
   getThreadsToSegmentsMapping()
   {
      return threadsToSegmentsMapping;
   }

   IndexType
   getUserDefinedThreadsPerSegment()
   {
      return userDefinedThreadsPerSegment;
   }

protected:
   ThreadsToSegmentsMapping threadsToSegmentsMapping = ThreadsToSegmentsMapping::WarpToSegmentMapping;
   IndexType userDefinedThreadsPerSegment = 1;
};

}  // namespace TNL::Algorithms::Segments
