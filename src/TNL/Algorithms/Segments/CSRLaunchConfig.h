// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include "LaunchConfig.h"
#include "CSR.h"
#include "CSRView.h"

namespace TNL::Algorithms::Segments {

template< typename Device, typename Index >
struct LaunchConfig< CSRView< Device, Index > >
{
   ThreadsToSegmentsMapping
   getThreadsToSegmentsMapping()
   {
      return threadsToSegmentsMapping;
   }

   Index
   getUserDefinedThreadsPerSegment()
   {
      return userDefinedThreadsPerSegment;
   }

protected:
   ThreadsToSegmentsMapping threadsToSegmentsMapping = ThreadsToSegmentsMapping::WarpToSegmentMapping;
   Index userDefinedThreadsPerSegment = 1;
};

template< typename Device, typename Index, typename IndexAllocator >
struct LaunchConfig< CSR< Device, Index, IndexAllocator > > : public LaunchConfig< CSRView< Device, Index > >
{};

}  // namespace TNL::Algorithms::Segments
