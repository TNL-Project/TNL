// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include <TNL/Devices/Host.h>

namespace TNL::Algorithms::Segments::detail {

// Returns the first segment of the given part when the segments [begin, end) are split into
// parts with approximately the same weight. The weight of a segment is its number of elements
// plus one, so that empty segments are accounted for, too.
// IndexType is spelled as a nested type of OffsetsView (a non-deduced context) rather than as its
// own template parameter, since callers may pass begin and end with different (but convertible)
// types, e.g. a literal 0 and segments.getSegmentCount().
template< typename OffsetsView >
typename OffsetsView::IndexType
getBalancedPartitionBoundary(
   const OffsetsView& offsets,
   typename OffsetsView::IndexType begin,
   typename OffsetsView::IndexType end,
   int part,
   int parts )
{
   using IndexType = typename OffsetsView::IndexType;
   const auto weight = [ &offsets ]( IndexType i )
   {
      return static_cast< std::int64_t >( offsets[ i ] ) + static_cast< std::int64_t >( i );
   };
   const std::int64_t target = weight( begin ) + ( weight( end ) - weight( begin ) ) * part / parts;
   // binary search for the first segment i in [begin, end] such that weight( i ) >= target
   IndexType low = begin;
   IndexType high = end;
   while( low < high ) {
      const IndexType middle = low + ( high - low ) / 2;
      if( weight( middle ) < target )
         low = middle + 1;
      else
         high = middle;
   }
   return low;
}

// Calls f( segmentIdx ) for all segments in [begin, end) on the host. Each OpenMP thread
// processes one contiguous block of segments with approximately the same number of elements.
// Contrary to dynamic scheduling, this keeps the memory access pattern of each thread
// sequential and maps the same segments to the same threads in repeated calls, which
// preserves data locality in caches and NUMA nodes.
template< typename OffsetsView, typename Function >
void
forSegmentsHost(
   const OffsetsView& offsets,
   typename OffsetsView::IndexType begin,
   typename OffsetsView::IndexType end,
   Function& f )
{
   using IndexType = typename OffsetsView::IndexType;
#ifdef HAVE_OPENMP
   // Small problems are not worth the overhead of the parallel region.
   const std::int64_t work = static_cast< std::int64_t >( offsets[ end ] ) - offsets[ begin ] + end - begin;
   if( Devices::Host::isOMPEnabled() && work > 4096 ) {
      #pragma omp parallel firstprivate( f )
      {
         const int parts = omp_get_num_threads();
         const int part = omp_get_thread_num();
         const IndexType blockBegin = getBalancedPartitionBoundary( offsets, begin, end, part, parts );
         const IndexType blockEnd = getBalancedPartitionBoundary( offsets, begin, end, part + 1, parts );
         for( IndexType segmentIdx = blockBegin; segmentIdx < blockEnd; segmentIdx++ )
            f( segmentIdx );
      }
      return;
   }
#endif
   for( IndexType segmentIdx = begin; segmentIdx < end; segmentIdx++ )
      f( segmentIdx );
}

}  // namespace TNL::Algorithms::Segments::detail
