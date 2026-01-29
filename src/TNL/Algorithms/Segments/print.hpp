// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/Array.h>
#include <ostream>
#include <utility>

namespace TNL::Algorithms::Segments {

template< typename Segments, typename T >
std::ostream&
operator<<( std::ostream& str, const Segments& segments )
{
   using IndexType = typename Segments::IndexType;
   auto segmentsCount = segments.getSegmentCount();
   str << " [";
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      auto segmentSize = segments.getSegmentSize( segmentIdx );
      str << " " << segmentSize;
      if( segmentIdx < segmentsCount - 1 )
         str << ",";
   }
   str << " ] ";
   return str;
}

template< typename SegmentsView, typename Fetch >
struct SegmentsPrinter
{
    SegmentsPrinter( SegmentsView segments, Fetch fetch )
    : segments( std::move( segments ) ),
       fetch( std::move( fetch ) )
    {}

    const SegmentsView segments;
    Fetch fetch;
};

template< typename SegmentsView, typename Fetch >
std::ostream&
operator<<( std::ostream& str, const SegmentsPrinter< SegmentsView, Fetch >& printer )
{
   using IndexType = typename SegmentsView::IndexType;
   using DeviceType = typename SegmentsView::DeviceType;

   using ValueType = decltype( printer.fetch( IndexType() ) );

   TNL::Containers::Array< ValueType, DeviceType > aux( 1 );
   for( IndexType segmentIdx = 0; segmentIdx < printer.segments.getSegmentCount(); segmentIdx++ ) {
      str << "Segment " << segmentIdx << ": [ ";
      const IndexType segmentSize = printer.segments.getSegmentSize( segmentIdx );
      for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ ) {
         aux.forAllElements(
            [ = ] __cuda_callable__( IndexType elementIdx, ValueType & v ) mutable
            {
               v = printer.fetch( printer.segments.getGlobalIndex( segmentIdx, localIdx ) );
            } );
         str << aux.getElement( 0 );
         if( localIdx < segmentSize - 1 )
            str << ", ";
      }
      str << " ]";
      if( segmentIdx < printer.segments.getSegmentCount() - 1 )
         str << '\n';
   }
   return str;
}

template< typename Segments, typename Fetch, std::enable_if_t< isSegments_v< Segments >, bool > >
SegmentsPrinter< typename Segments::ConstViewType, Fetch >
print( const Segments& segments, Fetch fetch )  // TODO: Fetch&& does not work with CUDA
{
   return SegmentsPrinter< typename Segments::ConstViewType, Fetch >( segments.getConstView(), std::forward< Fetch >( fetch ) );
}

}  // namespace TNL::Algorithms::Segments
