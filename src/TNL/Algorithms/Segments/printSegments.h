// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <ostream>

#include <TNL/Containers/Array.h>

namespace TNL::Algorithms::Segments {

/**
 * \brief Print segments sizes, i.e. the segments setup.
 *
 * \tparam Segments is type of segments.
 * \param segments is an instance of segments.
 * \param str is output stream.
 * \return reference to the output stream.
 *
 * \par Example
 * \include Algorithms/Segments/printSegmentsExample-1.cpp
 * \par Output
 * \include printSegmentsExample-1.out
 */
template< typename Segments >
[[deprecated( "Use `operator <<` instead." )]] std::ostream&
printSegments( std::ostream& str, const Segments& segments )
{
   using IndexType = typename Segments::IndexType;

   auto segmentsCount = segments.getSegmentCount();
   str << " [";
   for( IndexType segmentIdx = 0; segmentIdx < segmentsCount; segmentIdx++ ) {
      auto segmentSize = segments.getSegmentSize( segmentIdx );
      str << " " << segmentSize;
      if( segmentIdx < segmentsCount )
         str << ",";
   }
   str << " ] ";
   return str;
}

template< typename Segments, typename Fetch >
[[deprecated( "Use `std::ostream << TNL::Algorithms::Segments::print( segments, fetch)` instead." )]] std::ostream&
printSegments( std::ostream& str, const Segments& segments, Fetch&& fetch )
{
   using IndexType = typename Segments::IndexType;
   using DeviceType = typename Segments::DeviceType;
   using ValueType = decltype( fetch( IndexType() ) );

   TNL::Containers::Array< ValueType, DeviceType > aux( 1 );
   auto view = segments.getConstView();
   for( IndexType segmentIdx = 0; segmentIdx < segments.getSegmentCount(); segmentIdx++ ) {
      str << "Seg. " << segmentIdx << ": [ ";
      const IndexType segmentSize = segments.getSegmentSize( segmentIdx );
      for( IndexType localIdx = 0; localIdx < segmentSize; localIdx++ ) {
         aux.forAllElements(
            [ = ] __cuda_callable__( IndexType elementIdx, ValueType & v ) mutable
            {
               v = fetch( view.getGlobalIndex( segmentIdx, localIdx ) );
            } );
         str << aux.getElement( 0 );
         if( localIdx < segmentSize - 1 )
            str << ", ";
      }
      str << " ]\n";
   }
   return str;
}

}  // namespace TNL::Algorithms::Segments
