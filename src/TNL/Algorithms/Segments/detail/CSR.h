// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/scan.h>

namespace TNL::Algorithms::Segments::detail {

template< typename Device, typename Index >
class CSR
{
public:
   using DeviceType = Device;
   using IndexType = Index;

   template< typename SizesHolder, typename CSROffsets >
   static void
   setSegmentsSizes( const SizesHolder& sizes, CSROffsets& offsets )
   {
      offsets.setSize( sizes.getSize() + 1 );
      // GOTCHA: when sizes.getSize() == 0, getView returns a full view with size == 1
      if( sizes.getSize() > 0 ) {
         auto view = offsets.getView( 0, sizes.getSize() );
         view = sizes;
      }
      offsets.setElement( sizes.getSize(), 0 );
      inplaceExclusiveScan( offsets );
   }

   template< typename CSROffsets >
   [[nodiscard]] __cuda_callable__
   static IndexType
   getSegmentsCount( const CSROffsets& offsets )
   {
      return offsets.getSize() - 1;
   }

   /***
    * \brief Returns size of the segment number \r segmentIdx
    */
   template< typename CSROffsets >
   [[nodiscard]] __cuda_callable__
   static IndexType
   getSegmentSize( const CSROffsets& offsets, const IndexType segmentIdx )
   {
      if( ! std::is_same< DeviceType, Devices::Host >::value ) {
#ifdef __CUDA_ARCH__
         return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
#else
         return offsets.getElement( segmentIdx + 1 ) - offsets.getElement( segmentIdx );
#endif
      }
      return offsets[ segmentIdx + 1 ] - offsets[ segmentIdx ];
   }

   /***
    * \brief Returns number of elements that needs to be allocated.
    */
   template< typename CSROffsets >
   [[nodiscard]] __cuda_callable__
   static IndexType
   getStorageSize( const CSROffsets& offsets )
   {
      if( ! std::is_same< DeviceType, Devices::Host >::value ) {
#ifdef __CUDA_ARCH__
         return offsets[ getSegmentsCount( offsets ) ];
#else
         return offsets.getElement( getSegmentsCount( offsets ) );
#endif
      }
      return offsets[ getSegmentsCount( offsets ) ];
   }
};

}  // namespace TNL::Algorithms::Segments::detail
