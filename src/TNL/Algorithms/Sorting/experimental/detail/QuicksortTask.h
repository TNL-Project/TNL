// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

namespace TNL::Algorithms::Sorting::experimental::detail {

template< typename Index >
struct QuicksortTask
{
   using Offset = std::make_signed_t< Index >;

   Index partitionBegin, partitionEnd;
   //-----------------------------------------------
   // helper variables for blocks working on this task

   int iteration;
   Index pivotIdx;
   Offset dstBegin, dstEnd;
   int firstBlock, blockCount;

   __cuda_callable__
   QuicksortTask( Index begin, Index end, int iteration )
   : partitionBegin( begin ),
     partitionEnd( end ),
     iteration( iteration ),
     pivotIdx( static_cast< Index >( -1 ) ),
     dstBegin( static_cast< Offset >( -151561 ) ),
     dstEnd( static_cast< Offset >( -151561 ) ),
     firstBlock( -100 ),
     blockCount( -100 )
   {}

   __cuda_callable__
   void
   init( int firstBlock, int blocks, Index pivotIdx )
   {
      dstBegin = 0;
      dstEnd = partitionEnd - partitionBegin;
      this->firstBlock = firstBlock;
      blockCount = blocks;
      this->pivotIdx = pivotIdx;
   }

   [[nodiscard]] __cuda_callable__
   Index
   getSize() const
   {
      return partitionEnd - partitionBegin;
   }

   QuicksortTask() = default;
};

}  // namespace TNL::Algorithms::Sorting::experimental::detail
