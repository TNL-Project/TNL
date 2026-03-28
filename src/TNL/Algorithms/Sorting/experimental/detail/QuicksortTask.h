// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

namespace TNL::Algorithms::Sorting::experimental::detail {

struct QuicksortTask
{
   // start and end position of array to read and write from
   int partitionBegin, partitionEnd;
   //-----------------------------------------------
   // helper variables for blocks working on this task

   int iteration;
   int pivotIdx;
   int dstBegin, dstEnd;
   int firstBlock, blockCount;  // for workers read only values

   __cuda_callable__
   QuicksortTask( int begin, int end, int iteration )
   : partitionBegin( begin ),
     partitionEnd( end ),
     iteration( iteration ),
     pivotIdx( -1 ),
     dstBegin( -151561 ),
     dstEnd( -151561 ),
     firstBlock( -100 ),
     blockCount( -100 )
   {}

   __cuda_callable__
   void
   init( int firstBlock, int blocks, int pivotIdx )
   {
      dstBegin = 0;
      dstEnd = partitionEnd - partitionBegin;
      this->firstBlock = firstBlock;
      blockCount = blocks;
      this->pivotIdx = pivotIdx;
   }

   [[nodiscard]] __cuda_callable__
   int
   getSize() const
   {
      return partitionEnd - partitionBegin;
   }

   QuicksortTask() = default;
};

}  // namespace TNL::Algorithms::Sorting::experimental::detail
