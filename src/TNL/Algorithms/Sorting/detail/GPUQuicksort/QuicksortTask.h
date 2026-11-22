// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend/Macros.h>

namespace TNL::Algorithms::Sorting::detail {

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

   /**
    * \brief Returns a task for the range `[begin, end)` with the helper
    * variables set to sentinel values that mark the task as not yet claimed
    * by any block.
    *
    * The struct deliberately has no user-provided constructor: as an
    * aggregate of fundamental types it stays trivially constructible and
    * trivially destructible, which lets the task storage skip element
    * construction and destruction (only the slots claimed by the task
    * counters are ever read and each of them is written before the read).
    *
    * \param begin index of the first element of the partitioned range
    * \param end index one past the last element of the partitioned range
    * \param iteration index of the partitioning round the task belongs to
    */
   [[nodiscard]] __cuda_callable__
   static QuicksortTask
   make( Index begin, Index end, int iteration )
   {
      return QuicksortTask{
         begin, end,  iteration, static_cast< Index >( -1 ), static_cast< Offset >( -151561 ), static_cast< Offset >( -151561 ),
         -100,  -100,
      };
   }

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
};

}  // namespace TNL::Algorithms::Sorting::detail
