// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <limits>

#include <TNL/MPI/Comm.h>
#include <TNL/DiscreteMath.h>

#include "Block.h"
#include "Subrange.h"

namespace TNL::Containers {

/**
 * \brief A helper function which splits a one-dimensional range.
 *
 * \param rangeBegin Beginning of the interval `[rangeBegin, rangeEnd)`.
 * \param rangeEnd End of the interval `[rangeBegin, rangeEnd)`.
 * \param rank Index of the subinterval for which the output is calculated.
 * \param num_subintervals Total number of subintervals.
 * \return A subrange `[begin, end)` for the specified rank.
 */
template< typename Index >
[[nodiscard]] Subrange< Index >
splitRange( Index rangeBegin, Index rangeEnd, int rank, int num_subintervals )
{
   const Index rangeSize = rangeEnd - rangeBegin;
   const Index partSize = rangeSize / num_subintervals;
   const int remainder = rangeSize % num_subintervals;
   if( rank < remainder ) {
      const Index begin = rangeBegin + rank * ( partSize + 1 );
      const Index end = begin + partSize + 1;
      return { begin, end };
   }
   const Index begin = rangeBegin + remainder * ( partSize + 1 ) + ( rank - remainder ) * partSize;
   const Index end = begin + partSize;
   return { begin, end };
}

/**
 * \brief A helper function which splits a one-dimensional range.
 *
 * \param globalSize Size of the global range `[0, globalSize)` to be split.
 * \param communicator MPI communicator consisting of ranks for which the range
 *                     is split.
 * \return A subrange `[begin, end)` for the calling MPI rank.
 */
template< typename Index >
[[nodiscard]] Subrange< Index >
splitRange( Index globalSize, const MPI::Comm& communicator )
{
   if( communicator == MPI_COMM_NULL )
      return {};

   const int rank = communicator.rank();
   const int partitions = communicator.size();
   return splitRange( Index( 0 ), globalSize, rank, partitions );
}

/**
 * \brief Decompose a "global" block into several (sub-)blocks in a 3D manner
 * with given block counts along each axis.
 *
 * \param global The large block to decompose.
 * \param num_x Number of blocks along the x-axis.
 * \param num_y Number of blocks along the y-axis.
 * \param num_z Number of blocks along the z-axis.
 * \return A vector of the blocks into which the input was decomposed.
 */
template< typename Index >
std::vector< Block< 3, Index > >
decomposeBlock( const Block< 3, Index >& global, Index num_x, Index num_y = 1, Index num_z = 1 )
{
   std::vector< Block< 3, Index > > result;

   for( Index block_z = 0; block_z < num_z; block_z++ ) {
      // split the range along the z-axis
      const auto range_z = splitRange( global.begin.z(), global.end.z(), block_z, num_z );

      for( Index block_y = 0; block_y < num_y; block_y++ ) {
         // split the range along the y-axis
         const auto range_y = splitRange( global.begin.y(), global.end.y(), block_y, num_y );

         for( Index block_x = 0; block_x < num_x; block_x++ ) {
            // split the range along the x-axis
            const auto range_x = splitRange( global.begin.x(), global.end.x(), block_x, num_x );

            // add new block and initialize it with the global size
            Block< 3, Index >& block = result.emplace_back( global );

            // set the begin/end values for all axes
            block.begin.x() = range_x.getBegin();
            block.end.x() = range_x.getEnd();
            block.begin.y() = range_y.getBegin();
            block.end.y() = range_y.getEnd();
            block.begin.z() = range_z.getBegin();
            block.end.z() = range_z.getEnd();
         }
      }
   }

   return result;
}

/**
 * \brief Decompose a "global" block into several (sub-)blocks in an optimal 3D
 * manner.
 *
 * \param global The large block to decompose.
 * \param num_blocks Number of blocks.
 * \return A vector of the blocks into which the input was decomposed.
 */
template< typename Index >
std::vector< Block< 3, Index > >
decomposeBlockOptimal( const Block< 3, Index >& global, Index num_blocks )
{
   std::vector< Block< 3, Index > > best;
   Index best_interface_area = std::numeric_limits< Index >::max();

   for( const auto& [ num_x, num_y, num_z ] : integerFactorizationTuples< 3 >( num_blocks ) ) {
      const std::vector< Block< 3, Index > > decomposition = decomposeBlock( global, num_x, num_y, num_z );
      const Index interface_area = getInterfaceArea( decomposition );
      if( interface_area < best_interface_area ) {
         best = decomposition;
         best_interface_area = interface_area;
      }
   }

   return best;
}

}  // namespace TNL::Containers
