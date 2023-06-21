// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "Subrange.h"

#include <TNL/MPI/Comm.h>

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

}  // namespace TNL::Containers
