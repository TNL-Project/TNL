// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "Subrange.h"

#include <TNL/MPI/Comm.h>

namespace TNL::Containers {

template< typename Index >
class Partitioner
{
public:
   using SubrangeType = Subrange< Index >;

   [[nodiscard]] static SubrangeType
   splitRange( Index globalSize, const MPI::Comm& communicator )
   {
      if( communicator == MPI_COMM_NULL )
         return { 0, 0 };

      const int rank = communicator.rank();
      const int partitions = communicator.size();

      const Index partSize = globalSize / partitions;
      const int remainder = globalSize % partitions;
      if( rank < remainder ) {
         const Index begin = rank * ( partSize + 1 );
         const Index end = begin + partSize + 1;
         return { begin, end };
      }
      const Index begin = remainder * ( partSize + 1 ) + ( rank - remainder ) * partSize;
      const Index end = begin + partSize;
      return { begin, end };
   }

   // Gets the offset of data for given rank.
   [[nodiscard]] __cuda_callable__
   static Index
   getOffset( Index globalSize, int rank, int partitions )
   {
      const Index partSize = globalSize / partitions;
      const int remainder = globalSize % partitions;
      if( rank < remainder )
         return rank * ( partSize + 1 );
      return remainder * ( partSize + 1 ) + ( rank - remainder ) * partSize;
   }

   // Gets the size of data assigned to given rank.
   [[nodiscard]] __cuda_callable__
   static Index
   getSizeForRank( Index globalSize, int rank, int partitions )
   {
      const Index partSize = globalSize / partitions;
      const int remainder = globalSize % partitions;
      if( rank < remainder )
         return partSize + 1;
      return partSize;
   }
};

}  // namespace TNL::Containers
