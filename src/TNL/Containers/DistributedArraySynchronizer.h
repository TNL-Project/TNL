// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

#include "Subrange.h"
#include "ByteArraySynchronizer.h"

#include <TNL/MPI/Comm.h>

namespace TNL::Containers {

template< typename DistributedArray >
class DistributedArraySynchronizer
: public ByteArraySynchronizer< typename DistributedArray::DeviceType, typename DistributedArray::IndexType >
{
   using Base = ByteArraySynchronizer< typename DistributedArray::DeviceType, typename DistributedArray::IndexType >;
   using SubrangeType = Subrange< typename DistributedArray::IndexType >;

   SubrangeType localRange;
   int overlaps;
   MPI::Comm communicator;

public:
   using ByteArrayView = typename Base::ByteArrayView;
   using RequestsVector = typename Base::RequestsVector;

   ~DistributedArraySynchronizer() override
   {
      // wait for pending async operation, otherwise it would crash
      if( this->async_op.valid() )
         this->async_op.wait();
   }

   DistributedArraySynchronizer() = delete;

   DistributedArraySynchronizer( SubrangeType localRange, int overlaps, MPI::Comm communicator )
   : localRange( localRange ),
     overlaps( overlaps ),
     communicator( std::move( communicator ) )
   {}

   void
   synchronizeByteArray( ByteArrayView array, int bytesPerValue ) override
   {
      auto requests = synchronizeByteArrayAsyncWorker( array, bytesPerValue );
      MPI::Waitall( requests.data(), requests.size() );
   }

   [[nodiscard]] RequestsVector
   synchronizeByteArrayAsyncWorker( ByteArrayView array, int bytesPerValue ) override
   {
      if( array.getSize() != bytesPerValue * ( localRange.getSize() + 2 * overlaps ) )
         throw std::logic_error( "synchronizeByteArrayAsyncWorker: unexpected array size" );

      const int rank = communicator.rank();
      const int nproc = communicator.size();
      const int left = ( rank > 0 ) ? rank - 1 : nproc - 1;
      const int right = ( rank < nproc - 1 ) ? rank + 1 : 0;

      // buffer for asynchronous communication requests
      RequestsVector requests;

      // issue all async receive operations
      requests.push_back( MPI::Irecv(
         array.getData() + bytesPerValue * localRange.getSize(), bytesPerValue * overlaps, left, 0, communicator ) );
      requests.push_back( MPI::Irecv( array.getData() + bytesPerValue * ( localRange.getSize() + overlaps ),
                                      bytesPerValue * overlaps,
                                      right,
                                      0,
                                      communicator ) );

      // issue all async send operations
      requests.push_back( MPI::Isend( array.getData(), bytesPerValue * overlaps, left, 0, communicator ) );
      requests.push_back( MPI::Isend( array.getData() + bytesPerValue * ( localRange.getSize() - overlaps ),
                                      bytesPerValue * overlaps,
                                      right,
                                      0,
                                      communicator ) );

      return requests;
   }
};

}  // namespace TNL::Containers
