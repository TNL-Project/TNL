// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/NDArray.h>
#include <TNL/Containers/DistributedNDArraySyncDirections.h>

namespace TNL::Containers::detail {

template< typename DistributedNDArray >
struct SynchronizerBuffers
{
   using NDArrayType = NDArray< typename DistributedNDArray::ValueType,
                                typename DistributedNDArray::SizesHolderType,
                                typename DistributedNDArray::PermutationType,
                                typename DistributedNDArray::DeviceType >;
   NDArrayType send_buffer, recv_buffer;
   typename NDArrayType::ViewType send_view, recv_view;
   typename DistributedNDArray::LocalBeginsType send_offsets, recv_offsets;

   SyncDirection direction = SyncDirection::None;

   int neighbor = -1;

   int tag_recv = -1;
   int tag_send = -1;

   Backend::stream_t stream_id = 0;

   SynchronizerBuffers() = delete;

   SynchronizerBuffers( SyncDirection direction ) : direction( direction ) {}

   SynchronizerBuffers( const SynchronizerBuffers& ) = delete;

   SynchronizerBuffers( SynchronizerBuffers&& ) = delete;

   SynchronizerBuffers&
   operator=( const SynchronizerBuffers& ) = delete;

   SynchronizerBuffers&
   operator=( SynchronizerBuffers&& ) = delete;
};

}  // namespace TNL::Containers::detail
