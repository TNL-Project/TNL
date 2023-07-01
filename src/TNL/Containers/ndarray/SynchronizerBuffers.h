// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Containers/NDArray.h>

namespace TNL::Containers::detail {

template< typename DistributedNDArray >
struct SynchronizerBuffer
{
   using NDArrayType = NDArray< typename DistributedNDArray::ValueType,
                                typename DistributedNDArray::SizesHolderType,
                                typename DistributedNDArray::PermutationType,
                                typename DistributedNDArray::DeviceType >;
   NDArrayType send_buffer, recv_buffer;
   typename NDArrayType::ViewType send_view, recv_view;
   typename DistributedNDArray::LocalBeginsType send_offsets, recv_offsets;

   int neighbor = -1;

   int tag_recv = -1;
   int tag_send = -1;

   cudaStream_t stream_id = 0;

   void
   reset()
   {
      send_buffer.reset();
      recv_buffer.reset();

      send_view.reset();
      recv_view.reset();

      send_offsets = recv_offsets = typename DistributedNDArray::LocalBeginsType{};

      neighbor = -1;

      tag_recv = tag_send = -1;

      stream_id = 0;
   }
};

template< typename DistributedNDArray, std::size_t level >
struct SynchronizerBuffersLayer
{
   [[nodiscard]] SynchronizerBuffersLayer&
   getDimBuffers( std::integral_constant< std::size_t, level > )
   {
      return *this;
   }

   SynchronizerBuffer< DistributedNDArray > left;
   SynchronizerBuffer< DistributedNDArray > right;

   void
   reset()
   {
      left.reset();
      right.reset();
   }
};

template< typename DistributedNDArray,
          typename LevelTag = std::integral_constant< std::size_t, DistributedNDArray::getDimension() > >
struct SynchronizerBuffersLayerHelper
{};

template< typename DistributedNDArray, std::size_t level >
struct SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, level > >
: public SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, level - 1 > >,
  public SynchronizerBuffersLayer< DistributedNDArray, level >
{
   using SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, level - 1 > >::getDimBuffers;
   using SynchronizerBuffersLayer< DistributedNDArray, level >::getDimBuffers;
};

template< typename DistributedNDArray >
struct SynchronizerBuffersLayerHelper< DistributedNDArray, std::integral_constant< std::size_t, 0 > >
: public SynchronizerBuffersLayer< DistributedNDArray, 0 >
{
   using SynchronizerBuffersLayer< DistributedNDArray, 0 >::getDimBuffers;
};

template< typename DistributedNDArray >
struct SynchronizerBuffers : public SynchronizerBuffersLayerHelper< DistributedNDArray >
{
   using SynchronizerBuffersLayerHelper< DistributedNDArray >::getDimBuffers;

   template< std::size_t level >
   [[nodiscard]] auto&
   getDimBuffers()
   {
      return this->getDimBuffers( std::integral_constant< std::size_t, level >{} );
   }
};

}  // namespace TNL::Containers::detail
