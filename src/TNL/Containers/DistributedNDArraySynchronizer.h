// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <map>
#include <stdexcept>
#include <future>
// 3rd-party thread pool library
#include <TNL/3rdparty/BS_thread_pool_light.hpp>

#include <TNL/MPI/Comm.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/Timer.h>

#include "DistributedNDArraySyncDirections.h"
#include "ndarray/SynchronizerBuffers.h"

namespace TNL::Containers {

/**
 * \brief Synchronizer for \ref DistributedNDArray.
 *
 * \ingroup ndarray
 */
template< typename DistributedNDArray >
class DistributedNDArraySynchronizer
{
private:
   BS::thread_pool_light tp;

   int gpu_id = 0;

   int tag_offset = 0;

   static int
   reserve_tags( int count )
   {
      static int offset = 0;
      // we could use a post-increment, but we don't have to start from 0 either...
      return offset += count;
   }

   using DistributedNDArrayView = typename DistributedNDArray::ViewType;
   using Buffers = std::map< SyncDirection, detail::SynchronizerBuffers< DistributedNDArray > >;

   DistributedNDArrayView array_view;
   SyncDirection mask = SyncDirection::All;
   Buffers buffers;

public:
   using RequestsVector = std::vector< MPI_Request >;
   RequestsVector requests;

   enum class AsyncPolicy
   {
      synchronous,
      deferred,
      threadpool,
      async,
   };

   // DistributedNDArraySynchronizer(int max_threads = std::thread::hardware_concurrency())
   DistributedNDArraySynchronizer( int max_threads = 1 )
   : tp( max_threads ),
     // reserve tags for all directions (see how we set the default tags below)
     tag_offset( reserve_tags( static_cast< std::uint8_t >( SyncDirection::All ) ) )
   {}

   // BS::thread_pool_light is not move-constructible (due to std::atomic), so we need
   // custom move-constructor that skips moving tp
   DistributedNDArraySynchronizer( DistributedNDArraySynchronizer&& other ) noexcept
   : tp( other.tp.get_thread_count() ), gpu_id( std::move( other.gpu_id ) ), tag_offset( std::move( other.tag_offset ) ),
     array_view( std::move( other.array_view ) ), mask( std::move( other.mask ) ), buffers( std::move( other.buffers ) ),
     requests( std::move( other.requests ) )
   {}

   /**
    * \brief Set the communication pattern between neighbors during data
    * synchronization.
    *
    * \tparam Q is the number of elements in \e pattern.
    * \param pattern is the synchronization pattern (array of directions
    *                in which the data will be sent). It must be consistent
    *                with the partitioning of the distributed array.
    */
   template< std::size_t Q >
   void
   setSynchronizationPattern( const std::array< SyncDirection, Q >& pattern )
   {
      buffers.clear();
      for( SyncDirection direction : pattern ) {
         buffers.emplace( std::make_pair( direction, direction ) );
      }
   }

   void
   setNeighbor( SyncDirection direction, int neighbor )
   {
      buffers.at( direction ).neighbor = neighbor;
   }

   void
   setTagOffset( int offset )
   {
      tag_offset = offset;
   }

   void
   setTags( SyncDirection direction, int tag_recv, int tag_send )
   {
      buffers.at( direction ).tag_recv = tag_recv;
      buffers.at( direction ).tag_send = tag_send;
   }

   void
   setCudaStream( SyncDirection direction, cudaStream_t stream_id )
   {
      buffers.at( direction ).stream_id = stream_id;
   }

   // TODO: update for multidimensional decomposition (if it makes sense)
   // special thing for the A-A pattern in LBM
   void
   setBuffersShift( int shift )
   {
      constexpr int dim = getFirstDimensionWithOverlap();
      constexpr int overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
      if( overlap == 0 )
         return;

      using LocalBegins = typename DistributedNDArray::LocalBeginsType;
      using SizesHolder = typename DistributedNDArray::SizesHolderType;
      const LocalBegins& localBegins = array_view.getLocalBegins();
      const SizesHolder& localEnds = array_view.getLocalEnds();

      auto& buffer_left = buffers.at( SyncDirection::Left );
      auto& buffer_right = buffers.at( SyncDirection::Right );

      // offsets for left-send (local indexing for the local array)
      buffer_left.send_offsets = LocalBegins{};
      buffer_left.send_offsets.template setSize< dim >( -shift );

      // offsets for left-receive (local indexing for the local array)
      buffer_left.recv_offsets = LocalBegins{};
      buffer_left.recv_offsets.template setSize< dim >( -overlap + shift );

      // offsets for right-send (local indexing for the local array)
      buffer_right.send_offsets = LocalBegins{};
      buffer_right.send_offsets.template setSize< dim >( localEnds.template getSize< dim >()
                                                         - localBegins.template getSize< dim >() - overlap + shift );

      // offsets for right-receive (local indexing for the local array)
      buffer_right.recv_offsets = LocalBegins{};
      buffer_right.recv_offsets.template setSize< dim >( localEnds.template getSize< dim >()
                                                         - localBegins.template getSize< dim >() - shift );
   }

   /**
    * \brief Synchronizes data in \e array distributed among MPI ranks.
    *
    * \param array is the distributed array to be synchronized.
    * \param mask can be used to suppress specific directions from the
    *             pattern set with \ref setSynchronizationPattern (useful
    *             e.g. for the lattice Boltzmann method).
    */
   void
   synchronize( DistributedNDArray& array, SyncDirection mask = SyncDirection::All )
   {
      synchronize( AsyncPolicy::synchronous, array, mask );
   }

   /**
    * \brief Synchronizes data in \e array distributed among MPI ranks.
    *
    * This method is not thread-safe - only the thread which created and
    * "owns" the instance of this object can call this method.
    *
    * Also note that this method must not be called again until the previous
    * asynchronous operation has finished.
    *
    * \param policy determines the async policy used by the synchronizer.
    * \param array is the distributed array to be synchronized.
    * \param mask can be used to suppress specific directions from the
    *             pattern set with \ref setSynchronizationPattern (useful
    *             e.g. for the lattice Boltzmann method).
    */
   void
   synchronize( AsyncPolicy policy, DistributedNDArray& array, SyncDirection mask = SyncDirection::All )
   {
      // wait for any previous synchronization (multiple objects can share the
      // same synchronizer)
      wait();

      async_start_timer.start();

      // stage 0: set inputs, allocate buffers
      stage_0( array, mask );

      // everything offloaded to a separate thread
      if( policy == AsyncPolicy::threadpool || policy == AsyncPolicy::async ) {
         auto worker = [ this ]()
         {
            // set the GPU id, see this gotcha:
            // GOTCHA: https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
            if constexpr( std::is_same< typename DistributedNDArray::DeviceType, Devices::Cuda >::value )
               cudaSetDevice( this->gpu_id );

            // stage 1: fill send buffers
            this->stage_1();
            // stage 2: issue all send and receive async operations
            this->stage_2();
            // stage 3: copy data from receive buffers
            this->stage_3();
            // stage 4: ensure everything has finished
            this->stage_4();
         };

         if( policy == AsyncPolicy::threadpool )
            async_op = tp.submit( worker );
         else
            async_op = std::async( std::launch::async, worker );
      }
      // immediate start, deferred synchronization (but still in the same thread)
      else if( policy == AsyncPolicy::deferred ) {
         // stage 1: fill send buffers
         this->stage_1();
         // stage 2: issue all send and receive async operations
         this->stage_2();
         auto worker = [ this ]() mutable
         {
            // stage 3: copy data from receive buffers
            this->stage_3();
            // stage 4: ensure everything has finished
            this->stage_4();
         };
         this->async_op = std::async( std::launch::deferred, worker );
      }
      // synchronous
      else {
         // stage 1: fill send buffers
         this->stage_1();
         // stage 2: issue all send and receive async operations
         this->stage_2();
         // stage 3: copy data from receive buffers
         this->stage_3();
         // stage 4: ensure everything has finished
         this->stage_4();
      }

      async_ops_count++;
      async_start_timer.stop();
   }

   void
   wait()
   {
      if( async_op.valid() ) {
         async_wait_timer.start();
         async_op.wait();
         async_wait_timer.stop();
      }
   }

   ~DistributedNDArraySynchronizer()
   {
      if( this->async_op.valid() )
         this->async_op.wait();
   }

   /**
    * \brief Can be used for checking if a synchronization started
    * asynchronously has been finished.
    */
   std::future< void > async_op;

   // attributes for profiling
   Timer async_start_timer, async_wait_timer;
   std::size_t async_ops_count = 0;
   std::size_t sent_bytes = 0;
   std::size_t recv_bytes = 0;
   std::size_t sent_messages = 0;
   std::size_t recv_messages = 0;

   // stage 0: set inputs, allocate buffers
   void
   stage_0( DistributedNDArray& array, SyncDirection mask )
   {
      if( buffers.empty() )
         throw std::logic_error(
            "the buffers are empty - make sure that setSynchronizationPattern is called before synchronization" );

      // save the GPU id to be restored in async threads, see this gotcha:
      // https://devblogs.nvidia.com/cuda-pro-tip-always-set-current-device-avoid-multithreading-bugs/
      if constexpr( std::is_same< typename DistributedNDArray::DeviceType, Devices::Cuda >::value )
         cudaGetDevice( &this->gpu_id );

      // skip allocation on repeated calls - compare only sizes, not the actual data
      if( array_view.getCommunicator() != array.getCommunicator() || array_view.getSizes() != array.getSizes()
          || array_view.getLocalBegins() != array.getLocalBegins() || array_view.getLocalEnds() != array.getLocalEnds() )
      {
         array_view.bind( array.getView() );
         this->mask = mask;

         // allocate buffers
         allocateHelper( buffers, array_view, tag_offset );
      }
      else {
         // only bind to the actual data
         array_view.bind( array.getView() );
         this->mask = mask;
      }

      // clear profiling counters
      sent_bytes = recv_bytes = 0;
      sent_messages = recv_messages = 0;
   }

   // stage 1: fill send buffers
   void
   stage_1()
   {
      copyHelper( buffers, array_view, true, mask );
   }

   // stage 2: issue all send and receive async operations
   void
   stage_2()
   {
      // synchronize all CUDA streams to ensure the previous stage is finished
      if constexpr( std::is_same< typename DistributedNDArrayView::DeviceType, Devices::Cuda >::value ) {
         for( auto& [ _, buffer ] : buffers )
            cudaStreamSynchronize( buffer.stream_id );
         TNL_CHECK_CUDA_DEVICE;
      }

      // issue all send and receive async operations
      requests.clear();
      const MPI::Comm& communicator = array_view.getCommunicator();
      sendHelper( buffers, requests, communicator, mask, sent_bytes, recv_bytes, sent_messages, recv_messages );
   }

   // stage 3: copy data from receive buffers
   void
   stage_3()
   {
      // wait for all data to arrive
      MPI::Waitall( requests.data(), requests.size() );

      // copy data from receive buffers
      copyHelper( buffers, array_view, false, mask );
   }

   // stage 4: ensure everything has finished
   void
   stage_4()
   {
      // synchronize all CUDA streams
      if constexpr( std::is_same< typename DistributedNDArrayView::DeviceType, Devices::Cuda >::value ) {
         for( auto& [ _, buffer ] : buffers )
            cudaStreamSynchronize( buffer.stream_id );
         TNL_CHECK_CUDA_DEVICE;
      }
   }

protected:
   static constexpr int
   countDimensionsWithOverlap()
   {
      int count = 0;
      Algorithms::staticFor< std::size_t, 0, DistributedNDArray::getDimension() >(
         [ & ]( auto dim )
         {
            constexpr int overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
            if( overlap > 0 )
               count++;
         } );
      return count;
   }

   static constexpr int
   getFirstDimensionWithOverlap()
   {
      int firstDim = DistributedNDArray::getDimension();
      Algorithms::staticFor< std::size_t, 0, DistributedNDArray::getDimension() >(
         [ & ]( auto dim )
         {
            constexpr int overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
            if( overlap > 0 && dim < firstDim )
               firstDim = dim;
         } );
      return firstDim;
   }

   static void
   allocateHelper( Buffers& buffers, const DistributedNDArrayView& array_view, int tag_offset )
   {
      constexpr int dim = getFirstDimensionWithOverlap();
      constexpr int overlap = DistributedNDArrayView::LocalViewType::IndexerType::template getOverlap< dim >();
      if( overlap == 0 )
         return;

      using LocalBegins = typename DistributedNDArray::LocalBeginsType;
      using SizesHolder = typename DistributedNDArray::SizesHolderType;
      const LocalBegins& localBegins = array_view.getLocalBegins();
      const SizesHolder& localEnds = array_view.getLocalEnds();

      // TODO: update for multidimensional decomposition
      SizesHolder bufferSize( localEnds );
      bufferSize.template setSize< dim >( overlap );

      // allocate buffers
      auto& buffer_left = buffers.at( SyncDirection::Left );
      auto& buffer_right = buffers.at( SyncDirection::Right );
      buffer_left.send_buffer.setSize( bufferSize );
      buffer_left.recv_buffer.setSize( bufferSize );
      buffer_right.send_buffer.setSize( bufferSize );
      buffer_right.recv_buffer.setSize( bufferSize );

      // bind views to the buffers
      buffer_left.send_view.bind( buffer_left.send_buffer.getView() );
      buffer_left.recv_view.bind( buffer_left.recv_buffer.getView() );
      buffer_right.send_view.bind( buffer_right.send_buffer.getView() );
      buffer_right.recv_view.bind( buffer_right.recv_buffer.getView() );

      // TODO: check overlap offsets for 2D and 3D distributions (watch out for the corners - maybe use
      // SetSizesSubtractOverlapsHelper?)

      // offsets for left-send (local indexing for the local array)
      buffer_left.send_offsets = LocalBegins{};

      // offsets for left-receive (local indexing for the local array)
      buffer_left.recv_offsets = LocalBegins{};
      buffer_left.recv_offsets.template setSize< dim >( -overlap );

      // offsets for right-send (local indexing for the local array)
      buffer_right.send_offsets = LocalBegins{};
      buffer_right.send_offsets.template setSize< dim >( localEnds.template getSize< dim >()
                                                         - localBegins.template getSize< dim >() - overlap );

      // offsets for right-receive (local indexing for the local array)
      buffer_right.recv_offsets = LocalBegins{};
      buffer_right.recv_offsets.template setSize< dim >( localEnds.template getSize< dim >()
                                                         - localBegins.template getSize< dim >() );

      // set default neighbor IDs
      const MPI::Comm& communicator = array_view.getCommunicator();
      const int rank = communicator.rank();
      const int nproc = communicator.size();
      if( buffer_left.neighbor < 0 )
         buffer_left.neighbor = ( rank + nproc - 1 ) % nproc;
      if( buffer_right.neighbor < 0 )
         buffer_right.neighbor = ( rank + 1 ) % nproc;

      // set default tags from tag_offset
      for( auto& [ direction, buffer ] : buffers ) {
         if( buffer.tag_recv < 0 && buffer.tag_send < 0 ) {
            buffer.tag_recv = tag_offset + static_cast< std::uint8_t >( opposite( direction ) );
            buffer.tag_send = tag_offset + static_cast< std::uint8_t >( direction );
         }
      }
   }

   template< typename LaunchConfiguration >
   static void
   setCudaStream( LaunchConfiguration& launch_config, cudaStream_t stream )
   {}

   static void
   setCudaStream( Devices::Cuda::LaunchConfiguration& launch_config, cudaStream_t stream )
   {
      launch_config.stream = stream;
      launch_config.blockHostUntilFinished = false;
   }

   static void
   copyHelper( Buffers& buffers, DistributedNDArrayView& array_view, bool to_buffer, SyncDirection mask )
   {
      for( auto& [ _, buffer ] : buffers ) {
         // check if buffering is needed at runtime
         // GOTCHA: send_buffer.getSizes() may have a static size in some dimension, which the LocalBegins does not have
         typename DistributedNDArray::SizesHolderType ends = buffer.send_buffer.getSizes() + buffer.send_offsets;
         const bool is_contiguous = array_view.getLocalView().isContiguousBlock( buffer.send_offsets, ends );

         if( is_contiguous ) {
            // avoid buffering - bind buffer views directly to the array
            buffer.send_view.bind( &call_with_offsets( buffer.send_offsets, array_view.getLocalView() ) );
            buffer.recv_view.bind( &call_with_offsets( buffer.recv_offsets, array_view.getLocalView() ) );
         }
         else {
            CopyKernel< decltype( buffer.send_view ) > copy_kernel;
            copy_kernel.local_array_view.bind( array_view.getLocalView() );
            copy_kernel.to_buffer = to_buffer;

            // create launch configuration to specify the CUDA stream
            typename DistributedNDArray::DeviceType::LaunchConfiguration launch_config;

            if( to_buffer ) {
               if( ( mask & buffer.direction ) != SyncDirection::None ) {
                  copy_kernel.buffer_view.bind( buffer.send_view );
                  copy_kernel.local_array_offsets = buffer.send_offsets;
                  setCudaStream( launch_config, buffer.stream_id );
                  buffer.send_view.forAll( copy_kernel, launch_config );
               }
            }
            else {
               if( ( mask & opposite( buffer.direction ) ) != SyncDirection::None ) {
                  copy_kernel.buffer_view.bind( buffer.recv_view );
                  copy_kernel.local_array_offsets = buffer.recv_offsets;
                  setCudaStream( launch_config, buffer.stream_id );
                  buffer.recv_view.forAll( copy_kernel, launch_config );
               }
            }
         }
      }
   }

   static void
   sendHelper( Buffers& buffers,
               RequestsVector& requests,
               const MPI::Comm& communicator,
               SyncDirection mask,
               std::size_t& sent_bytes,
               std::size_t& recv_bytes,
               std::size_t& sent_messages,
               std::size_t& recv_messages )
   {
      for( auto& [ _, buffer ] : buffers ) {
         if( ( mask & buffer.direction ) != SyncDirection::None ) {
            // negative rank and tag IDs are not valid according to the MPI standard and may be used by
            // applications to skip communication, e.g. over the periodic boundary
            if( buffer.neighbor >= 0 && buffer.tag_send >= 0 ) {
               requests.push_back( MPI::Isend( buffer.send_view.getData(),
                                               buffer.send_view.getStorageSize(),
                                               buffer.neighbor,
                                               buffer.tag_send,
                                               communicator ) );
               sent_bytes += buffer.send_view.getStorageSize() * sizeof( typename DistributedNDArray::ValueType );
               ++sent_messages;
            }
            auto& opp_buffer = buffers.at( opposite( buffer.direction ) );
            if( opp_buffer.neighbor >= 0 && opp_buffer.tag_recv >= 0 ) {
               requests.push_back( MPI::Irecv( opp_buffer.recv_view.getData(),
                                               opp_buffer.recv_view.getStorageSize(),
                                               opp_buffer.neighbor,
                                               opp_buffer.tag_recv,
                                               communicator ) );
               recv_bytes += opp_buffer.recv_view.getStorageSize() * sizeof( typename DistributedNDArray::ValueType );
               ++recv_messages;
            }
         }
      }
   }

#ifdef __NVCC__
public:
#endif
   template< typename BufferView >
   struct CopyKernel
   {
      using LocalArrayView = typename DistributedNDArray::LocalViewType;
      using LocalBegins = typename DistributedNDArray::LocalBeginsType;

      BufferView buffer_view;
      LocalArrayView local_array_view;
      LocalBegins local_array_offsets;
      bool to_buffer;

      template< typename... Indices >
      __cuda_callable__
      void
      operator()( Indices... indices )
      {
         if( to_buffer )
            buffer_view( indices... ) = call_with_shifted_indices( local_array_offsets, local_array_view, indices... );
         else
            call_with_shifted_indices( local_array_offsets, local_array_view, indices... ) = buffer_view( indices... );
      }
   };
};

}  // namespace TNL::Containers
