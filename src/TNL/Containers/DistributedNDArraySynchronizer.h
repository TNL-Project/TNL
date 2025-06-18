// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
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
   using Buffer = detail::SynchronizerBuffers< DistributedNDArray >;
   using Buffers = std::map< SyncDirection, Buffer >;

   DistributedNDArrayView array_view;
   SyncDirection mask = SyncDirection::All;
   Buffers buffers;

public:
   using RequestsVector = std::vector< MPI_Request >;
   RequestsVector requests;

   enum class AsyncPolicy : std::uint8_t
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
   : tp( other.tp.get_thread_count() ),
     gpu_id( other.gpu_id ),
     tag_offset( other.tag_offset ),
     array_view( std::move( other.array_view ) ),
     mask( other.mask ),
     buffers( std::move( other.buffers ) ),
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
   setCudaStream( SyncDirection direction, Backend::stream_t stream_id )
   {
      buffers.at( direction ).stream_id = stream_id;
   }

   /**
    * \brief Sets the send and receive offsets for all buffer objects.
    *
    * This is primarily an internal function, but applications that require
    * special communication patterns (e.g. the A-A pattern in LBM) can use
    * it to adjust the behavior.
    *
    * \param shift determines by how many cells the offsets are shifted in
    *              the direction configured for the buffer.
    */
   void
   setBufferOffsets( int shift = 0 )
   {
      const int dim0 = getDimensionWithOverlap< 0 >( array_view );
      const int overlap0 = array_view.getOverlaps()[ dim0 ];

      const auto& localBegins = array_view.getLocalBegins();
      const auto& localEnds = array_view.getLocalEnds();

      for( auto& [ direction, buffer ] : buffers ) {
         // initialize offsets (local indexing for the local array)
         buffer.send_offsets = {};
         buffer.recv_offsets = {};

         if( ( direction & SyncDirection::Left ) != SyncDirection::None ) {
            buffer.send_offsets[ dim0 ] = -shift;
            buffer.recv_offsets[ dim0 ] = -overlap0 + shift;
         }
         if( ( direction & SyncDirection::Right ) != SyncDirection::None ) {
            buffer.send_offsets[ dim0 ] = localEnds[ dim0 ] - localBegins[ dim0 ] - overlap0 + shift;
            buffer.recv_offsets[ dim0 ] = localEnds[ dim0 ] - localBegins[ dim0 ] - shift;
         }
         if( ( direction & SyncDirection::Bottom ) != SyncDirection::None ) {
            if( countDimensionsWithOverlap( array_view ) >= 2 ) {
               const int dim1 = getDimensionWithOverlap< 1 >( array_view );
               const int overlap1 = array_view.getOverlaps()[ dim1 ];
               buffer.send_offsets[ dim1 ] = -shift;
               buffer.recv_offsets[ dim1 ] = -overlap1 + shift;
            }
         }
         if( ( direction & SyncDirection::Top ) != SyncDirection::None ) {
            if( countDimensionsWithOverlap( array_view ) >= 2 ) {
               const int dim1 = getDimensionWithOverlap< 1 >( array_view );
               const int overlap1 = array_view.getOverlaps()[ dim1 ];
               buffer.send_offsets[ dim1 ] = localEnds[ dim1 ] - localBegins[ dim1 ] - overlap1 + shift;
               buffer.recv_offsets[ dim1 ] = localEnds[ dim1 ] - localBegins[ dim1 ] - shift;
            }
         }
         if( ( direction & SyncDirection::Back ) != SyncDirection::None ) {
            if( countDimensionsWithOverlap( array_view ) == 3 ) {
               const int dim2 = getDimensionWithOverlap< 2 >( array_view );
               const int overlap2 = array_view.getOverlaps()[ dim2 ];
               buffer.send_offsets[ dim2 ] = -shift;
               buffer.recv_offsets[ dim2 ] = -overlap2 + shift;
            }
         }
         if( ( direction & SyncDirection::Front ) != SyncDirection::None ) {
            if( countDimensionsWithOverlap( array_view ) == 3 ) {
               const int dim2 = getDimensionWithOverlap< 2 >( array_view );
               const int overlap2 = array_view.getOverlaps()[ dim2 ];
               buffer.send_offsets[ dim2 ] = localEnds[ dim2 ] - localBegins[ dim2 ] - overlap2 + shift;
               buffer.recv_offsets[ dim2 ] = localEnds[ dim2 ] - localBegins[ dim2 ] - shift;
            }
         }
      }
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
               Backend::setDevice( this->gpu_id );

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
         this->gpu_id = Backend::getDevice();

      if( countDimensionsWithOverlap( array.getView() ) == 0 )
         throw std::invalid_argument( "the distributed array must have at least one dimension with overlap" );
      if( countDimensionsWithOverlap( array.getView() ) > 3 )
         throw std::invalid_argument( "at most 3 dimensions with overlap are supported" );

      // skip allocation on repeated calls - compare only sizes, not the actual data
      if( array_view.getCommunicator() != array.getCommunicator() || array_view.getSizes() != array.getSizes()
          || array_view.getLocalBegins() != array.getLocalBegins() || array_view.getLocalEnds() != array.getLocalEnds() )
      {
         array_view.bind( array.getView() );
         this->mask = mask;

         // allocate buffers
         allocateHelper();
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
      copyHelper( true, mask );
   }

   // stage 2: issue all send and receive async operations
   void
   stage_2()
   {
      // synchronize all CUDA streams to ensure the previous stage is finished
      if constexpr( std::is_same< typename DistributedNDArrayView::DeviceType, Devices::Cuda >::value ) {
         for( auto& [ _, buffer ] : buffers )
            Backend::streamSynchronize( buffer.stream_id );
      }

      // issue all send and receive async operations
      requests.clear();
      for( auto& [ _, buffer ] : buffers ) {
         if( ( mask & buffer.direction ) != SyncDirection::None ) {
            // negative rank and tag IDs are not valid according to the MPI standard and may be used by
            // applications to skip communication, e.g. over the periodic boundary
            if( buffer.neighbor >= 0 && buffer.tag_send >= 0 ) {
               requests.push_back( MPI::Isend( buffer.send_view.getData(),
                                               buffer.send_view.getStorageSize(),
                                               buffer.neighbor,
                                               buffer.tag_send,
                                               array_view.getCommunicator() ) );
               sent_bytes += buffer.send_view.getStorageSize() * sizeof( typename DistributedNDArray::ValueType );
               ++sent_messages;
            }
            auto& opp_buffer = buffers.at( opposite( buffer.direction ) );
            if( opp_buffer.neighbor >= 0 && opp_buffer.tag_recv >= 0 ) {
               requests.push_back( MPI::Irecv( opp_buffer.recv_view.getData(),
                                               opp_buffer.recv_view.getStorageSize(),
                                               opp_buffer.neighbor,
                                               opp_buffer.tag_recv,
                                               array_view.getCommunicator() ) );
               recv_bytes += opp_buffer.recv_view.getStorageSize() * sizeof( typename DistributedNDArray::ValueType );
               ++recv_messages;
            }
         }
      }
   }

   // stage 3: copy data from receive buffers
   void
   stage_3()
   {
      // wait for all data to arrive
      MPI::Waitall( requests.data(), requests.size() );

      // copy data from receive buffers
      copyHelper( false, mask );
   }

   // stage 4: ensure everything has finished
   void
   stage_4()
   {
      // synchronize all CUDA streams
      if constexpr( std::is_same< typename DistributedNDArrayView::DeviceType, Devices::Cuda >::value ) {
         for( auto& [ _, buffer ] : buffers )
            Backend::streamSynchronize( buffer.stream_id );
      }
   }

protected:
   static int
   countDimensionsWithOverlap( const DistributedNDArrayView& array_view )
   {
      int count = 0;
      for( std::size_t dim = 0; dim < DistributedNDArray::getDimension(); dim++ ) {
         const int overlap = array_view.getOverlaps()[ dim ];
         if( overlap > 0 )
            count++;
      }
      return count;
   }

   template< std::size_t order >
   static int
   getDimensionWithOverlap( const DistributedNDArrayView& array_view )
   {
      // we must return a valid index in [0, dimension), even if order is invalid
      if constexpr( order >= DistributedNDArray::getDimension() )
         return 0;

      // find the order-th dimension that has overlap > 0
      int i = 0;
      for( std::size_t dim = 0; dim < DistributedNDArray::getDimension(); dim++ ) {
         const int overlap = array_view.getOverlaps()[ dim ];
         if( overlap > 0 ) {
            if( i == order )
               return dim;
            i++;
         }
      }

      // we must return a valid index in [0, dimension), even if order is invalid
      return 0;
   }

   void
   allocateHelper()
   {
      for( auto& [ direction, buffer ] : buffers ) {
         const auto& localBegins = array_view.getLocalBegins();
         const auto& localEnds = array_view.getLocalEnds();

         SizesHolder bufferSize( localEnds - localBegins );

         if( ( direction & SyncDirection::Left ) != SyncDirection::None
             || ( direction & SyncDirection::Right ) != SyncDirection::None )
         {
            const int dim = getDimensionWithOverlap< 0 >( array_view );
            const int overlap = array_view.getOverlaps()[ dim ];
            bufferSize[ dim ] = overlap;
         }
         if( ( direction & SyncDirection::Bottom ) != SyncDirection::None
             || ( direction & SyncDirection::Top ) != SyncDirection::None )
         {
            if( countDimensionsWithOverlap( array_view ) >= 2 ) {
               const int dim = getDimensionWithOverlap< 1 >( array_view );
               const int overlap = array_view.getOverlaps()[ dim ];
               bufferSize[ dim ] = overlap;
            }
            else
               // skip allocation if the array does not have overlap for these directions
               continue;
         }
         if( ( direction & SyncDirection::Back ) != SyncDirection::None
             || ( direction & SyncDirection::Front ) != SyncDirection::None )
         {
            if( countDimensionsWithOverlap( array_view ) == 3 ) {
               const int dim = getDimensionWithOverlap< 2 >( array_view );
               const int overlap = array_view.getOverlaps()[ dim ];
               bufferSize[ dim ] = overlap;
            }
            else
               // skip allocation if the array does not have overlap for these directions
               continue;
         }

         // allocate buffers
         buffer.send_buffer.setSize( bufferSize );
         buffer.recv_buffer.setSize( bufferSize );

         // bind views to the buffers
         buffer.send_view.bind( buffer.send_buffer.getView() );
         buffer.recv_view.bind( buffer.recv_buffer.getView() );
      }

      // set the send and receive offsets
      setBufferOffsets();

      // set default neighbor IDs for D1Q3
      if( buffers.size() == 2 && buffers.count( SyncDirection::Left ) > 0 && buffers.count( SyncDirection::Right ) > 0 ) {
         const MPI::Comm& communicator = array_view.getCommunicator();
         const int rank = communicator.rank();
         const int nproc = communicator.size();
         auto& buffer_left = buffers.at( SyncDirection::Left );
         auto& buffer_right = buffers.at( SyncDirection::Right );
         if( buffer_left.neighbor < 0 )
            buffer_left.neighbor = ( rank + nproc - 1 ) % nproc;
         if( buffer_right.neighbor < 0 )
            buffer_right.neighbor = ( rank + 1 ) % nproc;
      }

      // set default tags from tag_offset
      for( auto& [ direction, buffer ] : buffers ) {
         if( tag_offset >= 0 && buffer.tag_recv < 0 && buffer.tag_send < 0 ) {
            buffer.tag_recv = tag_offset + static_cast< std::uint8_t >( opposite( direction ) );
            buffer.tag_send = tag_offset + static_cast< std::uint8_t >( direction );
         }
      }
   }

   template< typename LaunchConfiguration >
   static void
   setCudaStream( LaunchConfiguration& launch_config, Backend::stream_t stream )
   {}

   static void
   setCudaStream( Backend::LaunchConfiguration& launch_config, Backend::stream_t stream )
   {
      launch_config.stream = stream;
      launch_config.blockHostUntilFinished = false;
   }

   void
   copyHelper( bool to_buffer, SyncDirection mask )
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
            using BufferView = typename Buffer::NDArrayType::ViewType;
            CopyKernel< BufferView > copy_kernel;
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

/**
 * \brief Set neighbors for a synchronizer according to given synchronization pattern
 * and decomposition of a global block.
 *
 * \ingroup ndarray
 *
 * \tparam Q is the number of elements in \e pattern.
 * \param synchronizer is an instance of \ref DistributedNDArraySynchronizer.
 * \param pattern is the synchronization pattern (array of directions
 *                in which the data will be sent). It must be consistent
 *                with the partitioning of the distributed array.
 * \param rank is the ID of the current MPI rank and also an index of the
 *             corresponding block in \e decomposition.
 * \param decomposition is a vector of blocks forming a decomposition of the
 *                      global block. Its size must be equal to the size of
 *                      the MPI communicator and indices of the blocks in the
 *                      vector determine the rank IDs of the neighbors.
 * \param global is the global block (used for setting neighbors over the
 *               periodic boundary).
 */
template< typename DistributedNDArray, std::size_t Q, typename BlockType >
void
setNeighbors( DistributedNDArraySynchronizer< DistributedNDArray >& synchronizer,
              const std::array< SyncDirection, Q >& pattern,
              int rank,
              const std::vector< BlockType >& decomposition,
              const BlockType& global )
{
   const BlockType& reference = decomposition.at( rank );

   auto find = [ & ]( SyncDirection direction, typename BlockType::CoordinatesType point, SyncDirection vertexDirection )
   {
      // handle periodic boundaries
      if( ( direction & SyncDirection::Left ) != SyncDirection::None && point.x() == global.begin.x() )
         point.x() = global.end.x();
      if( ( direction & SyncDirection::Right ) != SyncDirection::None && point.x() == global.end.x() )
         point.x() = global.begin.x();
      if( ( direction & SyncDirection::Bottom ) != SyncDirection::None && point.y() == global.begin.y() )
         point.y() = global.end.y();
      if( ( direction & SyncDirection::Top ) != SyncDirection::None && point.y() == global.end.y() )
         point.y() = global.begin.y();
      if( ( direction & SyncDirection::Back ) != SyncDirection::None && point.z() == global.begin.z() )
         point.z() = global.end.z();
      if( ( direction & SyncDirection::Front ) != SyncDirection::None && point.z() == global.end.z() )
         point.z() = global.begin.z();

      for( std::size_t i = 0; i < decomposition.size(); i++ ) {
         const auto vertex = getBlockVertex( decomposition[ i ], vertexDirection );
         if( point == vertex ) {
            synchronizer.setNeighbor( direction, i );
            return;
         }
      }
      throw std::runtime_error( "coordinate [" + std::to_string( point.x() ) + "," + std::to_string( point.y() ) + ","
                                + std::to_string( point.z() ) + "] was not found in the decomposition" );
   };

   for( SyncDirection direction : pattern ) {
      switch( direction ) {
         case SyncDirection::Left:
            find( direction, getBlockVertex( reference, SyncDirection::FrontTopLeft ), SyncDirection::FrontTopRight );
            break;
         case SyncDirection::Right:
            find( direction, getBlockVertex( reference, SyncDirection::BackBottomRight ), SyncDirection::BackBottomLeft );
            break;
         case SyncDirection::Bottom:
            find( direction, getBlockVertex( reference, SyncDirection::FrontBottomRight ), SyncDirection::FrontTopRight );
            break;
         case SyncDirection::Top:
            find( direction, getBlockVertex( reference, SyncDirection::BackTopLeft ), SyncDirection::BackBottomLeft );
            break;
         case SyncDirection::Back:
            find( direction, getBlockVertex( reference, SyncDirection::BackTopRight ), SyncDirection::FrontTopRight );
            break;
         case SyncDirection::Front:
            find( direction, getBlockVertex( reference, SyncDirection::FrontTopRight ), SyncDirection::BackTopRight );
            break;
         case SyncDirection::BottomLeft:
            find( direction, getBlockVertex( reference, SyncDirection::FrontBottomLeft ), SyncDirection::FrontTopRight );
            break;
         case SyncDirection::BottomRight:
            find( direction, getBlockVertex( reference, SyncDirection::BackBottomRight ), SyncDirection::BackTopLeft );
            break;
         case SyncDirection::TopRight:
            find( direction, getBlockVertex( reference, SyncDirection::BackTopRight ), SyncDirection::BackBottomLeft );
            break;
         case SyncDirection::TopLeft:
            find( direction, getBlockVertex( reference, SyncDirection::BackTopLeft ), SyncDirection::BackBottomRight );
            break;
         case SyncDirection::BackLeft:
            find( direction, getBlockVertex( reference, SyncDirection::BackBottomLeft ), SyncDirection::FrontBottomRight );
            break;
         case SyncDirection::BackRight:
            find( direction, getBlockVertex( reference, SyncDirection::BackBottomRight ), SyncDirection::FrontBottomLeft );
            break;
         case SyncDirection::BackBottom:
            find( direction, getBlockVertex( reference, SyncDirection::BackBottomLeft ), SyncDirection::FrontTopLeft );
            break;
         case SyncDirection::BackTop:
            find( direction, getBlockVertex( reference, SyncDirection::BackTopLeft ), SyncDirection::FrontBottomLeft );
            break;
         case SyncDirection::FrontLeft:
            find( direction, getBlockVertex( reference, SyncDirection::FrontBottomLeft ), SyncDirection::BackBottomRight );
            break;
         case SyncDirection::FrontRight:
            find( direction, getBlockVertex( reference, SyncDirection::FrontBottomRight ), SyncDirection::BackBottomLeft );
            break;
         case SyncDirection::FrontBottom:
            find( direction, getBlockVertex( reference, SyncDirection::FrontBottomLeft ), SyncDirection::BackTopLeft );
            break;
         case SyncDirection::FrontTop:
            find( direction, getBlockVertex( reference, SyncDirection::FrontTopLeft ), SyncDirection::BackBottomLeft );
            break;
         case SyncDirection::BackBottomLeft:
         case SyncDirection::BackBottomRight:
         case SyncDirection::BackTopLeft:
         case SyncDirection::BackTopRight:
         case SyncDirection::FrontBottomLeft:
         case SyncDirection::FrontBottomRight:
         case SyncDirection::FrontTopLeft:
         case SyncDirection::FrontTopRight:
            find( direction, getBlockVertex( reference, direction ), opposite( direction ) );
            break;
         default:
            throw std::logic_error( "unhandled direction: " + std::to_string( static_cast< std::uint8_t >( direction ) ) );
      }
   }
}

}  // namespace TNL::Containers
