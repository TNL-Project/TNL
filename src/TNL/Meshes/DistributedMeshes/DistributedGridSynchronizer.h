// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/staticFor.h>
#include <TNL/Containers/Array.h>
#include <TNL/Devices/GPU.h>
#include <TNL/MPI/Wrappers.h>
#include <TNL/Meshes/DistributedMeshes/DistributedGrid.h>

namespace TNL::Meshes::DistributedMeshes {

namespace detail {

/**
 * \brief Kernel which copies the values of the grid entities in a box to a buffer or back.
 *
 * The entities in the box `[ begin, begin + sizes )` are linearized with the x-coordinate changing the fastest.
 * The orientation is a template parameter, so that it is a compile-time constant in the kernel,
 * see \ref TNL::Meshes::detail::OrientedEntitiesKernel.
 * If `Pack` is `true`, the values are obtained by `read( entity )` and stored in the buffer,
 * otherwise they are passed from the buffer to `write( entity, value )`.
 */
template<
   typename GridEntityType,
   typename Grid,
   int TotalOrientationIndex,
   bool Pack,
   typename Value,
   typename Read,
   typename Write >
struct GridBufferKernel
{
   using CoordinatesType = typename Grid::CoordinatesType;
   using IndexType = typename Grid::IndexType;

   Read read;
   Write write;
   Value* buffer;
   CoordinatesType begin;
   CoordinatesType sizes;

   __cuda_callable__
   void
   operator()( const CoordinatesType& coordinates, const Grid& grid )
   {
      GridEntityType entity( coordinates );
      entity.setTotalOrientationIndex( TotalOrientationIndex );
      entity.setGrid( grid );
      entity.refresh();
      IndexType idx = 0;
      for( int i = Grid::getMeshDimension() - 1; i >= 0; i-- )
         idx = idx * sizes[ i ] + coordinates[ i ] - begin[ i ];
      if constexpr( Pack )
         buffer[ idx ] = read( entity );
      else
         write( entity, buffer[ idx ] );
   }
};

}  // namespace detail

/**
 * \brief Synchronizes the values of the ghost entities of a distributed grid.
 *
 * Each grid entity is owned by exactly one subdomain,
 * see \ref TNL::Meshes::forLocalEntities.
 * The other entities of the local grid are the ghost entities,
 * i.e. the entities of the ghost cells and the entities on the upper sides of the subdomain
 * adjoining the neighboring subdomains.
 * The synchronization copies the values of the ghost entities from their owners.
 *
 * The values are not stored in the synchronizer.
 * They are accessed by the functions `read` and `write` given to \ref synchronize,
 * so that the values can be stored in any containers, for example:
 *
 * \code{.cpp}
 * auto view = values.getView();
 * synchronizer.template synchronize< Grid::getMeshDimension() >(
 *    [ = ] __cuda_callable__( const typename Grid::Cell& cell )
 *    {
 *       return view[ cell.getIndex() ];
 *    },
 *    [ = ] __cuda_callable__( const typename Grid::Cell& cell, double value ) mutable
 *    {
 *       view[ cell.getIndex() ] = value;
 *    } );
 * \endcode
 *
 * The values are exchanged with all neighbors including the diagonal ones.
 * On GPUs, the buffers are allocated on the device and passed to MPI directly,
 * so MPI must support the GPU memory.
 *
 * \tparam DistributedGrid is the type of the distributed grid,
 *         see \ref TNL::Meshes::DistributedMeshes::DistributedMesh.
 */
template< typename DistributedGrid >
class DistributedGridSynchronizer
{
public:
   //! \brief Type of the local grid.
   using GridType = typename DistributedGrid::GridType;

   //! \brief Device where the grid is allocated.
   using DeviceType = typename GridType::DeviceType;

   //! \brief Type of the indices of the grid entities.
   using IndexType = typename GridType::IndexType;

   //! \brief Type of the coordinates of the grid entities.
   using CoordinatesType = typename GridType::CoordinatesType;

   DistributedGridSynchronizer() = default;

   //! \brief Constructs the synchronizer of the given distributed grid, see \ref setDistributedGrid.
   explicit DistributedGridSynchronizer( const DistributedGrid& distributedGrid )
   {
      setDistributedGrid( distributedGrid );
   }

   /**
    * \brief Sets the distributed grid.
    *
    * The synchronizer keeps a pointer to the distributed grid,
    * so the grid must exist as long as the synchronizer is used.
    *
    * \param distributedGrid is the distributed grid.
    */
   void
   setDistributedGrid( const DistributedGrid& distributedGrid )
   {
      this->distributedGrid = &distributedGrid;
   }

   /**
    * \brief Synchronizes the values of the ghost entities with the given dimension.
    *
    * The values of the entities of all orientations are synchronized.
    * The function must be called by all ranks of the communicator of the distributed grid.
    *
    * The value of an entity is obtained as `read( entity )` on the owner
    * and stored as `write( entity, value )` on the ranks where the entity is a ghost entity.
    * The entity is of the type `GridType::EntityType< EntityDimension >`,
    * both functions are called on the device, so they must be `__cuda_callable__`.
    * The type of the values is the type returned by `read`.
    * The values are sent by MPI byte by byte,
    * so the type can be a fundamental type, \ref TNL::Containers::StaticVector
    * to synchronize several values at once,
    * or a structure of such types without pointers.
    * Types with a non-trivial destructor (owning resources) are rejected at compile time.
    *
    * \tparam EntityDimension is the dimension of the entities.
    * \param read is the function which returns the value of an entity.
    * \param write is the function which sets the value of an entity.
    */
   template< int EntityDimension, typename Read, typename Write >
   void
   synchronize( Read read, Write write )
   {
      constexpr int orientationsCount = GridType::EntitiesOrientations::template getOrientationsCount< EntityDimension >();
      synchronizeOrientations< EntityDimension, 0, orientationsCount >( read, write );
   }

   /**
    * \brief Synchronizes the values of the ghost entities with the given dimension and orientation.
    *
    * The values of the entities of the other orientations are not changed.
    * See the overload without the orientation for the details.
    *
    * \tparam EntityDimension is the dimension of the entities.
    * \tparam Orientation is the index of the orientation among the orientations of the entities with
    *         the given dimension, see \ref TNL::Meshes::GridEntitiesOrientations.
    * \param read is the function which returns the value of an entity.
    * \param write is the function which sets the value of an entity.
    */
   template< int EntityDimension, int Orientation, typename Read, typename Write >
   void
   synchronize( Read read, Write write )
   {
      static_assert(
         Orientation >= 0 && Orientation < GridType::EntitiesOrientations::template getOrientationsCount< EntityDimension >(),
         "Wrong orientation index." );
      synchronizeOrientations< EntityDimension, Orientation, Orientation + 1 >( read, write );
   }

   /**
    * \brief Returns the index of the neighbor in the given direction,
    * see \ref TNL::Meshes::DistributedMeshes::DistributedMesh::getNeighbors.
    */
   [[nodiscard]] static int
   getDirectionIndex( const CoordinatesType& direction )
   {
      int result = 0;
      int power = 1;
      for( int i = 0; i < GridType::getMeshDimension(); i++ ) {
         if( direction[ i ] == -1 )
            result += power;
         else if( direction[ i ] == 1 )
            result += 2 * power;
         power *= 3;
      }
      return result - 1;
   }

protected:
   using Box = std::pair< CoordinatesType, CoordinatesType >;

   static IndexType
   getBoxSize( const Box& box )
   {
      IndexType size = 1;
      for( int i = 0; i < GridType::getMeshDimension(); i++ )
         size *= TNL::max( box.second[ i ] - box.first[ i ], IndexType( 0 ) );
      return size;
   }

   /**
    * Computes the boxes of the entities with the given normals which are sent to the neighbor in the direction
    * `direction` and received from it. The entities are owned by the subdomain which owns the cell with the same
    * coordinates, the entities with nonzero normals on the upper side of the subdomain are owned by the upper
    * neighbor (or by this subdomain on the global boundary).
    */
   template< typename Normals >
   std::pair< Box, Box >
   getBoxes( const CoordinatesType& direction, const Normals& normals ) const
   {
      const GridType& grid = this->distributedGrid->getLocalMesh();
      const CoordinatesType& localBegin = grid.getLocalBegin();
      const CoordinatesType& localEnd = grid.getLocalEnd();
      const CoordinatesType& lowerOverlap = this->distributedGrid->getLowerOverlap();
      const CoordinatesType& upperOverlap = this->distributedGrid->getUpperOverlap();
      const auto& upperGlobal = grid.getUpperGlobalBoundarySides();
      Box send;
      Box receive;
      for( int i = 0; i < GridType::getMeshDimension(); i++ ) {
         const IndexType n = normals[ i ];
         if( direction[ i ] == -1 ) {
            receive.first[ i ] = localBegin[ i ] - lowerOverlap[ i ];
            receive.second[ i ] = localBegin[ i ];
            send.first[ i ] = localBegin[ i ];
            send.second[ i ] = localBegin[ i ] + lowerOverlap[ i ] + n;
         }
         else if( direction[ i ] == 1 ) {
            receive.first[ i ] = localEnd[ i ];
            receive.second[ i ] = localEnd[ i ] + upperOverlap[ i ] + n;
            send.first[ i ] = localEnd[ i ] - upperOverlap[ i ];
            send.second[ i ] = localEnd[ i ];
         }
         else {
            receive.first[ i ] = send.first[ i ] = localBegin[ i ];
            receive.second[ i ] = send.second[ i ] = localEnd[ i ] + ( upperGlobal[ i ] ? n : 0 );
         }
      }
      return { send, receive };
   }

   template< int EntityDimension, int OrientationsBegin, int OrientationsEnd, typename Read, typename Write >
   void
   synchronizeOrientations( Read read, Write write )
   {
      TNL_ASSERT_TRUE( this->distributedGrid, "The distributed grid is not set." );
      using GridEntityType = typename GridType::template EntityType< EntityDimension >;
      using Value = std::decay_t< decltype( read( std::declval< const GridEntityType& >() ) ) >;
      // StaticVector is not trivially copyable because of its user-provided copy constructor, so at least the types
      // owning resources (with a non-trivial destructor) and over-aligned types are rejected
      static_assert(
         std::is_trivially_destructible_v< Value >,
         "The values are sent by MPI byte by byte, so they must not own any resources." );
      static_assert( alignof( Value ) <= alignof( std::max_align_t ), "Over-aligned values are not supported." );
      constexpr int neighborsCount = DistributedGrid::getNeighborsCount();
      constexpr int totalOrientationsBegin = GridType::EntitiesOrientations::getTotalOrientationIndex( EntityDimension, 0 );

      if( ! this->distributedGrid->isDistributed() )
         return;
      const GridType& grid = this->distributedGrid->getLocalMesh();
      const auto& neighbors = this->distributedGrid->getNeighbors();

      // offsets of the messages in the buffers, in the number of values
      std::vector< IndexType > sendOffsets( neighborsCount + 1, 0 );
      std::vector< IndexType > receiveOffsets( neighborsCount + 1, 0 );
      for( int i = 0; i < neighborsCount; i++ ) {
         sendOffsets[ i + 1 ] = sendOffsets[ i ];
         receiveOffsets[ i + 1 ] = receiveOffsets[ i ];
         if( neighbors[ i ] < 0 )
            continue;
         const CoordinatesType direction( Directions::template getXYZ< GridType::getMeshDimension() >( i ) );
         for( int o = OrientationsBegin; o < OrientationsEnd; o++ ) {
            const auto boxes = getBoxes( direction, grid.getNormals( totalOrientationsBegin + o ) );
            sendOffsets[ i + 1 ] += getBoxSize( boxes.first );
            receiveOffsets[ i + 1 ] += getBoxSize( boxes.second );
         }
      }
      sendBuffer.setSize( sendOffsets[ neighborsCount ] * sizeof( Value ) );
      receiveBuffer.setSize( receiveOffsets[ neighborsCount ] * sizeof( Value ) );
      Value* sendData = reinterpret_cast< Value* >( sendBuffer.getData() );
      Value* receiveData = reinterpret_cast< Value* >( receiveBuffer.getData() );

      // copies the values between the boxes and the buffers, the kernels on GPUs are not synchronized one by one
      typename DeviceType::LaunchConfiguration launchConfiguration;
      if constexpr( std::is_same_v< DeviceType, Devices::GPU > )
         launchConfiguration.blockHostUntilFinished = false;
      auto copy = [ & ]( auto pack )
      {
         constexpr bool Pack = decltype( pack )::value;
         for( int i = 0; i < neighborsCount; i++ ) {
            if( neighbors[ i ] < 0 )
               continue;
            const CoordinatesType direction( Directions::template getXYZ< GridType::getMeshDimension() >( i ) );
            IndexType offset = Pack ? sendOffsets[ i ] : receiveOffsets[ i ];
            Algorithms::staticFor< int, OrientationsBegin, OrientationsEnd >(
               [ & ]( auto orientation )
               {
                  constexpr int totalOrientationIdx = totalOrientationsBegin + decltype( orientation )::value;
                  const auto boxes = getBoxes( direction, grid.getNormals( totalOrientationIdx ) );
                  const Box& box = Pack ? boxes.first : boxes.second;
                  const IndexType size = getBoxSize( box );
                  if( size == 0 )
                     return;
                  using Kernel =
                     detail::GridBufferKernel< GridEntityType, GridType, totalOrientationIdx, Pack, Value, Read, Write >;
                  const Kernel kernel{ read,
                                       write,
                                       ( Pack ? sendData : receiveData ) + offset,
                                       box.first,
                                       CoordinatesType( box.second - box.first ) };
                  Algorithms::parallelFor< DeviceType >( box.first, box.second, launchConfiguration, kernel, grid );
                  offset += size;
               } );
         }
         if constexpr( std::is_same_v< DeviceType, Devices::GPU > )
            Backend::streamSynchronize( launchConfiguration.stream );
      };

      copy( std::true_type{} );

      // exchange the messages, the message to the neighbor in the direction d is received
      // from the direction -d on the neighbor, the tag is the index of the direction of the sender
      const MPI::Comm& communicator = this->distributedGrid->getCommunicator();
      std::vector< MPI_Request > requests;
      for( int i = 0; i < neighborsCount; i++ ) {
         if( neighbors[ i ] < 0 )
            continue;
         const CoordinatesType direction( Directions::template getXYZ< GridType::getMeshDimension() >( i ) );
         const int sendCount = ( sendOffsets[ i + 1 ] - sendOffsets[ i ] ) * sizeof( Value );
         const int receiveCount = ( receiveOffsets[ i + 1 ] - receiveOffsets[ i ] ) * sizeof( Value );
         if( sendCount > 0 )
            requests.push_back(
               MPI::Isend(
                  reinterpret_cast< const std::uint8_t* >( sendData + sendOffsets[ i ] ),
                  sendCount,
                  neighbors[ i ],
                  i,
                  communicator ) );
         if( receiveCount > 0 )
            requests.push_back(
               MPI::Irecv(
                  reinterpret_cast< std::uint8_t* >( receiveData + receiveOffsets[ i ] ),
                  receiveCount,
                  neighbors[ i ],
                  getDirectionIndex( -direction ),
                  communicator ) );
      }
      MPI::Waitall( requests.data(), static_cast< int >( requests.size() ) );

      copy( std::false_type{} );
   }

   const DistributedGrid* distributedGrid = nullptr;

   Containers::Array< std::uint8_t, DeviceType, IndexType > sendBuffer;
   Containers::Array< std::uint8_t, DeviceType, IndexType > receiveBuffer;
};

}  // namespace TNL::Meshes::DistributedMeshes
