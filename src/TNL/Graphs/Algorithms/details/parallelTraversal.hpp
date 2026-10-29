// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Matrices/MatrixBase.h>

#if defined( __CUDACC__ )
   #include <cub/device/device_scan.cuh>
   #include <cub/device/device_select.cuh>
#elif defined( __HIPCC__ )
   #include <hipcub/device/device_scan.hpp>
   #include <hipcub/device/device_select.hpp>
#endif

namespace TNL::Graphs::Algorithms::detail {

/**
 * \brief Compacts newly discovered vertices into the next frontier.
 *
 * Given a 0/1 \e marks vector (1 = vertex was discovered/improved in this
 * iteration), computes the inclusive prefix sum into \e marksScan and uses it
 * to write the indices of marked vertices densely into \e frontier.
 *
 * \param marks      – 0/1 vector indicating newly discovered vertices
 * \param marksScan  – scratch vector for the inclusive scan (resized automatically)
 * \param frontier   – output vector receiving the compacted vertex indices
 * \return           – the number of vertices in the new frontier
 */
template< typename DeviceType, typename IndexType >
IndexType
compactFrontier(
   Containers::Vector< IndexType, DeviceType, IndexType >& marks,
   Containers::Vector< IndexType, DeviceType, IndexType >& marksScan,
   Containers::Vector< IndexType, DeviceType, IndexType >& frontier )
{
   TNL::Algorithms::inclusiveScan( marks, marksScan );
   const IndexType n = marks.getSize();
   const IndexType frontierSize = marksScan.getElement( n - 1 );
   if( frontierSize == 0 )
      return 0;
   frontier = 0;
   auto frontierView = frontier.getView();
   auto marksScanView = marksScan.getView();
   auto f = [ = ] __cuda_callable__( const IndexType idx, const IndexType value ) mutable
   {
      if( idx == 0 ) {
         if( marksScanView[ 0 ] == 1 )
            frontierView[ 0 ] = idx;
      }
      else if( marksScanView[ idx ] - marksScanView[ idx - 1 ] == 1 )
         frontierView[ marksScanView[ idx ] - 1 ] = idx;
   };
   marksScan.forAllElements( f );
   return frontierSize;
}

/**
 * \brief Writes the 0/1 bitmap of the vertices in a sparse \e frontier into \e marks.
 *
 * \param frontier     – vertex indices of the frontier
 * \param frontierSize – the number of valid entries in \e frontier
 * \param marks        – output 0/1 vector of size n
 */
template< typename DeviceType, typename IndexType >
void
frontierToMarks(
   const Containers::Vector< IndexType, DeviceType, IndexType >& frontier,
   IndexType frontierSize,
   Containers::Vector< IndexType, DeviceType, IndexType >& marks )
{
   marks = 0;
   const auto frontierView = frontier.getConstView();
   auto marksView = marks.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      frontierSize,
      [ = ] __cuda_callable__( IndexType k ) mutable
      {
         marksView[ frontierView[ k ] ] = 1;
      } );
}

template< typename Graph >
struct IsSubGraph : std::false_type
{};

template< typename Graph_, typename VertexFilter, typename EdgeFilter >
struct IsSubGraph< SubGraph< Graph_, VertexFilter, EdgeFilter > > : std::true_type
{};

template< typename Graph_, typename EdgeFilter >
struct IsSubGraph< MaskedSubGraph< Graph_, EdgeFilter > > : std::true_type
{};

/**
 * \brief Buffers of the top-down expand-contract step, reused across the BFS iterations.
 */
template< typename DeviceType, typename IndexType >
struct ExpandContractBuffers
{
   //! \brief The first slot of each frontier vertex in the edge frontier, indexed by vertex (size n).
   Containers::Vector< IndexType, DeviceType, IndexType > frontierOffsets;

   //! \brief The segment sizes of the frontier vertices.
   Containers::Vector< IndexType, DeviceType, IndexType > degrees;

   //! \brief The edge frontier with one slot per edge of the frontier vertices.
   Containers::Vector< IndexType, DeviceType, IndexType > edgeFrontier;

   //! \brief Prefix sums of the segment sizes and, on host, of the valid slots.
   Containers::Vector< IndexType, DeviceType, IndexType > scan;

   //! \brief Temporary storage of the CUB/hipCUB algorithms.
   Containers::Array< std::uint8_t, DeviceType, std::size_t > temporaryStorage;

   //! \brief The number of selected slots written by the CUB/hipCUB select.
   Containers::Array< IndexType, DeviceType, IndexType > selectedCount;
};

/**
 * \brief Enlarges \e buffer to hold at least \e size elements.
 *
 * The buffer grows geometrically, but not over \e maxSize, so that the BFS iterations with growing
 * frontiers do not reallocate it every time.
 */
template< typename Buffer, typename Index >
void
growBuffer( Buffer& buffer, Index size, Index maxSize )
{
   if( buffer.getSize() < size )
      buffer.setSize( std::max( size, std::min( 2 * buffer.getSize(), maxSize ) ) );
}

//! \brief Predicate of the valid slots of the edge frontier.
template< typename IndexType >
struct IsValidSlot
{
   __cuda_callable__
   bool
   operator()( const IndexType& v ) const
   {
      return v != -1;
   }
};

/**
 * \brief Computes the inclusive prefix sum of the first \e size elements of \e input into \e output.
 *
 * On GPU, it uses the single-pass decoupled look-back scan from CUB/hipCUB and reuses the temporary
 * storage in \e buffers.
 */
template< typename DeviceType, typename IndexType >
void
inclusiveSum(
   ExpandContractBuffers< DeviceType, IndexType >& buffers,
   const Containers::Vector< IndexType, DeviceType, IndexType >& input,
   Containers::Vector< IndexType, DeviceType, IndexType >& output,
   IndexType size )
{
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
   if constexpr( std::is_same_v< DeviceType, Devices::GPU > ) {
   #if defined( __HIPCC__ ) && ! defined( __CUDACC__ )
      namespace cub = hipcub;
   #endif
      std::size_t bytes = 0;
      TNL_BACKEND_SAFE_CALL( cub::DeviceScan::InclusiveSum( nullptr, bytes, input.getData(), output.getData(), size ) );
      if( buffers.temporaryStorage.getSize() < bytes )
         buffers.temporaryStorage.setSize( bytes );
      bytes = buffers.temporaryStorage.getSize();
      TNL_BACKEND_SAFE_CALL(
         cub::DeviceScan::InclusiveSum(
            static_cast< void* >( buffers.temporaryStorage.getData() ), bytes, input.getData(), output.getData(), size ) );
      return;
   }
#endif
   TNL::Algorithms::inclusiveScan( input, output, 0, size, 0 );
}

/**
 * \brief Moves the valid slots of the first \e size elements of \e edgeFrontier densely into
 * \e frontier and returns their number.
 *
 * On GPU, it uses the single-pass select from CUB/hipCUB.  Otherwise, the valid slots are flagged,
 * the flags are scanned and the valid slots are scattered.
 */
template< typename DeviceType, typename IndexType >
IndexType
selectValidSlots(
   ExpandContractBuffers< DeviceType, IndexType >& buffers,
   IndexType size,
   IndexType maxSize,
   Containers::Vector< IndexType, DeviceType, IndexType >& frontier )
{
#if defined( __CUDACC__ ) || defined( __HIPCC__ )
   if constexpr( std::is_same_v< DeviceType, Devices::GPU > ) {
   #if defined( __HIPCC__ ) && ! defined( __CUDACC__ )
      namespace cub = hipcub;
   #endif
      (void) maxSize;
      if( buffers.selectedCount.getSize() == 0 )
         buffers.selectedCount.setSize( 1 );
      std::size_t bytes = 0;
      TNL_BACKEND_SAFE_CALL(
         cub::DeviceSelect::If(
            nullptr,
            bytes,
            buffers.edgeFrontier.getData(),
            frontier.getData(),
            buffers.selectedCount.getData(),
            size,
            IsValidSlot< IndexType >{} ) );
      if( buffers.temporaryStorage.getSize() < bytes )
         buffers.temporaryStorage.setSize( bytes );
      bytes = buffers.temporaryStorage.getSize();
      TNL_BACKEND_SAFE_CALL(
         cub::DeviceSelect::If(
            static_cast< void* >( buffers.temporaryStorage.getData() ),
            bytes,
            buffers.edgeFrontier.getData(),
            frontier.getData(),
            buffers.selectedCount.getData(),
            size,
            IsValidSlot< IndexType >{} ) );
      return buffers.selectedCount.getElement( 0 );
   }
#endif
   growBuffer( buffers.scan, size, maxSize );
   const auto edgeFrontierView = buffers.edgeFrontier.getConstView();
   auto scanView = buffers.scan.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      size,
      [ = ] __cuda_callable__( IndexType j ) mutable
      {
         scanView[ j ] = edgeFrontierView[ j ] != -1;
      } );
   TNL::Algorithms::inplaceInclusiveScan( buffers.scan, 0, size );
   const IndexType selected = buffers.scan.getElement( size - 1 );
   auto frontierView = frontier.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      size,
      [ = ] __cuda_callable__( IndexType j ) mutable
      {
         const IndexType v = edgeFrontierView[ j ];
         if( v != -1 )
            frontierView[ scanView[ j ] - 1 ] = v;
      } );
   return selected;
}

/**
 * \brief Performs one top-down expand-contract step on a sparse vertex frontier.
 *
 * Expand: each edge of the frontier vertices gets its own slot in the edge frontier.  The slots of a
 * frontier vertex start at the prefix sum of the segment sizes of the preceding frontier vertices.
 * The \e visit callable is evaluated for each edge and the slot receives the target vertex if it
 * returns \c true, or \c -1 otherwise.  Contract: the valid slots are compacted back into
 * \e frontier.  The work is proportional to the number of frontier edges, not to the number of
 * vertices.  This corresponds to the advance and filter operators of Gunrock and to the
 * expand-contract scheme of Merrill et al.
 *
 * \param graph        – the graph
 * \param frontier     – input and output vertex frontier (must have capacity for all vertices)
 * \param frontierSize – the number of vertices in the input frontier
 * \param buffers      – the buffers reused across the iterations, \e frontierOffsets must have size n
 * \param visit        – callable `bool( IndexType source, IndexType target )`, returns \c true
 *                       when \e target enters the next frontier
 * \param launchConfig – the configuration for launching the segments traversal
 * \return             – the number of vertices in the new frontier
 */
template< typename Graph, typename Visit >
typename Graph::IndexType
expandContractFrontier(
   const Graph& graph,
   Containers::Vector< typename Graph::IndexType, typename Graph::DeviceType, typename Graph::IndexType >& frontier,
   typename Graph::IndexType frontierSize,
   ExpandContractBuffers< typename Graph::DeviceType, typename Graph::IndexType >& buffers,
   Visit visit,
   const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;

   if( frontierSize == 0 )
      return 0;

   // Expand: compute the slots of the frontier vertices in the edge frontier.
   const auto adjacencyMatrixView = graph.getAdjacencyMatrix().getConstView();
   const IndexType vertexCount = buffers.frontierOffsets.getSize();
   // the edge frontier cannot be larger than the number of elements of the adjacency matrix
   const IndexType maxEdgeFrontierSize = adjacencyMatrixView.getColumnIndexes().getSize();
   growBuffer( buffers.degrees, frontierSize, vertexCount );
   growBuffer( buffers.scan, frontierSize, std::max( vertexCount, maxEdgeFrontierSize ) );
   auto frontierView = frontier.getView();
   auto degreesView = buffers.degrees.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      frontierSize,
      [ = ] __cuda_callable__( IndexType k ) mutable
      {
         degreesView[ k ] = adjacencyMatrixView.getRowCapacity( frontierView[ k ] );
      } );
   inclusiveSum( buffers, buffers.degrees, buffers.scan, frontierSize );
   const IndexType edgeFrontierSize = buffers.scan.getElement( frontierSize - 1 );
   if( edgeFrontierSize == 0 )
      return 0;

   auto offsetsView = buffers.frontierOffsets.getView();
   const auto scanView = buffers.scan.getConstView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      frontierSize,
      [ = ] __cuda_callable__( IndexType k ) mutable
      {
         offsetsView[ frontierView[ k ] ] = ( k == 0 ) ? 0 : scanView[ k - 1 ];
      } );

   growBuffer( buffers.edgeFrontier, edgeFrontierSize, maxEdgeFrontierSize );
   auto edgeFrontierView = buffers.edgeFrontier.getView( 0, edgeFrontierSize );
   // The traversal of a subgraph skips the filtered edges, so their slots must be invalidated in advance.
   if constexpr( IsSubGraph< Graph >::value )
      edgeFrontierView = -1;

   forEdges(
      graph,
      frontier,
      0,
      frontierSize,
      [ = ] __cuda_callable__( IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const ValueType& weight ) mutable
      {
         const IndexType slot = offsetsView[ sourceIdx ] + localIdx;
         if( targetIdx == Matrices::paddingIndex< IndexType > )
            edgeFrontierView[ slot ] = -1;
         else
            edgeFrontierView[ slot ] = visit( sourceIdx, targetIdx ) ? targetIdx : -1;
      },
      launchConfig );

   // Contract: remove the invalid slots from the edge frontier.
   return selectValidSlots( buffers, edgeFrontierSize, maxEdgeFrontierSize, frontier );
}

}  // namespace TNL::Graphs::Algorithms::detail
