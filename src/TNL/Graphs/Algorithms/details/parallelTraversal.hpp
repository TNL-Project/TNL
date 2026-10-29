// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Graphs/SubGraph.h>
#include <TNL/Graphs/traverse.h>
#include <TNL/Matrices/MatrixBase.h>

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
 * \brief Performs one top-down expand-contract step on a sparse vertex frontier.
 *
 * Expand: each edge of the frontier vertices gets its own slot in \e edgeFrontier.  The slots of a
 * frontier vertex start at the prefix sum of the segment sizes of the preceding frontier vertices.
 * The \e visit callable is evaluated for each edge and the slot receives the target vertex if it
 * returns \c true, or \c -1 otherwise.  Contract: the valid slots are compacted back into
 * \e frontier.  The work is proportional to the number of frontier edges, not to the number of
 * vertices.  This corresponds to the advance and filter operators of Gunrock and to the
 * expand-contract scheme of Merrill et al.
 *
 * \param graph           – the graph
 * \param frontier        – input and output vertex frontier (must have capacity for all vertices)
 * \param frontierSize    – the number of vertices in the input frontier
 * \param frontierOffsets – scratch vector of size n, receives the first slot of each frontier vertex
 * \param edgeFrontier    – scratch vector for the edge frontier (resized automatically)
 * \param scanBuffer      – scratch vector for the prefix sums (resized automatically)
 * \param visit           – callable `bool( IndexType source, IndexType target )`, returns \c true
 *                          when \e target enters the next frontier
 * \param launchConfig    – the configuration for launching the segments traversal
 * \return                – the number of vertices in the new frontier
 */
template< typename Graph, typename Visit >
typename Graph::IndexType
expandContractFrontier(
   const Graph& graph,
   Containers::Vector< typename Graph::IndexType, typename Graph::DeviceType, typename Graph::IndexType >& frontier,
   typename Graph::IndexType frontierSize,
   Containers::Vector< typename Graph::IndexType, typename Graph::DeviceType, typename Graph::IndexType >& frontierOffsets,
   Containers::Vector< typename Graph::IndexType, typename Graph::DeviceType, typename Graph::IndexType >& edgeFrontier,
   Containers::Vector< typename Graph::IndexType, typename Graph::DeviceType, typename Graph::IndexType >& scanBuffer,
   Visit visit,
   const TNL::Algorithms::Segments::LaunchConfiguration& launchConfig )
{
   using DeviceType = typename Graph::DeviceType;
   using IndexType = typename Graph::IndexType;
   using ValueType = typename Graph::ValueType;

   if( frontierSize == 0 )
      return 0;

   // Expand: compute the slots of the frontier vertices in the edge frontier.
   if( scanBuffer.getSize() < frontierSize )
      scanBuffer.setSize( frontierSize );
   const auto adjacencyMatrixView = graph.getAdjacencyMatrix().getConstView();
   auto frontierView = frontier.getView();
   auto scanView = scanBuffer.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      frontierSize,
      [ = ] __cuda_callable__( IndexType k ) mutable
      {
         scanView[ k ] = adjacencyMatrixView.getRowCapacity( frontierView[ k ] );
      } );
   TNL::Algorithms::inplaceInclusiveScan( scanBuffer, 0, frontierSize );
   const IndexType edgeFrontierSize = scanBuffer.getElement( frontierSize - 1 );
   if( edgeFrontierSize == 0 )
      return 0;

   auto offsetsView = frontierOffsets.getView();
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      frontierSize,
      [ = ] __cuda_callable__( IndexType k ) mutable
      {
         offsetsView[ frontierView[ k ] ] = ( k == 0 ) ? 0 : scanView[ k - 1 ];
      } );

   if( edgeFrontier.getSize() < edgeFrontierSize )
      edgeFrontier.setSize( edgeFrontierSize );
   auto edgeFrontierView = edgeFrontier.getView( 0, edgeFrontierSize );
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
   if( scanBuffer.getSize() < edgeFrontierSize )
      scanBuffer.setSize( edgeFrontierSize );
   scanView.bind( scanBuffer.getView() );
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      edgeFrontierSize,
      [ = ] __cuda_callable__( IndexType j ) mutable
      {
         scanView[ j ] = edgeFrontierView[ j ] != -1;
      } );
   TNL::Algorithms::inplaceInclusiveScan( scanBuffer, 0, edgeFrontierSize );
   const IndexType newFrontierSize = scanBuffer.getElement( edgeFrontierSize - 1 );
   TNL::Algorithms::parallelFor< DeviceType >(
      0,
      edgeFrontierSize,
      [ = ] __cuda_callable__( IndexType j ) mutable
      {
         const IndexType v = edgeFrontierView[ j ];
         if( v != -1 )
            frontierView[ scanView[ j ] - 1 ] = v;
      } );
   return newFrontierSize;
}

}  // namespace TNL::Graphs::Algorithms::detail
