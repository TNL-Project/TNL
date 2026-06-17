// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <limits>
#include <vector>

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Graphs/Edge.h>
#include <TNL/Graphs/Algorithms/trees.h>
#include <TNL/Matrices/SparseMatrix.h>

#include "boruvkaMST.h"

namespace TNL::Graphs::Algorithms::experimental {

template< typename InGraph, typename OutGraph, typename Real >
void
boruvkaMST( const InGraph& graph, OutGraph& tree, Real& sum )
{
   static_assert( InGraph::isUndirected(), "The input graph must be undirected." );
   static_assert( OutGraph::isUndirected(), "The output graph must be undirected." );

   using DeviceType = typename InGraph::DeviceType;
   using IndexType = typename InGraph::IndexType;
   using RealType = typename InGraph::ValueType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using RealVector = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using BoolVector = TNL::Containers::Vector< bool, DeviceType, IndexType >;

   sum = 0;
   RealVector sumV( 1, 0 );

   const auto& adjMatrix = graph.getAdjacencyMatrix();
   auto adjMatrixView = adjMatrix.getConstView();
   const IndexType size = graph.getVertexCount();

   // Allocate the output graph with full row capacities (over-allocated; will trim at the end)
   // Cichra 2024, Sec. 5.5: "there is no way to change certain graph attributes, namely
   // node capacities, without wiping its data. Because of this, we need to build tempGraph
   // and count the new node capacities, then set them for outGraph and copy the tempGraph
   // to it."
   OutGraph tempGraph;
   tempGraph.setVertexCount( size );
   IndexVector nodeCapacities( size );
   graph.getAdjacencyMatrix().getRowCapacities( nodeCapacities );
   tempGraph.setVertexCapacities( nodeCapacities );

   auto& tempGraphMatrix = tempGraph.getAdjacencyMatrix();
   auto tempGraphMatrixView = tempGraphMatrix.getView();

   // Tracks how many edges have been inserted into each row so far
   RealVector rowCapacitiesTracker( size, 0 );
   auto rowCapacitiesTrackerView = rowCapacitiesTracker.getView();

   // Parent vector: p[i] is the parent of vertex i in the Boruvka forest
   IndexVector p( size );
   BoolVector star( size, true );
   BoolVector starRoot( size, true );
   BoolVector T( size );

   auto starView = star.getView();
   auto starRootView = starRoot.getView();
   auto pView = p.getView();
   auto TView = T.getView();

   // Initialize: each vertex is its own parent (singleton stars)
   // Algorithm 8, line 1 (Cichra 2024, Sec. 5.4)
   auto initParents = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
   {
      pView[ vertexIdx ] = vertexIdx;
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, size, initParents );

   IndexVector pOld( size, -1 );
   auto pOldView = pOld.getView();

   // Minimum outgoing edge data per vertex
   RealVector minOutWeights( size, -1 );
   RealVector minOutWeightsFrom( size, -1 );
   IndexVector minOutWeightsTo( size, -1 );

   auto minOutWeightsView = minOutWeights.getView();
   auto minOutWeightsFromView = minOutWeightsFrom.getView();
   auto minOutWeightsToView = minOutWeightsTo.getView();

   // Star-check lambda: updates star/starRoot vectors.
   // Must reset starView = true before calling.
   // Algorithm 8, line 1 (Cichra 2024, Sec. 5.4): initially each vertex is its
   // own parent, forming singleton stars. Star = root + direct children only.
   auto starCheck = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
   {
      bool isRoot = pView[ vertexIdx ] == vertexIdx;
      bool isDirectChild = pView[ pView[ vertexIdx ] ] == pView[ vertexIdx ];

      starView[ vertexIdx ] = isRoot || isDirectChild;
      starRootView[ vertexIdx ] = starView[ vertexIdx ] && isRoot;
   };

   // MAIN ITERATION LOOP
   do {
      pOldView = pView;

      starView = true;
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, starCheck );

      // Find minimum-weight outgoing edge from each star vertex
      auto findMinimalOutEdge = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( starView[ vertexIdx ] ) {
            RealType minWeight = -1;
            IndexType minTo = -1;
            IndexType minFrom = -1;
            RealType currentMin = std::numeric_limits< RealType >::max();

            auto const row = adjMatrixView.getRow( vertexIdx );
            for( IndexType i = 0; i < row.getSize(); i++ ) {
               if( row.getValue( i ) < currentMin && pView[ vertexIdx ] != pView[ row.getColumnIndex( i ) ] ) {
                  currentMin = row.getValue( i );
                  minWeight = currentMin;
                  minTo = row.getColumnIndex( i );
                  minFrom = vertexIdx;
               }
            }
            minOutWeightsView[ vertexIdx ] = minWeight;
            minOutWeightsToView[ vertexIdx ] = minTo;
            minOutWeightsFromView[ vertexIdx ] = minFrom;
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, findMinimalOutEdge );

      // BUG: O(n^2) loop — each star root scans ALL vertices to find its minimum outgoing edge.
      // This negates parallel scaling and is the primary performance bottleneck.
      // Cichra 2024, Sec. 5.5 (tnlMSF1, minPerStar lambda): "Note the sequential loop
      // inside - looping over the edges removes the need for it, as can be seen later
      // in tnlMSF2."
      auto minPerStar = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( starRootView[ vertexIdx ] ) {
            RealType minWeight = -1;
            IndexType minTo = -1;
            IndexType minFrom = -1;
            RealType currentMin = std::numeric_limits< RealType >::max();

            for( IndexType i = 0; i < size; i++ ) {
               if( ( pView[ i ] == vertexIdx ) && ( minOutWeightsView[ i ] < currentMin ) && ( minOutWeightsView[ i ] > -1 ) ) {
                  currentMin = minOutWeightsView[ i ];
                  minWeight = currentMin;
                  minTo = minOutWeightsToView[ i ];
                  minFrom = minOutWeightsFromView[ i ];
               }
            }
            minOutWeightsView[ vertexIdx ] = minWeight;
            minOutWeightsToView[ vertexIdx ] = minTo;
            minOutWeightsFromView[ vertexIdx ] = minFrom;
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, minPerStar );

   // Hook stars using their minimum outgoing edges
   // Algorithm 8, lines 22-27 (Cichra 2024, Sec. 5.4): "By hooking, we mean changing
   // the parent of the root in p vector, so that it now shares parent with the endpoint
   // of the smallest weighted outgoing edge."
   auto starHook = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( starRootView[ vertexIdx ] && minOutWeightsToView[ vertexIdx ] > -1 ) {
            // Use pOldView for reading target's parent to avoid race conditions
            // (parent vector changed in operations above)
            pView[ vertexIdx ] = pOldView[ minOutWeightsToView[ vertexIdx ] ];
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, starHook );

      // NOTE: parent vector has changed but star/starRoot are NOT updated here.
      // The stale star/starRoot vectors serve as "memory" of what WAS a star/root,
      // which is needed for loop-breaking below. starCheck is delayed to just
      // before shortcutting where up-to-date information is required.
      // Cichra 2024, Sec. 5.5: "our star & starRoot vectors will serve as a memory
      // vectors remembering what WAS star / star root, which is needed in the
      // following part -> thus we delay starcheck to just prior of shortcutting."

      // Break mutual-hook loops: identify edges where two roots hooked each other.
      // Algorithm 8, lines 28-32 (Cichra 2024, Sec. 5.4): "A loop is found by simply
      // checking, whether a star root vertex is also its grandparent - this would mean
      // we hooked it onto a vertex whose parent is the star root vertex itself."
      // BUG: Only handles 2-node mutual hooks (i < p[i] && i == p[p[i]]).
      // Multi-edge cycles (>2 nodes hooking simultaneously) are NOT detected,
      // which can cause the algorithm to enter infinite loops on larger graphs.
      // Cichra 2024, Sec. 5.5: "we were unable to create a fully functional
      // implementation...they can break and enter an infinite runtime loop."
      auto breakLoops_identify = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( starRootView[ vertexIdx ] && ( vertexIdx < pView[ vertexIdx ] ) && ( vertexIdx == pView[ pView[ vertexIdx ] ] ) ) {
            TView[ vertexIdx ] = true;
         }
         else {
            TView[ vertexIdx ] = false;
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, breakLoops_identify );

      auto breakLoops_reset = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( TView[ vertexIdx ] ) {
            pView[ vertexIdx ] = vertexIdx;
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, breakLoops_reset );

      // Collect unique weights of edges added in this iteration
      RealVector uniqueWeights( size, -1 );
      auto uniqueWeightsView = uniqueWeights.getView();
      auto fetchUnique = [ = ] __cuda_callable__( IndexType weightIdx ) mutable
      {
         if( ! TView[ weightIdx ] && minOutWeightsView[ weightIdx ] > -1 && starRootView[ weightIdx ] ) {
            uniqueWeightsView[ weightIdx ] = minOutWeightsView[ weightIdx ];
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, fetchUnique );

      // Build tree edges — device-specific paths
      if constexpr( std::is_same_v< DeviceType, TNL::Devices::Cuda > ) {
         auto sumView = sumV.getView();
         auto updateTree = [ = ] __cuda_callable__( IndexType weightIdx ) mutable
         {
            if( uniqueWeightsView[ weightIdx ] > -1 ) {
               auto i = TNL::Algorithms::AtomicOperations< TNL::Devices::Cuda >::add(
                  rowCapacitiesTrackerView[ minOutWeightsFromView[ weightIdx ] ], 1 );
               auto j = TNL::Algorithms::AtomicOperations< TNL::Devices::Cuda >::add(
                  rowCapacitiesTrackerView[ minOutWeightsToView[ weightIdx ] ], 1 );

               auto rowFrom = tempGraphMatrixView.getRow( minOutWeightsFromView[ weightIdx ] );
               // BUG: Original code used parentheses minOutWeightsToView(weightIdx) instead of
               // brackets [weightIdx]. Fixed here, but verify the original was indeed a typo.
               auto rowTo = tempGraphMatrixView.getRow( minOutWeightsToView[ weightIdx ] );

               if( ! ( rowFrom.getValue( i ) ) ) {
                  TNL::Algorithms::AtomicOperations< TNL::Devices::Cuda >::add( sumView[ 0 ], uniqueWeightsView[ weightIdx ] );
                  rowFrom.setElement( i, minOutWeightsToView[ weightIdx ], uniqueWeightsView[ weightIdx ] );
                  rowTo.setElement( j, minOutWeightsFromView[ weightIdx ], uniqueWeightsView[ weightIdx ] );
               }
            }
         };
         TNL::Algorithms::parallelFor< DeviceType >( 0, size, updateTree );
         sum = sumView.getElement( 0 );
      }
      else {
         // Host/Sequential path: sequential getElement/setElement loop
         // NOTE: This is slow (O(n) getElement calls) but avoids atomic overhead on CPU.
         for( IndexType i = 0; i < size; i++ ) {
            if( uniqueWeightsView.getElement( i ) > -1 ) {
               auto from = minOutWeightsFromView.getElement( i );
               auto to = minOutWeightsToView.getElement( i );
               auto weight = uniqueWeightsView.getElement( i );

               if( tempGraphMatrixView.getElement( to, from ) == 0 ) {
                  rowCapacitiesTrackerView.setElement( from, rowCapacitiesTrackerView.getElement( from ) + 1 );
                  rowCapacitiesTrackerView.setElement( to, rowCapacitiesTrackerView.getElement( to ) + 1 );
                  sum += weight;
                  tempGraphMatrixView.setElement( from, to, weight );
                  tempGraphMatrixView.setElement( to, from, weight );
               }
            }
         }
      }

      // Now update star status (needed for shortcutting)
      starView = true;
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, starCheck );

      // Pointer-doubling shortcutting: compress non-star paths toward roots
      // Algorithm 8, lines 39-40 (Cichra 2024, Sec. 5.4): "we shortcut the trees,
      // so that they can become stars and be processed in future iterations."
      auto shortCut = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( ! starView[ vertexIdx ] ) {
            pView[ vertexIdx ] = pView[ pView[ vertexIdx ] ];
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, shortCut );

   } while( pView != pOldView );

   // Copy the result into the output graph
   tree.setVertexCapacities( rowCapacitiesTracker );
   tree = tempGraph;
}

template< typename InGraph, typename OutGraph, typename Real >
void
boruvkaMST_edgeList( const InGraph& graph, OutGraph& tree, Real& sum )
{
   static_assert( InGraph::isUndirected(), "The input graph must be undirected." );
   static_assert( OutGraph::isUndirected(), "The output graph must be undirected." );

   using DeviceType = typename InGraph::DeviceType;
   using IndexType = typename InGraph::IndexType;
   using RealType = typename InGraph::ValueType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using RealVector = TNL::Containers::Vector< RealType, DeviceType, IndexType >;
   using BoolVector = TNL::Containers::Vector< bool, DeviceType, IndexType >;

   sum = 0;

   const auto& adjMatrix = graph.getAdjacencyMatrix();
   auto adjMatrixView = adjMatrix.getConstView();
   const IndexType size = graph.getVertexCount();

   // Allocate the output graph
   OutGraph tempGraph;
   tempGraph.setVertexCount( size );
   IndexVector nodeCapacities( size );
   graph.getAdjacencyMatrix().getRowCapacities( nodeCapacities );
   tempGraph.setVertexCapacities( nodeCapacities );

   auto& tempGraphMatrix = tempGraph.getAdjacencyMatrix();
   auto tempGraphMatrixView = tempGraphMatrix.getView();

   RealVector rowCapacitiesTracker( size, 0 );
   auto rowCapacitiesTrackerView = rowCapacitiesTracker.getView();

   // Pre-extract all edges into flat arrays
   auto totalEdges = adjMatrix.getSegments().getStorageSize();
   IndexVector indices( size, -1 );
   indices = nodeCapacities;
   TNL::Algorithms::inplaceExclusiveScan( indices );
   auto const indicesView = indices.getConstView();

   IndexVector fromVector( totalEdges, -1 );
   auto fromVectorView = fromVector.getView();
   IndexVector toVector( totalEdges, -1 );
   auto toVectorView = toVector.getView();
   RealVector weightVector( totalEdges, -1 );
   auto weightVectorView = weightVector.getView();

   // Edge mask: marks which edges belong to the MSF (set during hooking)
   BoolVector edgeMask( totalEdges, false );
   auto edgeMaskView = edgeMask.getView();

   // Loop-breaking mask
   BoolVector loopMask( size, false );
   auto loopMaskView = loopMask.getView();

   // Minimum outgoing edge tracking per vertex
   IndexVector minOutEdge( size, -1 );
   auto minOutEdgeView = minOutEdge.getView();
   RealVector minOutEdgeWeight( size, std::numeric_limits< RealType >::max() );
   auto minOutEdgeWeightView = minOutEdgeWeight.getView();

   // Minimum outgoing edge tracking per star root
   IndexVector minOutEdgeRoots( size, -1 );
   auto minOutEdgeRootsView = minOutEdgeRoots.getView();
   RealVector minOutEdgeWeightRoots( size, std::numeric_limits< RealType >::max() );
   auto minOutEdgeWeightRootsView = minOutEdgeWeightRoots.getView();

   // Parent vector
   IndexVector p( size );
   auto pView = p.getView();
   IndexVector pOld( size );
   auto pOldView = pOld.getView();

   // Star status vectors
   BoolVector star( size, true );
   auto starView = star.getView();
   BoolVector starRoot( size, true );
   auto starRootView = starRoot.getView();

   auto starCheck = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
   {
      bool isRoot = pView[ vertexIdx ] == vertexIdx;
      bool isDirectChild = pView[ pView[ vertexIdx ] ] == pView[ vertexIdx ];

      starView[ vertexIdx ] = isRoot || isDirectChild;
      starRootView[ vertexIdx ] = starView[ vertexIdx ] && isRoot;
   };

   // Initialize parents and extract edges into flat arrays
   auto init = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
   {
      pView[ vertexIdx ] = vertexIdx;

      const auto row = adjMatrixView.getRow( vertexIdx );
      auto starting_index = indicesView[ vertexIdx ];
      for( IndexType i = 0; i < row.getSize(); i++ ) {
         fromVectorView[ starting_index + i ] = vertexIdx;
         toVectorView[ starting_index + i ] = row.getColumnIndex( i );
         weightVectorView[ starting_index + i ] = row.getValue( i );
      }
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, size, init );

   // MAIN ITERATION LOOP
   do {
      pOldView = pView;

      starView = true;
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, starCheck );

      // Find minimum outgoing edge per star vertex using edge list
      // Cichra 2024, Sec. 5.5 (tnlMSF2, minPerVertex lambda): uses atomicMin and
      // atomicExch for updating the currently considered minimal outgoing edge.
      // WARNING: Uses raw atomicMin/atomicExch which are CUDA/HIP-only intrinsics.
      // These are NOT available on Host/Sequential devices — this function will
      // fail to compile or produce wrong results on non-GPU backends.
      // Cichra 2024, Sec. 5.5: "It also uses some CUDA atomic operations, so it
      // would have to be tweaked to work on CPU."
      auto minPerVertex = [ = ] __cuda_callable__( IndexType edgeIdx ) mutable
      {
         auto u = fromVectorView[ edgeIdx ];
         auto v = toVectorView[ edgeIdx ];
         auto w = weightVectorView[ edgeIdx ];

         if( ! starView[ u ] || pView[ pView[ u ] ] == pView[ pView[ v ] ] ) {
            return;
         }
         else {
            RealType currMinWeight = atomicMin( &minOutEdgeWeightView[ u ], w );
            if( w <= currMinWeight ) {
               atomicExch( &minOutEdgeView[ u ], edgeIdx );
            }
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, totalEdges, minPerVertex );

      // BUG: O(n^2) loop — each star root scans ALL vertices to propagate its minimum edge.
      // This is the same bottleneck as in boruvkaMST.
      // Cichra 2024, Sec. 5.5 (tnlMSF2, minPerStar lambda).
      auto minPerStar = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( starRootView[ vertexIdx ] ) {
            for( IndexType u = 0; u < size; u++ ) {
               if( pView[ u ] == vertexIdx ) {
                  RealType w = minOutEdgeWeightView[ u ];
                  IndexType edgeIdx = minOutEdgeView[ u ];

                  RealType currMinWeight = atomicMin( &minOutEdgeWeightRootsView[ vertexIdx ], w );
                  if( w <= currMinWeight ) {
                     atomicExch( &minOutEdgeRootsView[ vertexIdx ], edgeIdx );
                  }
               }
            }
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, minPerStar );

      // Hook star roots via their minimum outgoing edges and mark edges in edgeMask
      auto hook = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( starRootView[ vertexIdx ] && minOutEdgeRootsView[ vertexIdx ] > -1 ) {
            auto edge = minOutEdgeRootsView[ vertexIdx ];
            auto v = toVectorView[ edge ];
            pView[ vertexIdx ] = pView[ v ];
            edgeMaskView[ edge ] = true;
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, hook );

      // Detect mutual-hook loops (same 2-node limitation as boruvkaMST)
      auto findLoops = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         loopMaskView[ vertexIdx ] =
            ( starRootView[ vertexIdx ] && ( vertexIdx < pView[ vertexIdx ] ) && ( vertexIdx == pView[ pView[ vertexIdx ] ] ) );
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, findLoops );

      // Break loops using compare-and-swap
      // Cichra 2024, Sec. 5.5 (tnlMSF2, breakLoops lambda): "We use another atomic
      // operation - compare and swap - to break the loops."
      // WARNING: Uses raw atomicCAS — CUDA/HIP-only intrinsic.
      auto breakLoops = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( loopMaskView[ vertexIdx ] ) {
            IndexType expected = pView[ vertexIdx ];
            if( atomicCAS( &pView[ vertexIdx ], expected, vertexIdx ) == expected ) {
               auto edgeIdx = minOutEdgeRootsView[ vertexIdx ];
               if( edgeIdx != -1 ) {
                  edgeMaskView[ edgeIdx ] = false;
               }
            }
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, breakLoops );

      starView = true;
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, starCheck );

      // Pointer-doubling shortcutting
      auto shortcut = [ = ] __cuda_callable__( IndexType vertexIdx ) mutable
      {
         if( ! starView[ vertexIdx ] ) {
            pView[ vertexIdx ] = pView[ pView[ vertexIdx ] ];
         }
      };
      TNL::Algorithms::parallelFor< DeviceType >( 0, size, shortcut );

      // Reset per-iteration tracking vectors
      minOutEdgeView = -1;
      minOutEdgeWeightView = std::numeric_limits< RealType >::max();
      minOutEdgeRootsView = -1;
      minOutEdgeWeightRootsView = std::numeric_limits< RealType >::max();

   } while( pView != pOldView );

   // Build the tree in a single parallel pass over all edges using the edge mask
   // Cichra 2024, Sec. 5.5 (tnlMSF2, buildTree lambda): "we simply add all edges
   // marked by the edgeMask vector to the empty tempGraph."
   RealVector sumV( 1, 0 );
   auto sumVView = sumV.getView();

   // WARNING: Uses raw atomicAdd — CUDA/HIP-only intrinsic.
   auto buildTree = [ = ] __cuda_callable__( IndexType edgeIdx ) mutable
   {
      if( edgeMaskView[ edgeIdx ] ) {
         auto u = fromVectorView[ edgeIdx ];
         auto v = toVectorView[ edgeIdx ];
         auto w = weightVectorView[ edgeIdx ];

         auto i = atomicAdd( &rowCapacitiesTrackerView[ u ], 1 );
         auto j = atomicAdd( &rowCapacitiesTrackerView[ v ], 1 );

         auto rowFrom = tempGraphMatrixView.getRow( u );
         auto rowTo = tempGraphMatrixView.getRow( v );

         rowFrom.setElement( i, v, w );
         rowTo.setElement( j, u, w );

         atomicAdd( &sumVView[ 0 ], w );
      }
   };
   TNL::Algorithms::parallelFor< DeviceType >( 0, totalEdges, buildTree );

   tree.setVertexCapacities( rowCapacitiesTracker );
   tree = tempGraph;
   sum = sumVView.getElement( 0 );
}

template< typename InGraph, typename OutGraph, typename Real >
bool
isValidMSF( const InGraph& graph, const OutGraph& tree, Real sum )
{
   using DeviceType = typename InGraph::DeviceType;
   using IndexType = typename InGraph::IndexType;
   using RealType = typename InGraph::ValueType;

   const auto& graphMatrix = graph.getAdjacencyMatrix();
   auto graphMatrixView = graphMatrix.getConstView();
   const IndexType size = graph.getVertexCount();

   const auto& treeMatrix = tree.getAdjacencyMatrix();
   auto treeMatrixView = treeMatrix.getConstView();

   // Check 1: every edge in the tree exists in the original graph with matching weight
   // NOTE: O(n^2) over the full adjacency matrix — only for testing/debugging
   auto verifyEdge = [ = ] __cuda_callable__( IndexType cellIdx ) -> bool
   {
      auto i = cellIdx % size;
      auto j = cellIdx / size;
      if( treeMatrixView.getElement( i, j ) != 0 ) {
         return ( graphMatrixView.getElement( i, j ) == treeMatrixView.getElement( i, j ) );
      }
      return true;
   };
   bool edgeCheck = TNL::Algorithms::reduce< DeviceType >( 0, size * size, verifyEdge, TNL::LogicalAnd{} );
   std::cout << "are all edges valid? --> " << ( edgeCheck ? "yes" : "no" ) << "\n";

   // Check 2: the tree is a valid tree or forest
   bool forestCheck = TNL::Graphs::Algorithms::isForest< OutGraph >( tree );
   bool treeCheck = TNL::Graphs::Algorithms::isTree< OutGraph >( tree );
   std::cout << "is it a tree? --> " << ( treeCheck ? "yes" : "no" ) << "\n";
   std::cout << "is it a forest? --> " << ( forestCheck ? "yes" : "no" ) << "\n";

   // Check 3: the sum of edge weights matches the claimed sum
   // NOTE: O(n^2) over the full adjacency matrix — only for testing/debugging
   auto sumUp = [ = ] __cuda_callable__( IndexType idx ) -> RealType
   {
      auto i = idx / size;
      auto j = idx % size;
      return ( treeMatrixView.getElement( i, j ) );
   };
   RealType sumCheck = TNL::Algorithms::reduce< DeviceType >( 0, size * size, sumUp, TNL::Plus{} ) / 2;
   std::cout << "is the edge weight sum " << sumCheck << " == " << sum << "? --> " << ( sumCheck == sum ? "yes" : "no" )
            << "\n";

   return ( edgeCheck && ( sumCheck == sum ) && ( treeCheck || forestCheck ) );
}

}  // namespace TNL::Graphs::Algorithms::experimental
