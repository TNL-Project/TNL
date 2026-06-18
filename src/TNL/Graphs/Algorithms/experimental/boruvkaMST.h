// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs::Algorithms::experimental {

/**
 * \brief Computes minimum spanning forest using Boruvka's algorithm (adjacency-matrix variant).
 *
 * Based on the Awerbuch-Shiloach [AS87] variant adapted for TNL as described in
 * R. Cichra, "Parallel graph algorithms for GPU", Bachelor's thesis, CTU Prague, 2024,
 * Chapter 5, Section 5.4 (Algorithm 8: "TNL-adapted MST"), Section 5.5 (tnlMSF1).
 * The parallel algorithm itself follows Baer, Kanakagiri, Solomonik [BKS22].
 *
 * This is a parallel Boruvka-style MSF algorithm that operates directly on the
 * adjacency matrix rows. Each iteration:
 *   1. Identifies stars (root + direct children in the parent vector)
 *   2. Finds minimum-weight outgoing edge per star vertex
 *   3. Propagates minima to star roots
 *   4. Hooks stars via minimum edges
 *   5. Breaks mutual-hook loops (only handles 2-node cycles)
 *   6. Builds tree edges and performs pointer-doubling shortcutting
 *
 * \warning This implementation has known issues (Cichra 2024, Sec. 5.5):
 *   - "We were unable to create a fully functional implementation...they can break
 *     and enter an infinite runtime loop." All implementations "seemed reliable for
 *     smaller graphs and would even pass unit tests, but for larger graph sizes,
 *     they became unreliable and prone to breaking."
 *   - The O(n^2) minPerStar loop negates parallel scaling on large graphs.
 *   - Loop-breaking only handles 2-node mutual hooks; multi-edge cycles may persist
 *     and cause the infinite loop behaviour.
 *   - The Host/Sequential tree-building path uses getElement/setElement (slow).
 *
 * \tparam InGraph The type of the input graph (must be undirected).
 * \tparam OutGraph The type of the output spanning forest graph.
 * \tparam Real The type for the edge weight sum.
 *
 * \param graph The input undirected graph.
 * \param tree The output spanning forest graph.
 * \param sum Output parameter receiving the total weight of the spanning forest.
 */
template< typename InGraph, typename OutGraph = InGraph, typename Real = typename InGraph::ValueType >
void
boruvkaMST(
   const InGraph& graph,
   OutGraph& tree,
   Real& sum,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Computes minimum spanning forest using Boruvka's algorithm (edge-list variant).
 *
 * Based on the second implementation (tnlMSF2) from R. Cichra, "Parallel graph
 * algorithms for GPU", Bachelor's thesis, CTU Prague, 2024, Chapter 5, Section 5.5.
 *
 * This variant pre-extracts all edges into flat arrays (from, to, weight) and uses
 * an edge mask to track which edges belong to the MSF. It uses atomic operations
 * (atomicMin, atomicExch, atomicCAS) for finding minimum outgoing edges per vertex
 * and per star root, and for loop-breaking with compare-and-swap.
 *
 * Tree construction happens after the main loop in a single parallel pass over
 * all edges using the edge mask.
 *
 * \warning This implementation has known issues (Cichra 2024, Sec. 5.5):
 *   - "It could therefore not be 100% correct in its results, when it finishes."
 *     Marked as "todo fix" by the original author.
 *   - "It also uses some CUDA atomic operations, so it would have to be tweaked
 *     to work on CPU." Raw atomics (atomicMin, atomicExch, atomicCAS) are not
 *     available on Host/Sequential devices.
 *   - The O(n^2) minPerStar loop negates parallel scaling on large graphs.
 *   - May enter infinite loops on larger graphs — same fundamental issue as
 *     boruvkaMST.
 *
 * \tparam InGraph The type of the input graph (must be undirected).
 * \tparam OutGraph The type of the output spanning forest graph.
 * \tparam Real The type for the edge weight sum.
 *
 * \param graph The input undirected graph.
 * \param tree The output spanning forest graph.
 * \param sum Output parameter receiving the total weight of the spanning forest.
 */
template< typename InGraph, typename OutGraph = InGraph, typename Real = typename InGraph::ValueType >
void
boruvkaMST_edgeList(
   const InGraph& graph,
   OutGraph& tree,
   Real& sum,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Verifies that the given tree is a valid minimum spanning forest of the graph.
 *
 * Corresponds to isMSF from R. Cichra, "Parallel graph algorithms for GPU",
 * Bachelor's thesis, CTU Prague, 2024, Section 6.1 (Verification functions).
 *
 * Checks three conditions:
 *   1. Every edge in the tree exists in the original graph with matching weight.
 *   2. The tree is a valid tree or forest (via isTree/isForest).
 *   3. The sum of edge weights matches the claimed sum.
 *
 * \note This verifier uses O(n^2) reduce operations over the full adjacency
 *       matrix and is intended only for testing/debugging, not production use.
 *
 * \tparam InGraph The type of the input graph.
 * \tparam OutGraph The type of the tree graph.
 * \tparam Real The type for the edge weight sum.
 *
 * \param graph The original input graph.
 * \param tree The purported spanning forest of the graph.
 * \param sum The expected total weight of the spanning forest.
 * \return true If all three verification conditions are satisfied.
 */
template< typename InGraph, typename OutGraph = InGraph, typename Real = typename InGraph::ValueType >
bool
isValidMSF(
   const InGraph& graph,
   const OutGraph& tree,
   Real sum,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

}  // namespace TNL::Graphs::Algorithms::experimental

#include "boruvkaMST.hpp"
