// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Graphs::Algorithms::experimental {

// clang-format off
/**
 * \page MinimumSpanningTreeOverview Overview of Minimum Spanning Tree Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all minimum spanning tree functions
 * available in the experimental namespace, helping to understand the
 * differences between variants and choose the right function for your needs.
 *
 * \section MSTWhatIs What is a Minimum Spanning Tree?
 *
 * A minimum spanning tree (MST) of a connected, undirected, weighted graph is
 * a spanning tree whose total edge weight is minimal. If the graph is not
 * connected, the result is a minimum spanning forest (MSF) — a collection of
 * MSTs, one per connected component.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Minimum_spanning_tree) for more
 * details about minimum spanning trees.
 *
 * \section MSTVariants Function Variants
 *
 * | Function                   | Algorithm  | Status          | Device     | Return |
 * |----------------------------|------------|-----------------|------------|--------|
 * | \ref minimumSpanningTree   | Kruskal    | Working         | Sequential | void   |
 * | \ref boruvkaMST            | Boruvka    | Semi-functional | All        | void   |
 * | \ref boruvkaMST_edgeList   | Boruvka    | Semi-functional | GPU only   | void   |
 * | \ref isValidMSF            | Verifier   | Debug only      | All        | bool   |
 *
 * \section MSTWarning Warnings
 *
 * The parallel Boruvka implementations (\ref boruvkaMST and
 * \ref boruvkaMST_edgeList) are semi-functional and may enter infinite loops
 * on larger graphs. Only \ref minimumSpanningTree (Kruskal) is reliable.
 *
 * \subsection MSTBoruvkaIssues Known issues with Boruvka variants
 *
 * - May break and enter an infinite runtime loop on larger graphs.
 * - The O(n^2) minPerStar loop negates parallel scaling on large graphs.
 * - Loop-breaking only handles 2-node mutual hooks; multi-edge cycles may persist.
 * - \ref boruvkaMST_edgeList uses raw CUDA atomics and does not compile on
 *   Host/Sequential devices.
 *
 * \section MSTCommonParameters Common Parameters
 *
 * - **graph** — The input undirected graph (const reference).
 * - **tree** — The output spanning forest graph.
 * - **roots** — Vector containing the roots of the trees in the forest.
 * - **sum** — Output parameter receiving the total weight of the spanning forest.
 */
// clang-format on

/**
 * \brief Computes minimum spanning tree of a graph using Kruskal's algorithm.
 *
 * The input graph must be undirected. The output graph representing the minimum spanning tree must
 * be of the same type in this sense. If the input graph is not connected, the output graph will be a forest and the
 * \e roots vector will contain the roots of the trees in the forest.
 *
 * \note This is the only working MST algorithm in this namespace. The parallel
 *       Boruvka implementations (boruvkaMST, boruvkaMST_edgeList) are semi-functional
 *       and may enter infinite loops on larger graphs.
 *
 * \tparam InGraph The type of the input graph.
 * \tparam OutGraph The type of the output graph.
 * \tparam RootsVector The type of the vector containing the roots of the trees.
 * \tparam Value The type of the values of the input graph.
 * \tparam Index The type of the indices of the input graph.
 *
 * \param graph The input graph.
 * \param tree The output graph representing the minimum spanning tree.
 * \param roots The vector containing the roots of the trees in the forest.
 */
template<
   typename InGraph,
   typename OutGraph = InGraph,
   typename RootsVector = Containers::Vector< typename InGraph::IndexType >,
   typename Value = typename InGraph::ValueType,
   typename Index = typename InGraph::IndexType >
void
minimumSpanningTree(
   const InGraph& graph,
   OutGraph& tree,
   RootsVector& roots,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

}  //namespace TNL::Graphs::Algorithms::experimental

#include "boruvkaMST.h"
#include "minimumSpanningTree.hpp"
