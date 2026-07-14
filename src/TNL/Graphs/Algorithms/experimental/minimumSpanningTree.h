// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/Containers/Vector.h>

namespace TNL::Graphs::Algorithms::experimental {

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
