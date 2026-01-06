// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs::Algorithms {

// TODO: Move to the experimental folder - the parallel algorithm is not working properly yet
/**
 * \brief Computes minimum spanning tree of a graph.
 *
 * The input graph must be undirected. The output graph representing the minimum spanning tree must
 * be of the same type in this sense. If the input graph is not connected, the output graph will be a forest and the
 * \e roots vector will contain the roots of the trees in the forest.
 *
 * \tparam InGraph is the type of the input graph.
 * \tparam OutGraph is the type of the output graph.
 * \tparam RootsVector is the type of the vector containing the roots of the
 * \tparam Value is the type of the values of the input graph.
 * \tparam Index is the type of the indices of the input graph.
 *
 * \param graph is the input graph
 * \param spanning_tree is the output graph representing the minimum spanning tree.
 * \param roots is the vector containing the roots of the trees in the forest.
 */
template< typename InGraph,
          typename OutGraph = InGraph,
          typename RootsVector = Containers::Vector< typename InGraph::IndexType >,
          typename Value = typename InGraph::ValueType,
          typename Index = typename InGraph::IndexType >
void
minimumSpanningTree( const InGraph& graph, OutGraph& spanning_tree, RootsVector& roots );

}  //namespace TNL::Graphs::Algorithms

#include "minimumSpanningTree.hpp"
