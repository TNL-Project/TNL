// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs::Algorithms {

/**
 * \brief Checks if the given graph is a tree.
 *
 * See [Wikipedia page](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph.
 * \param graph The graph to check.
 * \param start_node The starting node for the tree check.
 * \return true If the graph is a tree.
 * \return false Otherwise.
 */
template< typename Graph >
bool
isTree( const Graph& graph, typename Graph::IndexType start_node = 0 );

/**
 * \brief Checks if the given graph is a forest.
 *
 * See [Wikipedia page](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph.
 * \param graph The graph to check.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 */
template< typename Graph, typename Vector >
bool
isForest( const Graph& graph, const Vector& roots );

/**
 * \brief Checks if the given graph is a forest.
 *
 * See [Wikipedia page](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph.
 * \param graph The graph to check.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 */
template< typename Graph >
bool
isForest( const Graph& graph );

}  //namespace TNL::Graphs::Algorithms

#include "trees.hpp"
