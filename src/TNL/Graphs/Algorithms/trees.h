// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page TreeDetectionOverview Overview of Tree and Forest Detection Functions
 *
 * A **tree** is a connected acyclic graph; a **forest** is a disjoint union of
 * trees. The functions below verify whether a given (sub)graph has these
 * properties.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more
 * details about trees and forests in graph theory.
 *
 * | Function              | Description                                                |
 * |-----------------------|------------------------------------------------------------|
 * | \ref isTree           | Checks if the (sub)graph is a single tree                  |
 * | \ref isForest         | Checks if the (sub)graph is a forest (auto-detected roots) |
 * | \ref isForestWithRoots| Checks if the (sub)graph is a forest with explicit roots  |
 *
 * To check a filtered subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexPredicate, edgePredicate );
 * isTree( sg, start );
 * ```
 */
// clang-format on

/**
 * \brief Checks if the given graph is a tree.
 *
 * The graph is a tree if it is connected and has exactly n-1 edges,
 * starting the traversal from the given \e start vertex.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \param launchConfig The configuration for parallel execution.
 * \return true If the graph is a tree.
 * \return false Otherwise.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_Trees.cpp is tree basic
 */
template< typename Graph >
bool
isTree(
   const Graph& graph,
   typename Graph::IndexType start = 0,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks if the given graph is a forest with auto-detected roots.
 *
 * Roots of each tree component are detected automatically by finding unvisited
 * vertices during BFS traversal.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about forests.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \param graph The graph to check.
 * \param launchConfig The configuration for parallel execution.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest basic
 */
template< typename Graph >
bool
isForest( const Graph& graph, TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks if the given graph is a forest using the provided root candidates.
 *
 * Each root candidate starts a BFS traversal for one tree component.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about forests.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param roots The root candidates of the trees in the forest.
 * \param launchConfig The configuration for parallel execution.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest with roots basic
 */
template< typename Graph, typename Vector >
bool
isForestWithRoots( const Graph& graph, const Vector& roots, TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

}  // namespace TNL::Graphs::Algorithms

#include "trees.hpp"
