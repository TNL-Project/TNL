// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page TreeDetectionOverview Overview of Tree and Forest Detection Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all tree and forest detection functions,
 * helping to understand the differences between variants and choose the right
 * function for your needs.
 *
 * \section TDWhatIs What are Trees and Forests?
 *
 * A **tree** is a connected acyclic graph. A **forest** is a disjoint union
 * of trees. The functions below verify whether a given graph has these
 * properties.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more
 * details about trees and forests in graph theory.
 *
 * \section TDVariants Function Variants
 *
 * | Function        | Description                          | Parameters       | Return |
 * |-----------------|--------------------------------------|------------------|--------|
 * | \ref isTree     | Check if graph is a single tree      | graph, start     | bool   |
 * | \ref isForest   | Check if graph is a forest (roots)   | graph, roots     | bool   |
 * | \ref isForest   | Check if graph is a forest (auto)    | graph            | bool   |
 *
 * \section TDDetails Variant Details
 *
 * - **isTree** checks that the graph is connected and has exactly n-1 edges,
 *   starting the traversal from the given \e start vertex.
 * - **isForest (with roots)** checks that each connected component is a tree,
 *   using the provided root candidates as traversal starting points.
 * - **isForest (auto)** detects roots automatically by finding unvisited
 *   vertices during BFS traversal.
 *
 * \section TDCommonParameters Common Parameters
 *
 * - **graph** — The input graph (const reference).
 * - **start** — The starting vertex for tree check (default 0).
 * - **roots** — Vector of root candidates for forest check.
 */
// clang-format on

/**
 * \brief Checks if the given graph is a tree.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph.
 * \param graph The graph to check.
 * \param start The starting vertex for the tree check.
 * \return true If the graph is a tree.
 * \return false Otherwise.
 */
template< typename Graph >
bool
isTree( const Graph& graph, typename Graph::IndexType start = 0 );

/**
 * \brief Checks if the given graph is a forest.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph.
 * \tparam Vector The type of the vector containing the root candidates.
 * \param graph The graph to check.
 * \param roots The root candidates of the trees in the forest.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 */
template< typename Graph, typename Vector >
bool
isForest( const Graph& graph, const Vector& roots );

/**
 * \brief Checks if the given graph is a forest with auto-detected roots.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Tree_(graph_theory)) for more details about trees.
 *
 * \tparam Graph The type of the graph.
 * \param graph The graph to check.
 * \return true If the graph is a forest.
 * \return false Otherwise.
 */
template< typename Graph >
bool
isForest( const Graph& graph );

}  // namespace TNL::Graphs::Algorithms

#include "trees.hpp"
