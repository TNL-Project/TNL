// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page GraphColoringOverview Overview of Graph Coloring Functions
 *
 * Graph coloring assigns integer labels (colors) to vertices so that no two
 * adjacent vertices share the same color. Colors are zero-based in the output;
 * inactive vertices are marked by \c -1.
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Graph_coloring) for more details.
 *
 * | Function              | Description                                              |
 * |-----------------------|----------------------------------------------------------|
 * | \ref graphColoring    | Greedy (speculative) coloring of the (sub)graph          |
 * | \ref graphColoringLuby| Luby MIS-based coloring of the (sub)graph               |
 * | \ref isProperlyColored| Verifies that adjacent vertices have different colors  |
 *
 * To color a filtered subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexPredicate, edgePredicate );
 * graphColoring( sg, colors );
 * ```
 */
// clang-format on

/**
 * \brief Colors an undirected graph with zero-based integer labels by greedy algorithm.
 *
 * The implementation uses speculative rounds: every uncolored vertex proposes
 * the smallest color not used by already colored neighbors, and conflicts
 * among equal-color neighbors are resolved deterministically by vertex
 * priority.
 *
 * See \ref GraphColoringOverview for an overview of all coloring functions.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param colors The output vector of zero-based color labels.
 * \param launchConfig The configuration for parallel execution.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring basic
 */
template< typename Graph, typename Vector >
void
graphColoring( const Graph& graph, Vector& colors, TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Colors an undirected graph by repeated Luby-style MIS extraction.
 *
 * Each color class is built by finding one maximal independent set on the
 * still-uncolored subgraph and assigning one color to all of its vertices.
 *
 * See \ref GraphColoringOverview for an overview of all coloring functions.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector used to store color labels.
 * \param graph The input undirected graph.
 * \param colors The output vector of zero-based color labels.
 * \param launchConfig The configuration for parallel execution.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring luby basic
 */
template< typename Graph, typename Vector >
void
graphColoringLuby( const Graph& graph, Vector& colors, TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that all color labels are non-negative and adjacent vertices differ.
 *
 * See \ref GraphColoringOverview for an overview of all coloring functions.
 *
 * To verify a coloring on a subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it as the \e graph argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector holding color labels.
 * \param graph The input undirected graph.
 * \param colors The vector of color labels to verify.
 * \param launchConfig The configuration for parallel execution.
 * \return true If the coloring is proper (no adjacent vertices share a color and all labels are non-negative).
 * \return false Otherwise.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp is properly colored basic
 */
template< typename Graph, typename Vector >
bool
isProperlyColored( const Graph& graph, const Vector& colors, TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

}  // namespace TNL::Graphs::Algorithms

#include "graphColoring.hpp"
