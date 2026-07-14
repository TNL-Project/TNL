// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page MaximalIndependentSetOverview Overview of Maximal Independent Set Functions
 *
 * A maximal independent set (MIS) is a set of vertices where no two are
 * adjacent (independence) and none can be added without violating independence
 * (maximality). The output is a 0/1 mask (1 = vertex in the MIS).
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Maximal_independent_set) for more details.
 *
 * | Function                    | Description                                      |
 * |-----------------------------|--------------------------------------------------|
 * | \ref maximalIndependentSet | Finds a MIS on a graph or subgraph (0/1 output)  |
 * | \ref isMaximalIndependentSet | Verifies a 0/1 mask is a valid MIS              |
 *
 * To run MIS on a filtered subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexPredicate, edgePredicate );
 * maximalIndependentSet( sg, independentSet );
 * ```
 */
// clang-format on

/**
 * \brief Finds a maximal independent set in an undirected graph.
 *
 * The implementation uses deterministic Luby-style priority rounds. The
 * output is a 0/1 mask where value 1 marks vertices that belong to the
 * maximal independent set.
 *
 * To operate on a subgraph (vertex predicate, edge predicate, or both),
 * construct a \ref SubGraph via \ref makeSubGraph and pass it as the \e graph
 * argument.  Vertices outside the active subgraph remain zero in the output.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector used to store the 0/1 mask.
 * \param graph The input undirected graph.
 * \param independentSet The output 0/1 mask (1 = vertex in the MIS).
 * \param launchConfig The configuration for parallel execution.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis basic
 */
template< typename Graph, typename Vector >
void
maximalIndependentSet(
   const Graph& graph,
   Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

/**
 * \brief Checks that the given 0/1 mask defines a maximal independent set.
 *
 * To verify a mask on a subgraph (vertex predicate, edge predicate, or both),
 * construct a \ref SubGraph via \ref makeSubGraph and pass it as the \e graph
 * argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or MaskedSubGraph).
 * \tparam Vector The type of the vector holding the 0/1 mask.
 * \param graph The input undirected graph.
 * \param independentSet The 0/1 mask to verify.
 * \param launchConfig The configuration for parallel execution.
 * \return true If the mask defines a maximal independent set.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis basic
 */
template< typename Graph, typename Vector >
bool
isMaximalIndependentSet(
   const Graph& graph,
   const Vector& independentSet,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = {} );

}  // namespace TNL::Graphs::Algorithms

#include "maximalIndependentSet.hpp"
