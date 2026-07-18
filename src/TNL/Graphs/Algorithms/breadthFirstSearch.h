// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>
#include <TNL/TypeTraits.h>

namespace TNL::Graphs::Algorithms {

// clang-format off
/**
 * \page BFSOverview Overview of Breadth-first Search Functions
 *
 * Breadth-first search traverses a graph layer by layer from a source vertex,
 * producing a distance vector (edge count to source, or \c -1 if unreachable).
 *
 * See [Wikipedia](https://en.wikipedia.org/wiki/Breadth-first_search) for more
 * details about the BFS algorithm.
 *
 * | Function                                          | Visitor | Predecessors | Description                            |
 * |---------------------------------------------------|---------|--------------|----------------------------------------|
 * | \ref breadthFirstSearch                           | No      | No           | Plain BFS, returns distances           |
 * | \ref breadthFirstSearchWithVisitor                | Yes     | No           | BFS with a visitor callback per vertex |
 * | \ref breadthFirstSearchWithPredecessors           | No      | Yes          | BFS with predecessor tracking          |
 * | \ref breadthFirstSearchWithVisitorAndPredecessors | Yes     | Yes          | BFS with visitor + predecessors        |
 *
 * \section BFSVisitor Visitor callable
 *
 * The visitor is a callable with signature
 * \code
 * [=] __cuda_callable__( Index vertex, Index distance )
 * \endcode
 * It is invoked exactly once per discovered vertex (excluding the start
 * vertex), in the iteration in which the vertex is first reached.
 *
 * \section BFSTraversalModes Traversal modes
 *
 * The parallel BFS implementation supports three traversal modes that are
 * selected automatically on a per-iteration basis according to the current
 * frontier size relative to the total vertex count \c n:
 *
 * | Mode              | Condition                          | Best suited for                                     |
 * |-------------------|------------------------------------|-----------------------------------------------------|
 * | Top-down compact  | default                            | Medium frontier sizes; general-purpose fallback.    |
 * | Top-down bitmap   | frontier/n < \c bitmapThreshold    | Graphs with large diameter (paths, chains), where the frontier stays small for many iterations. Skips the O(n) frontier compaction. |
 * | Bottom-up         | frontier/n > \c bottomUpThreshold  | Graphs with small diameter (small-world, scale-free), where the frontier grows large near the end of traversal. Each unvisited vertex scans its own neighbors with early exit. Undirected graphs only. |
 *
 * When both thresholds are non-zero, the algorithm may use top-down bitmap in
 * early iterations (small frontier), top-down compact in the middle, and
 * bottom-up at the peak (large frontier).  Setting a threshold to \c 0.0
 * disables the corresponding mode, yielding fully backward-compatible behavior.
 *
 * The three modes differ in how they produce the next frontier from the
 * current one.  Top-down compact iterates the outgoing edges of the compacted
 * frontier and rebuilds the frontier with a prefix scan — the O(n) compaction
 * pays off when the frontier carries enough edges.  Top-down bitmap skips the
 * compaction and instead scans all edges, testing a cheap \c marks bitmap to
 * keep only sources that belong to the current frontier; this is profitable
 * when the frontier is small relative to \c n.  Bottom-up inverts the loop:
 * each unvisited vertex scans its adjacency row for any frontier neighbor and
 * exits early on the first hit, so the cost is proportional to the number of
 * unvisited vertices rather than to the frontier size.
 *
 * The bottom-up mode is the direction-optimizing BFS of
 * Beamer et al. \cite beamer2013direction.  The top-down bitmap mode is
 * inspired by the data-centric frontier abstraction of Gunrock
 * \cite wang2016gunrock, where frontier operations are expressed as advance
 * and filter steps; skipping the filter (compaction) step when the frontier
 * is small is the key idea.  See also Merrill et al. \cite merrill2012scalable
 * for the foundational work on work-efficient, prefix-scan-based GPU BFS that
 * the top-down compact mode builds on.
 *
 * \section BFSSubgraph Filtered subgraphs
 *
 * To run BFS on a filtered subgraph, construct a \ref SubGraph via
 * \ref makeSubGraph and pass it:
 * ```cpp
 * auto sg = makeSubGraph( graph, vertexPredicate, edgePredicate );
 * breadthFirstSearch( sg, start, distances );
 * ```
 */
// clang-format on

/**
 * \brief Performs breadth-first search (BFS) on the given graph starting from the specified vertex.
 *
 * See \ref BFSOverview for an overview of all BFS variants, traversal modes,
 * and visitor semantics.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument. Vertices outside the active subgraph
 * keep distance \c -1 in the output.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \param graph The graph on which BFS is performed.
 * \param start The starting vertex for BFS.
 * \param distances The vector where distances from the start vertex will be stored.
 * \param bitmapThreshold When the frontier size drops below this fraction of the
 *   total vertex count, BFS switches to top-down bitmap mode.  See
 *   \ref BFSOverview "Traversal modes".  \c 0.0 (default) disables it.
 * \param bottomUpThreshold When the frontier size exceeds this fraction of the
 *   total vertex count, BFS switches to bottom-up mode (undirected graphs only).
 *   See \ref BFSOverview "Traversal modes".  \c 0.0 (default) disables it.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs basic
 */
template< typename Graph, typename Vector >
void
breadthFirstSearch(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   double bitmapThreshold = 0.0,
   double bottomUpThreshold = 0.0,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) with a visitor callback.
 *
 * See \ref BFSOverview for the visitor signature and traversal mode details.
 *
 * To operate on a subgraph, construct a \ref SubGraph via \ref makeSubGraph
 * and pass it as the \e graph argument.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting vertex for BFS.
 * \param visitor The callable invoked upon visiting each vertex.
 * \param distances The vector where distances from the start vertex will be stored.
 * \param bitmapThreshold See \ref breadthFirstSearch.
 * \param bottomUpThreshold See \ref breadthFirstSearch.
 * \param launchConfig The configuration for launching the segments traversal.
 *
 * \par Example
 * \snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor
 */
template<
   typename Graph,
   typename Vector,
   typename Visitor,
   typename Enable = std::enable_if_t< ! IsArrayType< Visitor >::value > >
void
breadthFirstSearchWithVisitor(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   double bitmapThreshold = 0.0,
   double bottomUpThreshold = 0.0,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) with predecessor tracking.
 *
 * In addition to distances, this function computes a predecessor vector where
 * \c predecessors[v] is the parent of vertex \c v in the BFS tree.  The start
 * vertex and any unreachable vertices have predecessor \c -1.
 *
 * See \ref BFSOverview for traversal mode details.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam PredecessorVector The type of the vector used to store predecessors.
 * \param graph The graph on which BFS is performed.
 * \param start The starting vertex for BFS.
 * \param distances The vector where distances from the start vertex will be stored.
 * \param predecessors The vector where predecessor indices will be stored.
 * \param deterministic If \c true, the predecessor of each vertex is chosen as
 *   the smallest source index among all valid parents in the same BFS layer.
 *   If \c false (default), the predecessor is whichever thread wins the
 *   atomic update — faster but non-reproducible between runs.
 * \param bitmapThreshold See \ref breadthFirstSearch.
 * \param bottomUpThreshold See \ref breadthFirstSearch.
 * \param launchConfig The configuration for launching the segments traversal.
 */
template< typename Graph, typename Vector, typename PredecessorVector >
void
breadthFirstSearchWithPredecessors(
   const Graph& graph,
   typename Graph::IndexType start,
   Vector& distances,
   PredecessorVector& predecessors,
   bool deterministic = false,
   double bitmapThreshold = 0.0,
   double bottomUpThreshold = 0.0,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs breadth-first search (BFS) with a visitor callback and
 *   predecessor tracking.
 *
 * Combines the functionality of \ref breadthFirstSearchWithVisitor and
 * \ref breadthFirstSearchWithPredecessors.  The visitor is invoked exactly
 * once per discovered vertex (excluding the start vertex).
 *
 * See \ref BFSOverview for traversal mode details.
 *
 * \tparam Graph The type of the graph (Graph, SubGraph, or GraphView).
 * \tparam Vector The type of the vector used to store distances.
 * \tparam PredecessorVector The type of the vector used to store predecessors.
 * \tparam Visitor The type of the visitor callable.
 * \param graph The graph on which BFS is performed.
 * \param start The starting vertex for BFS.
 * \param visitor The callable invoked upon visiting each vertex.
 * \param distances The vector where distances from the start vertex will be stored.
 * \param predecessors The vector where predecessor indices will be stored.
 * \param deterministic If \c true, predecessors are chosen deterministically
 *   (smallest source index per layer).  See
 *   \ref breadthFirstSearchWithPredecessors.
 * \param bitmapThreshold See \ref breadthFirstSearch.
 * \param bottomUpThreshold See \ref breadthFirstSearch.
 * \param launchConfig The configuration for launching the segments traversal.
 */
template<
   typename Graph,
   typename Vector,
   typename PredecessorVector,
   typename Visitor,
   typename Enable = std::enable_if_t< ! IsArrayType< Visitor >::value > >
void
breadthFirstSearchWithVisitorAndPredecessors(
   const Graph& graph,
   typename Graph::IndexType start,
   Visitor&& visitor,
   Vector& distances,
   PredecessorVector& predecessors,
   bool deterministic = false,
   double bitmapThreshold = 0.0,
   double bottomUpThreshold = 0.0,
   TNL::Algorithms::Segments::LaunchConfiguration launchConfig = TNL::Algorithms::Segments::LaunchConfiguration() );
}  // namespace TNL::Graphs::Algorithms

#include "breadthFirstSearch.hpp"
