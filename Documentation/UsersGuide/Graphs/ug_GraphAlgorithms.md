# Graph algorithms  {#ug_GraphAlgorithms}

[TOC]

## Introduction

TNL provides a collection of parallel graph algorithms operating on the \ref TNL::Graphs::Graph class.
Each algorithm exposes a minimal public API — one function per algorithmic task — and operates on
any graph-like type, including \ref TNL::Graphs::SubGraph and \ref TNL::Graphs::MaskedSubGraph.

To restrict an algorithm to a subset of vertices, a subset of edges, or both, construct a
\ref TNL::Graphs::SubGraph via the \ref TNL::Graphs::makeSubGraph factory and pass it as the graph
argument. The examples below demonstrate this pattern for each algorithm. They are all structured as
function templates parameterised by the device type and instantiated for the host, CUDA, and HIP.

## Common concepts

The following concepts are shared by all graph algorithms in TNL.

### Device dispatch

Every algorithm is templated on the graph type, which carries the device tag
(\ref TNL::Devices::Host, \ref TNL::Devices::Cuda, or \ref TNL::Devices::Hip). No explicit device
argument is passed to the algorithm – the device is inferred from the graph. The examples therefore
define the graph type inside a device-templated function:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp graph type definition

The same function is then instantiated for each device in `main`:

```cpp
breadthFirstSearchExample< TNL::Devices::Host >();
#ifdef __CUDACC__
breadthFirstSearchExample< TNL::Devices::Cuda >();
#endif
#ifdef __HIP__
breadthFirstSearchExample< TNL::Devices::Hip >();
#endif
```

### Launch configuration

All algorithm overloads accept an optional trailing parameter of type
\ref TNL::Algorithms::Segments::LaunchConfiguration, which defaults to a default-constructed value.
This allows advanced users to control kernel launch parameters such as grid and block sizes or
stream selection. In most use cases the default value is sufficient and the parameter can be omitted,
as shown in all examples below.

### Edge predicates and edge-weight callables

Two kinds of edge-related callables are used across the algorithms:

* **Edge predicate** – a callable with signature `bool(IndexType src, IndexType tgt, ValueType w)`
  (or `bool(IndexType src, IndexType tgt)` for unweighted graphs). It returns `true` for edges that
  may be traversed and `false` for edges that should be ignored. Edge predicates are used by BFS,
  connected components, SCC, tree recognition, MIS, and graph coloring.
* **Edge-weight callable** – a callable with signature `ValueType(IndexType src, IndexType tgt, ValueType w)`.
  It transforms the stored edge weight before it is used by the algorithm. Returning infinity marks
  an edge as non-traversable. Edge-weight callables are used exclusively by SSSP.

Both kinds of callables must be decorated with `__cuda_callable__` so that they can be executed on
the host as well as on GPU devices.

### Subgraphs and vertex predicates

The set of active vertices and traversable edges can be restricted by constructing a
\ref TNL::Graphs::SubGraph via \ref TNL::Graphs::makeSubGraph and passing it to the algorithm in
place of the original graph. The factory accepts:

* a **vertex predicate** — a callable `bool(IndexType vertex)` returning `true` for active vertices;
* an **edge predicate** — a callable `bool(IndexType src, IndexType tgt, ValueType w)` returning
  `true` for edges that may be traversed;
* a **vertex-index array** — an explicit list of active vertex indices;
* or any combination of the above.

Use the \ref TNL::Graphs::edgeOnly tag to pass an edge predicate alone (without a vertex filter).
Inactive vertices receive the algorithm's inactive sentinel value in the output.

### Graph requirements (table)

Some algorithms require a specific graph orientation, enforced at compile time via `static_assert`:

| Algorithm           | Graph type requirement                                              |
| ------------------- | ------------------------------------------------------------------- |
| BFS, SSSP           | General (non-symmetric) adjacency matrix — `static_assert` enforced |
| SCC                 | Directed graph — `static_assert` enforced                           |
| MIS, Graph Coloring | Undirected graph — `static_assert` enforced                         |
| CC                  | Both symmetric and general accepted                                 |
| Trees               | Undirected graph — `static_assert` enforced                         |

### Output vector conventions (table)

The following table summarises the output type and the sentinel value used for unreachable or
inactive vertices:

| Algorithm      | Output                                            | Unreachable/Inactive sentinel           |
| -------------- | ------------------------------------------------- | --------------------------------------- |
| BFS distances  | `IndexType` values                                | `-1` = unreachable                      |
| SSSP distances | `ValueType` values                                | `-1` = unreachable                      |
| CC components  | `IndexType` values; label = smallest vertex index | `-1` = inactive                         |
| SCC components | `IndexType` values; labels start at `1`           | `-1` = inactive                         |
| MIS mask       | 0/1 values                                        | `0` = not in MIS                        |
| Coloring       | 0-based color labels                              | `-1` = inactive (masked overloads only) |
| Trees          | return `bool`, no output vector                   | N/A                                     |


## Breadth-First Search

[Breadth-first search](https://en.wikipedia.org/wiki/Breadth-first_search) (BFS) explores a graph
level by level starting from a source vertex. TNL implements a level-synchronous BFS in which the
distance of each vertex equals the number of edges on the shortest path from the source. Unreachable
vertices keep the value `-1`. See \ref TNL::Graphs::Algorithms::breadthFirstSearch and
\ref BFSOverview for details.

The graph type used in the examples below is a sparse directed graph:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp graph type definition

The basic overload computes BFS distances from a single source vertex:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs if edge predicate

In addition to the distance-computing overloads, TNL provides
\ref TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor variants that invoke a visitor callable
for every reached vertex. The callable receives the vertex index and its distance:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor

The visitor overload restricted to an induced subgraph:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor induced

The predicate-based visitor variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor if

The visitor overload combined with an edge predicate:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor edge predicate

The induced-subgraph + edge-predicate visitor variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor induced edge predicate

The predicate + edge-predicate visitor variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_BFS.cpp bfs visitor if edge predicate

The whole example reads as follows:

\includelineno Graphs/Algorithms/GraphExample_BFS.cpp

The output looks as follows:

\include GraphExample_BFS.out


## Single-Source Shortest Path

The [single-source shortest path](https://en.wikipedia.org/wiki/Shortest_path_problem) (SSSP) problem
asks for the shortest-path distances from a given source vertex to all other vertices. TNL uses
Dijkstra's algorithm in the sequential case and a Bellman-Ford-style parallel relaxation on the host
(with OpenMP) and on GPUs. Unreachable vertices keep the value `-1`. An edge-weight callable can
transform the stored weights before relaxation; returning infinity blocks an edge. See
\ref TNL::Graphs::Algorithms::singleSourceShortestPath and \ref SSSPOverview for details.

The graph type used in the examples below is a sparse directed graph:

\snippet Graphs/Algorithms/GraphExample_SSSP.cpp graph type definition

The basic overload computes shortest-path distances from a single source vertex using the stored edge
weights:

\snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp basic

The edge-weight callable variant passes a weight-transforming callable to SSSP:

\snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp edge weight callable

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp induced

The induced-subgraph + edge-weight callable variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp induced edge weight callable

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp if

The predicate + edge-weight callable variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_SSSP.cpp sssp if edge weight callable

The whole example reads as follows:

\includelineno Graphs/Algorithms/GraphExample_SSSP.cpp

The output looks as follows:

\include GraphExample_SSSP.out


## Connected Components

[Connected components](https://en.wikipedia.org/wiki/Component_(graph_theory)) label each vertex with
the identifier of the component it belongs to. TNL treats the graph as undirected, so for directed
graphs the algorithm computes *weakly* connected components. Each component is labeled with the
smallest vertex index it contains; inactive vertices receive the value `-1`. See
\ref TNL::Graphs::Algorithms::connectedComponents and \ref ConnectedComponentsOverview for details.

The graph type used in the examples below is a sparse directed graph:

\snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp graph type definition

The basic overload labels every vertex with the smallest vertex index in its component:

\snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_ConnectedComponents.cpp cc if edge predicate

The whole example reads as follows:

\includelineno Graphs/Algorithms/GraphExample_ConnectedComponents.cpp

The output looks as follows:

\include GraphExample_ConnectedComponents.out


## Strongly Connected Components

[Strongly connected components](https://en.wikipedia.org/wiki/Strongly_connected_component) (SCC)
partition a directed graph into maximal subgraphs in which every vertex is reachable from every other
vertex. TNL uses a pivot-based algorithm that performs forward and backward BFS on the directed graph.
Component labels start at `1`; inactive vertices receive the value `-1`. A directed graph is required
(`static_assert` enforced). See \ref TNL::Graphs::Algorithms::stronglyConnectedComponents and
\ref StronglyConnectedComponentsOverview for details.

The graph type used in the examples below is a sparse directed graph:

\snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp graph type definition

The basic overload labels every vertex with its strongly connected component (labels start at `1`):

\snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp scc if edge predicate

The whole example reads as follows:

\includelineno Graphs/Algorithms/GraphExample_StronglyConnectedComponents.cpp

The output looks as follows:

\include GraphExample_StronglyConnectedComponents.out


## Tree Recognition

TNL provides three functions for recognising [trees and forests](https://en.wikipedia.org/wiki/Tree_(graph_theory))
in undirected graphs:

* \ref TNL::Graphs::Algorithms::isTree – checks whether the graph is a tree (connected, exactly n−1 edges).
* \ref TNL::Graphs::Algorithms::isForest – checks whether the graph is a forest (acyclic); roots are auto-detected.
* \ref TNL::Graphs::Algorithms::isForestWithRoots – checks whether the graph is a forest using explicit root candidates.

All three return `bool` and require an undirected graph (use the `SymmetricMixed` encoding).
See \ref TNL::Graphs::Algorithms::isTree and \ref TreeDetectionOverview for details.

The graph type used in the examples below is a sparse undirected graph:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp graph type definition

### isTree

The basic overload checks whether the graph is a tree starting from a given root vertex:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is tree basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is tree edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is tree induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is tree induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is tree if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is tree if edge predicate

### isForest

The basic overload checks whether the graph is a forest; roots are auto-detected:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest if edge predicate

### isForestWithRoots

The basic overload checks whether the graph is a forest using explicit root candidates; each root
starts a BFS for one tree component:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest with roots basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest with roots edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array, combined with explicit roots:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest with roots induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest with roots induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest with roots if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_Trees.cpp is forest with roots if edge predicate

The whole example reads as follows:

\includelineno Graphs/Algorithms/GraphExample_Trees.cpp

The output looks as follows:

\include GraphExample_Trees.out


## Maximal Independent Set

A [maximal independent set](https://en.wikipedia.org/wiki/Maximal_independent_set) (MIS) is a set of
vertices no two of which are adjacent, and to which no further vertex can be added without breaking
independence. TNL implements Luby's randomized algorithm, made deterministic via a splitmix64 hash.
The output is a 0/1 mask (`1` = vertex in the MIS, `0` = not in the MIS). The verifier
\ref TNL::Graphs::Algorithms::isMaximalIndependentSet checks both independence and maximality.
An undirected graph is required (`static_assert` enforced). See
\ref TNL::Graphs::Algorithms::maximalIndependentSet and \ref MaximalIndependentSetOverview for details.

The graph type used in the examples below is a sparse undirected graph:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp graph type definition

### Producing a maximal independent set

The basic overload computes a maximal independent set; the output is a 0/1 mask:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp mis if edge predicate

### Verifying a maximal independent set

The basic verifier overload checks that a given mask is a valid maximal independent set:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis basic

The edge-predicate verifier variant runs on the same SubGraph used during computation:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis edge predicate

The induced-subgraph verifier overload checks the MIS on the induced subgraph:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis induced

The induced-subgraph verifier overload combined with an edge predicate:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis induced edge predicate

The predicate-based verifier variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis if

The predicate + edge-predicate verifier variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp is mis if edge predicate

The whole example reads as follows:

\includelineno Graphs/Algorithms/GraphExample_MaximalIndependentSet.cpp

The output looks as follows:

\include GraphExample_MaximalIndependentSet.out


## Graph Coloring

[Graph coloring](https://en.wikipedia.org/wiki/Graph_coloring) assigns colors to vertices so that no
two adjacent vertices share the same color. TNL offers two strategies:

* \ref TNL::Graphs::Algorithms::graphColoring – speculative greedy coloring.
* \ref TNL::Graphs::Algorithms::graphColoringLuby – MIS-based coloring (each color class is a maximal independent set).

Colors are 0-based. The verifier \ref TNL::Graphs::Algorithms::isProperlyColored checks that no
adjacent vertices share a color. An undirected graph is required (`static_assert` enforced). See
\ref TNL::Graphs::Algorithms::graphColoring and \ref GraphColoringOverview for details.

The graph type used in the examples below is a sparse undirected graph:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp graph type definition

### Greedy coloring

The basic overload assigns zero-based color labels using a speculative greedy strategy:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring basic

The edge-predicate variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring edge predicate

The induced-subgraph variant constructs a SubGraph from a vertex-index array:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring induced

The induced-subgraph + edge-predicate variant constructs a SubGraph with both a vertex-index array and an edge predicate:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring induced edge predicate

The predicate-based variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring if

The predicate + edge-predicate variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring if edge predicate

### Luby MIS-based coloring

The basic Luby overload computes each color class as a maximal independent set:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring luby basic

The edge-predicate Luby variant constructs a SubGraph via `makeSubGraph(graph, edgeOnly, edgePredicate)`:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring luby edge predicate

The induced-subgraph Luby overload restricts the coloring to an explicit list of active vertices:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring luby induced

The induced-subgraph Luby overload combined with an edge predicate:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring luby induced edge predicate

The predicate-based Luby variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring luby if

The predicate + edge-predicate Luby variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp coloring luby if edge predicate

### Verifying a coloring

The basic verifier overload checks that the coloring is proper (no adjacent vertices share a color):

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp is properly colored basic

The edge-predicate verifier variant runs on the same SubGraph used during computation:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp is properly colored edge predicate

The induced-subgraph verifier overload checks the coloring on the induced subgraph:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp is properly colored induced

The induced-subgraph verifier overload combined with an edge predicate:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp is properly colored induced edge predicate

The predicate-based verifier variant constructs a SubGraph with a vertex predicate:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp is properly colored if

The predicate + edge-predicate verifier variant constructs a SubGraph with both filters:

\snippet Graphs/Algorithms/GraphExample_GraphColoring.cpp is properly colored if edge predicate

The whole example reads as follows:

\includelineno Graphs/Algorithms/GraphExample_GraphColoring.cpp

The output looks as follows:

\include GraphExample_GraphColoring.out


## Further Reading

- \ref TNL::Graphs::Algorithms::breadthFirstSearch - Breadth-first search
- \ref TNL::Graphs::Algorithms::breadthFirstSearchWithVisitor - BFS with a visitor callable
- \ref BFSOverview - Overview of all breadth-first search functions
- \ref TNL::Graphs::Algorithms::singleSourceShortestPath - Single-source shortest path
- \ref SSSPOverview - Overview of all single-source shortest path functions
- \ref TNL::Graphs::Algorithms::connectedComponents - Connected components
- \ref ConnectedComponentsOverview - Overview of all connected components functions
- \ref TNL::Graphs::Algorithms::stronglyConnectedComponents - Strongly connected components
- \ref StronglyConnectedComponentsOverview - Overview of all strongly connected components functions
- \ref TNL::Graphs::Algorithms::isTree - Tree recognition
- \ref TNL::Graphs::Algorithms::isForest - Forest recognition
- \ref TNL::Graphs::Algorithms::isForestWithRoots - Forest recognition with explicit roots
- \ref TreeDetectionOverview - Overview of all tree and forest detection functions
- \ref TNL::Graphs::Algorithms::maximalIndependentSet - Maximal independent set
- \ref TNL::Graphs::Algorithms::isMaximalIndependentSet - MIS verifier
- \ref MaximalIndependentSetOverview - Overview of all maximal independent set functions
- \ref TNL::Graphs::Algorithms::graphColoring - Greedy graph coloring
- \ref TNL::Graphs::Algorithms::graphColoringLuby - Luby MIS-based graph coloring
- \ref TNL::Graphs::Algorithms::isProperlyColored - Coloring verifier
- \ref GraphColoringOverview - Overview of all graph coloring functions
