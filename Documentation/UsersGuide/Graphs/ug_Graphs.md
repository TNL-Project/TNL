# Graphs  {#ug_Graphs}

[TOC]

## Introduction

Graphs are fundamental data structures used to represent relationships between objects. In TNL, the \ref TNL::Graphs::Graph class provides a flexible and efficient implementation of graphs that can run on both CPUs and GPUs.

A graph in TNL consists of **vertices** (also called nodes) and **edges** connecting pairs of vertices. Edges can be:
* **Directed**: An edge from vertex A to vertex B is different from an edge from B to A
* **Undirected**: An edge between A and B can be traversed in both directions
* **Weighted**: Each edge has an associated weight (numerical value)

Internally, TNL represents graphs using an **adjacency matrix**, which stores information about which vertices are connected by edges and what weights those edges have.

## Graph Orientation: Directed vs. Undirected Graphs

TNL supports both directed and undirected graphs through the type tags \ref TNL::Graphs::DirectedGraph and \ref TNL::Graphs::UndirectedGraph :

* **Directed graphs**: Each edge has a direction. An edge from vertex `u` to vertex `v` does not imply an edge from `v` to `u`.
\snippet Graphs/GraphExample_Constructors.cpp graph type definition
* **Undirected graphs**: Edges have no direction. If there's an edge between vertices `u` and `v`, it can be traversed in both directions.
\snippet Graphs/GraphExample_Constructors.cpp undirected graph type definition

## Sparse vs. Dense Adjacency Matrices

\anchor ug_graphs_sparse_vs_dense

TNL graphs can use either **sparse** or **dense** adjacency matrices to store edge information. The choice significantly impacts memory usage and performance.

### Sparse Adjacency Matrices And Sparse Graphs (Default)

By default, graphs use sparse matrix representations (typically CSR format). A sparse graph only stores information about edges that actually exist:

\snippet Graphs/GraphExample_Constructors.cpp graph type definition

The memory usage is proportional to the number of edges in the graph. By default, the sparse adjacency matrix is stored in the CSR format (segments) but any other format (segments) defined in \ref TNL::Algorithms::Segments can be used instead. For unweighted graphs, using `ValueType = bool` avoids storing numerical edge weights and reduces memory overhead, as only the presence or absence of edges is represented.

### Dense Adjacency Matrices And Dense Graphs

Dense graphs store information about all possible edges between all vertex pairs:

\snippet Graphs/GraphExample_Constructors.cpp dense graph type definition

Memory usage is proportional to the square of the number of vertices (O(V²)).
This representation is best suited for complete or nearly complete graphs, where most vertex pairs are connected.
Missing edges must be represented explicitly by the user (e.g., via a chosen sentinel value or encoding), since the matrix representation assumes that all vertex pairs are present, including diagonal entries representing self-loops.
It is therefore the user’s responsibility to omit such edges when they are not desired.

### Comparison Table

| Aspect          | Sparse Graph        | Dense Graph        |
| --------------- | ------------------- | ------------------ |
| **Memory**      | O(V + E)            | O(V²)              |
| **Edge lookup** | O(degree)           | O(1)               |
| **Best for**    | Sparse connectivity | Dense connectivity |

In the following examples, we assume the following graph types:

* Sparse directed graph:
\snippet Graphs/GraphExample_Constructors.cpp graph type definition
* Sparse undirected graph:
\snippet Graphs/GraphExample_Constructors.cpp undirected graph type definition
* Dense directed graph:
\snippet Graphs/GraphExample_Constructors.cpp dense graph type definition


## Constructing Graphs

TNL provides several ways to construct graphs.

### Default Constructor

To create an empty graph, the default constructor can be used:

\snippet Graphs/GraphExample_Constructors.cpp default constructor

### Constructor with Vertex Count

Constructor with vertex count creates a graph with specified number of vertices but no edges:

\snippet Graphs/GraphExample_Constructors.cpp constructor with vertex count

### Constructor with Initializer List for Sparse and Dense Graphs

Edges in both **sparse** and **dense** graphs can be specified as `(sourceIdx, targetIdx, weight)` tuples in an initializer list, together with the number of vertices:

\snippet Graphs/GraphExample_Constructors.cpp constructor with edges

In a sparse graph, unspecified edges are considered missing.
In a dense graph, unspecified edges are represented by zero weights.

### Constructor with Initializer List for Sparse Undirected Graphs

Edges of sparse undirected graphs can be specified in the same way, but only half of them are required:

\snippet Graphs/GraphExample_Constructors.cpp constructor with edges for undirected graph

The last parameter specifies the encoding of the edges (see also \ref TNL::Matrices::MatrixElementsEncoding):

- `Complete` – all edges must be specified explicitly, and symmetry is required.
- `SymmetricLower` – only the lower triangular part of the adjacency matrix is specified, i.e. edges `(u, v)` with `u > v`.
- `SymmetricUpper` – only the upper triangular part of the adjacency matrix is specified, i.e. edges `(u, v)` with `u < v`.
- `SymmetricMixed` – elements from both the lower and upper triangular parts of the adjacency matrix may be specified; the corresponding symmetric elements are filled automatically. This is the default setting for the undirected graphs.

By default, undirected graphs are represented by a symmetric sparse matrix, i.e. a matrix in which only the lower triangular part is stored. This behavior can be changed by explicitly selecting a general adjacency matrix type (see also \ref TNL::Matrices::SparseMatrix).

### Constructor with Initializer List for Dense Graphs

Edges of **dense graphs** can be specified by providing all edge weights in a two-dimensional structure:

\snippet Graphs/GraphExample_Constructors.cpp constructor with edges for dense graph

### Constructor with std::map

This constructor builds graphs from a map of edge pairs to weights:

\snippet Graphs/GraphExample_Constructors.cpp constructor with edge map

If the graph is **dense**, unspecified edges are assigned zero weight.
If the graph is **undirected**, the `SymmetricMixed` encoding of adjacency matrix elements is required.
The encoding can be changed using the third constructor parameter, in the same way as for the constructor that takes an initializer list.

### Copy constructor

The copy constructor creates a copy of another graph.

\snippet Graphs/GraphExample_Constructors.cpp copy constructor


### Constructor from Adjacency Matrix

This constructor creates a graph from an existing adjacency matrix:

\snippet Graphs/GraphExample_Constructors.cpp constructor from adjacency matrix

The whole example reads as follows:

\includelineno Graphs/GraphExample_Constructors.cpp

And the output looks as follows:

\include GraphExample_Constructors.out


## Setting Edges

After creating a graph, you can set or modify its edges using the `setEdges` and `setDenseEdges` methods. These methods behave similarly to the corresponding constructors.

### Setting Edges with Initializer List (Sparse)

For **sparse** and **dense** graphs:

\snippet Graphs/GraphExample_setEdges.cpp setEdges with initializer list

The method \ref TNL::Graphs::Graph::setEdges also accepts a matrix elements encoding (see \ref TNL::Matrices::MatrixElementsEncoding) as the second parameter, following the initializer list. This is particularly useful when setting undirected graphs.

### Setting Edges with Initializer List (Dense)

For **dense** graphs, the method \ref TNL::Graphs::Graph::setDenseEdges sets the graph edges based on the complete adjacency matrix:

\snippet Graphs/GraphExample_setEdges.cpp setDenseEdges with initializer list

### Setting Edges with std::map

The graph edges can also be set using a `std::map`:

\snippet Graphs/GraphExample_setEdges.cpp setEdges with std map

## Graph Traversal

TNL provides powerful parallel traversal capabilities through the functions
\ref TNL::Graphs::forVertices, \ref TNL::Graphs::forEdges, and related utilities.
A graph can be traversed either by **vertices** or by **edges**. Vertex-based traversal is performed using a vertex view.

The following provides an overview of graph traversal functions:

1. TNL distinguishes between **vertex-based** and **edge-based traversal**. Vertex-based traversal assigns at most one thread to each vertex, while edge-based traversal enables finer-grained parallelism by distributing work across edges.
| Category                                          | Operates On         | Lambda Parameter       | Use Case                         |
| ------------------------------------------------- | ------------------- | ---------------------- | -------------------------------- |
| **Edge-wise** (`forEdges`, `forAllEdges`)         | Individual edges    | Edge indices & weights | Operate on each  edge separately |
| **Vertex-wise** (`forVertices`, `forAllVertices`) | Individual vertices | VertexView object      | Operate on vertices              |
2. The scope of vertex traversal can be defined in several ways: **all vertices**, **a vertex range**, **an explicit list of vertex indices**, or **a condition on the vertex index**.
| Scope     | Vertices Processed               | Parameters                                        |
| --------- | -------------------------------- | ------------------------------------------------- |
| **All**   | All vertices                     | No range/array parameters                         |
| **Range** | Vertices in `[begin, end)`       | `begin` and `end` indices                         |
| **Array** | Specific vertices                | Array of vertex indices                           |
| **If**    | Vertices filtered by a condition | Process vertices based on vertex-level properties |
3. All functions can be called for both **constant** and **non-constant** graphs.
| Category      | Graph Modifiable? | Use Case                                     |
| ------------- | ----------------- | -------------------------------------------- |
| **Non-const** | Yes               | Can modify graph edges and structure         |
| **Const**     | No                | Read-only access to graph vertices and edges |

See also \ref GraphTraversalOverview for more details.

### Traversing By Vertices

A \ref TNL::Graphs::GraphVertexView provides access to a single vertex and its outgoing edges. Vertex views are the primary way to interact with graph structure in parallel GPU code.

#### Traversing All Vertices

Functions \ref TNL::Graphs::forAllVertices iterates over all vertices in parallel:

\snippet Graphs/Traverse/GraphExample_forAllVertices.cpp traverse all vertices

If the function is called with the constant graphs the `typename GraphType::VertexType` needs to be replaced with `typename GraphType::ConstVertexView`.

The whole example reads as:

\includelineno Graphs/Traverse/GraphExample_forAllVertices.cpp

and the output reads as:

\include GraphExample_forAllVertices.out

#### Traversing a Range of Vertices

Traversing vertices in a specific range `[begin, end)` can be done using the function \ref TNL::Graphs::forVertices :

\snippet Graphs/Traverse/GraphExample_forVertices.cpp traverse vertices in range

If the function is called with the constant graphs the `typename GraphType::VertexType` needs to be replaced with `typename GraphType::ConstVertexView`.

The whole example reads as:

\includelineno Graphs/Traverse/GraphExample_forVertices.cpp

and the output reads as:

\include GraphExample_forVertices.out

#### Traversing Vertices with a Given Indecis

The function \ref TNL::Graphs::forVertices also allows traversing vertices with explicitly specified indices.
The indices are provided as an array:

\snippet Graphs/Traverse/GraphExample_forVerticesWithIndexes.cpp create vertex index array

The vertices with these indecis can be traversed as follows:

\snippet Graphs/Traverse/GraphExample_forVerticesWithIndexes.cpp traverse only specified vertices

If the function is called with the constant graphs the `typename GraphType::VertexType` needs to be replaced with `typename GraphType::ConstVertexView`.

The whole example reads as:

\includelineno Graphs/Traverse/GraphExample_forVerticesWithIndexes.cpp

and the output reads as:

\include GraphExample_forVerticesWithIndexes.out

#### Traversing Vertices with a Condition

Use `forAllVerticesIf` to process only vertices that meet criteria specified by the lambda function `condition`:

\snippet Graphs/Traverse/GraphExample_forVerticesIf.cpp condition lambda

The lambda function takes a vertex index and returns `true` for vertices that should be traversed; all other vertices are skipped.
The function `forAllVerticesIf` is invoked as follows:

\snippet Graphs/Traverse/GraphExample_forVerticesIf.cpp traverse vertices with condition

There is also an alternative function, \ref TNL::Graphs::forVerticesIf, which accepts a range of vertex indices `[begin, end)`.
Only vertices within this range are considered for traversal.

If the function is called with the constant graphs the `typename GraphType::VertexType` needs to be replaced with `typename GraphType::ConstVertexView`.

The complete example reads as:

\includelineno Graphs/Traverse/GraphExample_forVerticesIf.cpp

The output looks as follows:

\include GraphExample_forVerticesIf.out

### Traversing By Edges

Accessing edges from a vertex is possible; however, in this mode each vertex is processed by at most one thread. For higher degrees of parallelism, specialized traversal functions must be used.

#### Traversing All Edges

The \ref TNL::Graphs::forAllEdges function iterates over all edges in the graph:

\snippet Graphs/Traverse/GraphExample_forAllEdges.cpp traverse all edges

The `localIdx` index specifies the rank of an edge within the set of all edges incident to the vertex `sourceIdx`.
Note that for sparse, non-constant graphs, both `targetIdx` and `weight` can be modified. For dense graphs, only `weight` can be modified, and the `localIdx` index is equal to `targetIdx`.

The whole example reads as follows:

\includelineno Graphs/Traverse/GraphExample_forAllEdges.cpp

The output looks as follows:

\include GraphExample_forAllEdges.out

#### Traversing Edges in a Range

Traversing only edges connected to vertices in a specific range `[begin, end)` can be done as follows:

\snippet Graphs/Traverse/GraphExample_forEdges.cpp traverse edges in range

The whole example reads as follows:

\includelineno Graphs/Traverse/GraphExample_forEdges.cpp

The output looks as follows:

\include GraphExample_forEdges.out

#### Traversing Edges for Specific Vertices

The function \ref TNL::Graphs::forEdges can traverse only edges of **vertices** with **explicitly specified indices**, similarly to \ref TNL::Graphs::forVertices.

The indecis of vertices for traversing are specified as follows:

\snippet Graphs/Traverse/GraphExample_forEdgesWithIndexes.cpp vertex indexes for traversing

The traversal itself is performed as follows:

\snippet Graphs/Traverse/GraphExample_forEdgesWithIndexes.cpp traverse edges from specified vertices

The whole example reads as follows:

\includelineno Graphs/Traverse/GraphExample_forEdgesWithIndexes.cpp

The output looks as follows:

\include GraphExample_forEdgesWithIndexes.out

#### Conditional Edge Traversal

For conditional traversal of edges, the function \ref TNL::Graphs::forAllEdgesIf can be used.
The condition is expressed by a lambda function `condition`:

\snippet Graphs/Traverse/GraphExample_forEdgesIf.cpp condition lambda

The lambda function takes a **vertex index** and returns `true` if **the edges incident to that vertex**
should be traversed; otherwise, it returns `false`. The traversal is executed as follows:

\snippet Graphs/Traverse/GraphExample_forEdgesIf.cpp traverse edges from vertices satisfying condition

The function \ref TNL::Graphs::forEdgesIf behaves in the same way and additionally accepts a range of
vertex indices to be traversed.

The whole example reads as follows:

\includelineno Graphs/Traverse/GraphExample_forEdgesIf.cpp

The output looks as follows:

\include GraphExample_forEdgesIf.out

#### Const vs. Non-Const Edge Traversal

For constant graphs, edge data is read-only:

```cpp
TNL::Graphs::forAllEdges(
    constGraph,
    [] __cuda_callable__ ( int source, int local, int target, const float& weight ) {
        // target and weight are read-only
        // No 'mutable' needed
    }
);
```

## Reductions on Graphs

Graph reductions allow efficient computation of aggregate values over edges or vertices.
Such computations can also be implemented using vertex traversal functions
(\ref TNL::Graphs::forAllVertices, \ref TNL::Graphs::forVertices,
\ref TNL::Graphs::forAllVerticesIf, \ref TNL::Graphs::forVerticesIf).
However, to achieve a higher level of parallelism, TNL provides specialized reduction
operations for graph data structures.

The following provides an overview of graph reduction functions:

1. Reductions can be performed either in a **basic** mode or **with arguments**, which additionally return the position of the edge being searched for.
| Category         | Tracks Position? | Use Case                                                                   |
| ---------------- | ---------------- | -------------------------------------------------------------------------- |
| **Basic**        | No               | Only the reduced weight is needed (e.g., vertex sum, vertex max)           |
| **WithArgument** | Yes              | Need weight and target vertex index (e.g., max weight and where it occurs) |
2. The scope of reduction can be defined in several ways: **all vertices**,  **a vertex range**, **an explicit list of vertex indices**, or **a condition on the vertex index**.
| Scope     | Vertices Processed               | Parameters                                        |
| --------- | -------------------------------- | ------------------------------------------------- |
| **All**   | All vertices                     | No range/array parameters                         |
| **Range** | Vertices `[begin, end)`          | `begin` and `end` indices                         |
| **Array** | Specific vertices                | Array of vertex indices                           |
| **If**    | Vertices filtered by a condition | Process vertices based on vertex-level properties |
3. All functions can be called for both **constant** and **non-constant** graphs.
| Category      | Graph Modifiable? | Use Case                                |
| ------------- | ----------------- | --------------------------------------- |
| **Non-const** | Yes               | Can modify graph edges during reduction |
| **Const**     | No                | Read-only access to graph edges         |

See also \ref GraphReductionOverview for more details.

### Reducing Over Edges of a Vertex

The function \ref TNL::Graphs::reduceVertices computes a reduction over edges adjacent to the specified vertices.
First, we define a vector `vertexMaxWeights` (together with the corresponding vector view `vertexMaxWeights_view `)
to store the results of the reduction:

\snippet Graphs/Reduce/GraphExample_reduceVertices.cpp vector for results

Next, we define the `fetch` lambda function responsible for reading the required data:

\snippet Graphs/Reduce/GraphExample_reduceVertices.cpp fetch lambda

The lambda function `fetch` takes the indices `sourceIdx` and `targetIdx` of the source and target vertices of an edge,
respectively. The third parameter is the edge `weight`. For non-constant graphs, the `weight` parameter may be passed
by reference and can be modified during the `fetch` operation. This allows the reduction and graph modification to be performed simultaneously. For sparse graphs, the `targetIdx` index can also be modified.

Next, we define the `store` lambda function, which is responsible for storing the reduction results for individual vertices:

\snippet Graphs/Reduce/GraphExample_reduceVertices.cpp store lambda

Finally, the reduction is executed as follows:

\snippet Graphs/Reduce/GraphExample_reduceVertices.cpp reduce vertices

Here, we use \ref TNL::Max for reduction (see also \ref ReductionFunctionObjects for other
functionals). Another variant of \ref TNL::Graphs::reduceVertices allows to define the reduction operation via a lambda function.

Note that the function \ref TNL::Graphs::reduceAllVertices is also available for performing reductions over all vertices.

The whole example reads as follows:

\includelineno Graphs/Reduce/GraphExample_reduceVertices.cpp

The output looks as follows:

\include GraphExample_reduceVertices.out

### Reducing Over Edges of Specific Vertices and Conditional Reduction

Reductions over a specified set of vertices can be performed using \ref TNL::Graphs::reduceVertices.
Conditional reductions are provided by \ref TNL::Graphs::reduceAllVerticesIf and \ref TNL::Graphs::reduceVerticesIf.
Their behavior is analogous to the corresponding vertex traversal functions; however, they differ in how the results of the reduction are stored, as demonstrated in the following code snippet.

Assume that we perform a reduction over a given set of vertices:

\snippet Graphs/Reduce/GraphExample_reduceVerticesWithIndexes.cpp reduce vertices with indecis

The `store` lambda function takes the following parameters:

- `indexOfVertexIdx` – the rank of the vertex within the set of vertices being processed.
In this example, the reduction is performed only for vertices 0, 2, and 3.
Their corresponding `indexOfVertexIdx` values are 0, 1, and 2, respectively.
- `vertexIdx` – the index of the vertex whose reduction result is currently being stored.
- `sum` – the result of the reduction, i.e. the sum of the weights of all edges connected to the vertex.

The whole exmpale reads as follows:

\includelineno Graphs/Reduce/GraphExample_reduceVerticesWithIndexes.cpp

The output looks as follows:

\include GraphExample_reduceVerticesWithIndexes.out

The `store` lambda function behaves in the same way for conditional reductions, as demonstrated by the following code snippet:

\snippet Graphs/Reduce/GraphExample_reduceVerticesIf.cpp reduce vertices if

The function \ref TNL::Graphs::reduceVerticesIf also returns the number of vertices (`reducedVertexCount`) for which the
reduction has been performed. This is particularly useful for subsequent processing of data stored in arrays with
compressed results (for example, `compressedVertexMinWeights_view` in this example).

The whole example reads as:

\includelineno Graphs/Reduce/GraphExample_reduceVerticesIf.cpp

The output looks as follows:

\include GraphExample_reduceVerticesIf.out

### Reducing Over Edges of a Vertex with Argument

Reductions with arguments allow tracking graph edges during the computation.
This is useful, for example, when searching for the edge with the maximal (or minimal) weight among edges adjacent to given vertices,
or for similar operations. Such functionality is provided by \ref TNL::Graphs::reduceVerticesWithArgument and related functions
(\ref TNL::Graphs::reduceAllVerticesWithArgument, \ref TNL::Graphs::reduceVerticesWithArgumentIf, \ref TNL::Graphs::reduceAllVerticesWithArgumentIf).

In the following example, for each vertex in a given range, we search for the edge with the minimal weight.
We begin by defining vectors and corresponding vector views to store the results of the reduction:

\snippet Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp vectors for results

The `fetch` lambda function behaves in the same way as in a standard reduction:

\snippet Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp fetch lambda

The `store` lambda function is defined as follows:

\snippet Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp store lambda

In addition to the reduction result, it receives the parameters `localIdx`, `targetIdx`, and the boolean flag `isolatedVertex`:

- `vertexIdx` is the index of the vertex for which the result is being stored.
- `targetIdx` is the index of the target vertex of the edge with the minimal weight.
- `localIdx` is the position of the edge with the minimal weight within the set of all edges adjacent to the vertex `vertexIdx`.
If the graph is dense (i.e., the adjacency matrix is dense), `localIdx` has the same value as `targetIdx` and may be omitted.
- `result` is the value of the minimal edge weight.
- `isolatedVertex` is a boolean flag. If it is set to `true`, the vertex with index `vertexIdx` is isolated and has no adjacent edges.
In this case, result is equal to the identity value of the reduction operation, and both `localIdx` and `targetIdx` are undefined and must be ignored.

The reduction is performed as follows:

\snippet Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp reduce vertices with argument

We use \ref TNL::MinWithArg for reduction (see also \ref ReductionFunctionObjectsWithArgument for other
functionals). Another variant of \ref TNL::Graphs::reduceVerticesWithArgument allows to define the reduction operation via
a lambda function.

The whole example reads as follows:

\includelineno Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp

The output looks as follows:

\include GraphExample_reduceVerticesWithArgument.out

### Reducing Over Edges of Specific Vertices and Conditional Reduction With Argument

The following example demonstrates a reduction with arguments over a specific set of vertices, specified either by their indices or by a condition:

\snippet Graphs/Reduce/GraphExample_reduceVerticesWithArgumentIf.cpp reduce vertices with argument if

The lambda functions `condition` and `fetch` remain unchanged.
The `store` lambda function accepts the following parameters:

- `indexOfVertexIdx` – the position of the vertex whose result is being stored within the set of all vertices reduced during the operation.
This index can be used to store results in a compressed form.
- `vertexIdx` – the index of the vertex whose reduction result is being stored.
- `localIdx` – the position of the edge with the minimal weight within the set of all edges adjacent to the vertex `vertexIdx`.
- `targetIdx` – the index of the target vertex of the edge with the minimal weight.
- `minWeight` – the weight value of the edge with the minimal weight.
- `isolatedVertex` – indicates (if set to `true`) that the vertex with index `vertexIdx` is isolated and has no adjacent edges over which the reduction can be performed.

In this case, the reduction result (minWeight in this example) is set to the identity value of the reduction operation, and the variables `localIdx` and `targetIdx` are undefined and must not be used.

The whole example reads as follows:

\includelineno Graphs/Reduce/GraphExample_reduceVerticesWithArgumentIf.cpp

The output looks as follows:

\include GraphExample_reduceVerticesWithArgumentIf.out

## Graph Views

Similar to matrix views, graph views (\ref TNL::Graphs::GraphView) are lightweight reference objects that allow passing graphs to GPU kernels. The graph views can be obtained from the graph using the method \ref TNL::Graphs::Graph::getView or \ref TNL::Graphs::Graph::getConstView.

Graph views are particularly useful when:
- Passing graphs to custom GPU kernels
- Working with graphs in lambda functions
- Creating shallow copies for parallel algorithms

## Performance Considerations

### Choosing the Right Matrix Format

For sparse graphs, different segment formats offer different performance characteristics. The user may choose any type of segments
 - see \ref TNL::Algorithms::Segments.

For example graph with Ellpack format can be obtained as follows:

```cpp
using EllpackGraph = TNL::Graphs::Graph< float, Device, int,
                                         TNL::Graphs::DirectedGraph,
                                         TNL::Algorithms::Segments::Ellpack >;
```

## Further Reading

- \ref TNL::Graphs::Graph - Main graph class documentation
- \ref TNL::Graphs::GraphView - Graph view documentation
- \ref TNL::Graphs::GraphVertexView - Vertex view documentation
- \ref TNL::Graphs::forAllVertices - Parallel vertex traversal
- \ref TNL::Graphs::forAllVerticesIf - Conditional vertex traversal
- \ref TNL::Graphs::forAllEdges - Parallel edge traversal
- \ref TNL::Graphs::forEdges - Range-based edge traversal
- \ref TNL::Graphs::forAllEdgesIf - Conditional edge traversal
- \ref GraphTraversalOverview - Complete overview of all traversal functions
- \ref GraphReductionOverview - Complete overview of all traversal functions
