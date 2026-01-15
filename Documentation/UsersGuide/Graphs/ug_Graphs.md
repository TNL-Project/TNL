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

TNL supports both directed and undirected graphs through the type tags \ref TNL::Graphs::DirectedGraph and \ref TNL::Graphs::UndirectedGraph:

```cpp
using DirectedGraphType = TNL::Graphs::Graph< float, TNL::Devices::Host, int, TNL::Graphs::DirectedGraph >;
using UndirectedGraphType = TNL::Graphs::Graph< float, TNL::Devices::Host, int, TNL::Graphs::UndirectedGraph >;
```

* **Directed graphs**: Each edge has a direction. An edge from vertex `u` to vertex `v` does not imply an edge from `v` to `u`.
* **Undirected graphs**: Edges have no direction. If there's an edge between vertices `u` and `v`, it can be traversed in both directions.

## Sparse vs. Dense Adjacency Matrices

\anchor ug_graphs_sparse_vs_dense

TNL graphs can use either **sparse** or **dense** adjacency matrices to store edge information. The choice significantly impacts memory usage and performance.

### Sparse Adjacency Matrices And Sparse Graphs (Default)

By default, graphs use sparse matrix representations (typically CSR format). A sparse graph only stores information about edges that actually exist:

```cpp
// Sparse graph - only stores existing edges
TNL::Graphs::Graph< float, TNL::Devices::Host, int, TNL::Graphs::DirectedGraph, TNL::Algorithms::Segments::CSR > graph;
```

The memory usage is proportional to the number of edges in the graph. For unweighted graphs, using `ValueType = bool` avoids storing numerical edge weights and reduces memory overhead, as only the presence or absence of edges is represented.

### Dense Adjacency Matrices And Dense Graphs

Dense graphs store information about all possible edges between all vertex pairs:

```cpp
// Dense graph - stores all possible edges
using DenseGraphType = TNL::Graphs::Graph< float, TNL::Devices::Host, int,
                                           TNL::Graphs::DirectedGraph,
                                           TNL::Algorithms::Segments::CSR, // this is ignored for dense adjacency matrix
                                           TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int > >;
```

The memory usage is proportional to the square of the number of vertices (O(V²)). It is best for complete or nearly complete graphs where most vertex pairs are connected. Missing edges must be represented explicitly by the user (e.g., by a chosen sentinel/encoding), since the matrix representation assumes all vertex pairs are present.

### Comparison Table

| Aspect          | Sparse Graph        | Dense Graph        |
| --------------- | ------------------- | ------------------ |
| **Memory**      | O(V + E)            | O(V²)              |
| **Edge lookup** | O(degree)           | O(1)               |
| **Best for**    | Sparse connectivity | Dense connectivity |

## Constructing Graphs

TNL provides several ways to construct graphs, demonstrated in \ref GraphExample_Constructors.cpp.

### Default Constructor

TODO: Use code snippets from the example.

\include GraphExample_Constructors.cpp

Creates an empty graph:

```cpp
GraphType graph;  // Empty graph with no vertices or edges
```

### Constructor with Vertex Count

Creates a graph with specified number of vertices but no edges:

```cpp
GraphType graph( 5 );  // 5 vertices, 0 edges
```

### Constructor with Initializer List (Sparse)

For both **sparse** and **dense** graphs, specify edges as tuples `(source, target, weight)`:

```cpp
GraphType graph( 5,  // number of vertices
    {  // edges: {source, target, weight}
       { 0, 1, 10.0 }, { 0, 2, 20.0 },
       { 1, 2, 30.0 }, { 1, 3, 40.0 },
       { 2, 3, 50.0 },
       { 3, 0, 60.0 }, { 3, 4, 70.0 }
    } );
```

In a sparse graph, unspecified edges are considered missing.
In a dense graph, unspecified edges are represented by zero weights.

### Constructor with Initializer List (Dense)

For **dense graphs**, specify all edge weights in a 2D structure:

```cpp
DenseGraphType graph( { { 0.0, 10.0, 20.0,  0.0,  0.0 },
                        { 0.0,  0.0, 30.0, 40.0,  0.0 },
                        { 0.0,  0.0,  0.0, 50.0,  0.0 },
                        {60.0,  0.0,  0.0,  0.0, 70.0 },
                        { 0.0,  0.0,  0.0,  0.0,  0.0 } } );
```

### Constructor with std::map

Build graphs from a map of edge pairs to weights:

```cpp
std::map< std::pair< int, int >, float > edgeMap;
edgeMap[ {0, 1} ] = 1.5;
edgeMap[ {0, 2} ] = 2.5;
edgeMap[ {1, 2} ] = 3.5;

GraphType graph( 4, edgeMap );
```

### Constructor from Adjacency Matrix

Create a graph from an existing adjacency matrix:

```cpp
using MatrixType = typename GraphType::AdjacencyMatrixType;
MatrixType matrix( 3, 3 );
matrix.setElements( { {0, 1, 1.0}, {0, 2, 2.0}, {1, 2, 3.0}, {2, 0, 4.0} } );

GraphType graph( matrix );
```

## Setting Edges

After creating a graph, you can set or modify its edges using the `setEdges` method, as shown in \ref GraphExample_setEdges.cpp.

### Setting Edges with Initializer List (Sparse)

For **sparse** and **dense** graphs:

```cpp
GraphType graph;
graph.setVertexCount( 5 );
graph.setEdgeCounts( TNL::Containers::Vector< int, Device >( { 2, 3, 1, 2, 0 } ) );

graph.setEdges( {
    { 0, 1, 10.0 }, { 0, 2, 20.0 },
    { 1, 2, 30.0 }, { 1, 3, 40.0 }, { 1, 4, 50.0 },
    { 2, 3, 60.0 },
    { 3, 0, 70.0 }, { 3, 4, 80.0 }
} );
```

### Setting Edges with Initializer List (Dense)

For **dense graphs**, provide the complete adjacency matrix:

```cpp
DenseGraphType graph;
graph.setDimensions( 5, 5 );

graph.setEdges( { { 0.0, 10.0, 20.0,  0.0,  0.0 },
                  { 0.0,  0.0, 30.0, 40.0, 50.0 },
                  { 0.0,  0.0,  0.0, 60.0,  0.0 },
                  {70.0,  0.0,  0.0,  0.0, 80.0 },
                  { 0.0,  0.0,  0.0,  0.0,  0.0 } } );
```

### Setting Edges with std::map

```cpp
std::map< std::pair< int, int >, float > edgeMap;
edgeMap[ {0, 1} ] = 1.5;
edgeMap[ {0, 2} ] = 2.5;
// ... more edges

graph.setEdges( edgeMap );
```

## Graph Traversal

TNL provides powerful parallel traversal capabilities through the \ref TNL::Graphs::forVertices function and related utilities.

The graph can be traversed either by **vertices** or by **edges**. Vertex-based traversal is performed using a vertex view.

A \ref TNL::Graphs::GraphVertexView provides access to a single vertex and its outgoing edges. Vertex views are the primary way to interact with graph structure in parallel GPU code.

```cpp
TNL::Graphs::forAllVertices(
    graph,
    [] __cuda_callable__ ( typename GraphType::VertexView vertex ) {
        int idx = vertex.getVertexIndex();      // Vertex index
        int degree = vertex.getDegree();        // Number of outgoing edges
    }
);
```

Vertex views also allow modifying edge weights:

```cpp
TNL::Graphs::forAllVertices(
    graph,
    3, 4,  // Process only vertex 3
    [] __cuda_callable__ ( typename GraphType::VertexView vertex ) mutable {
        for( int i = 0; i < vertex.getDegree(); i++ ) {
            vertex.getEdgeWeight( i ) *= 2.0;  // Double all edge weights
        }
    }
);
```

### Traversing By Vertices

#### Processing All Vertices

Iterate over all vertices in parallel:

```cpp
TNL::Graphs::forAllVertices(
    graph,
    [] __cuda_callable__ ( typename GraphType::VertexView& vertex ) {
        // Process vertex ...
    }
);
```

#### Processing a Range of Vertices

Process vertices in a specific range `[begin, end)`:

```cpp
// Process only vertices 1-3 (range [1, 4))
TNL::Graphs::forAllVertices(
    graph,
    1, 4,  // begin, end
    [] __cuda_callable__ ( typename GraphType::VertexView& vertex ) {
        // Process vertex...
    }
);
```

#### Processing Vertices with a Condition

Use `forAllVerticesIf` to process only vertices that meet certain criteria given by the lambda function `condition`:

```cpp
// Define condition: vertices with more than 2 edges
auto condition = [] __cuda_callable__ ( int vertexIdx ) -> bool {
    return graph.getVertexDegree( vertexIdx ) > 2;
};
```

The lambda function takes a vertex index and returns `true` for vertices that should be traversed; all other vertices are skipped.
The function `forAllVerticesIf` is invoked as:

```cpp
// Process only vertices that satisfy the condition
TNL::Graphs::forAllVerticesIf(
    graph,
    condition,
    [] __cuda_callable__ ( typename GraphType::VertexView& vertex ) mutable {
        // Process high-degree vertices...
    }
);
```

See \ref GraphExample_forVerticesIf.cpp for a complete example.

### Traversing By Edges

Access edges from a vertex:

```cpp
TNL::Graphs::forAllVertices(
    graph,
    [] __cuda_callable__ ( auto vertex ) {
        for( int i = 0; i < vertex.getDegree(); i++ ) {
            int target = vertex.getTargetIndex( i );    // Target vertex
            auto weight = vertex.getEdgeWeight( i );    // Edge weight
            // Process edges ...
        }
    }
);
```

This way, each vertex is processed by at most one thread. For higher degrees of parallelism, specialized traversal functions must be used.

#### Processing All Edges

The \ref TNL::Graphs::forAllEdges function iterates over all edges in the graph:

```cpp
TNL::Graphs::forAllEdges(
    graph,
    [] __cuda_callable__ ( int source, int local, int& target, float& weight ) mutable {
        // Process each edge
        // source: source vertex index
        // local: edge index within the source vertex (0-based)
        // target: target vertex index (modifiable)
        // weight: edge weight (modifiable)
    }
);
```


**Important notes:**
- Use `mutable` in the lambda when modifying edge targets or weights
- For sparse graphs, you can modify both `target` and `weight`
- For dense/structured graphs, only `weight` can be modified (target is implicit)
- Multiple threads may process edges of the same vertex in parallel

#### Processing Edges in a Range

Process only edges connected to vertices in a specific range `[begin, end)`:

```cpp
// Process edges from vertices 1-3 (range [1, 4))
TNL::Graphs::forEdges(
    graph,
    1, 4,  // begin, end
    [] __cuda_callable__ ( int source, int local, int& target, float& weight ) mutable {
        // Process edges...
    }
);
```

#### Processing Edges for Specific Vertices

Use an array to specify which vertices' edges to process:

```cpp
TNL::Containers::Array< int, Device > vertexIndexes{ 0, 2, 4 };

TNL::Graphs::forEdges(
    graph,
    vertexIndexes,
    0, vertexIndexes.getSize(),  // Process entire array
    [] __cuda_callable__ ( int source, int local, int& target, float& weight ) mutable {
        // Process edges of vertices 0, 2, and 4
    }
);
```

#### Conditional Edge Traversal

Use \ref TNL::Graphs::forAllEdgesIf to process edges only from vertices meeting specific criteria:

```cpp
// Define condition: vertices with degree > 2
auto condition = [] __cuda_callable__ ( int vertexIdx ) -> bool {
    return graph.getVertexDegree( vertexIdx ) > 2;
};

// Process edges only from high-degree vertices
TNL::Graphs::forAllEdgesIf(
    graph,
    condition,
    [] __cuda_callable__ ( int source, int local, int& target, float& weight ) mutable {
        // Process edges...
    }
);
```

#### Const vs. Non-Const Edge Traversal

For const graphs, edge data is read-only:

```cpp
const auto& constGraph = graph;

TNL::Graphs::forAllEdges(
    constGraph,
    [] __cuda_callable__ ( int source, int local, int target, const float& weight ) {
        // target and weight are read-only
        // No 'mutable' needed
    }
);
```

A complete example is available in \ref GraphExample_forEdges.cpp.


**Important**: Use `mutable` in the lambda when modifying the graph!

### Computing Vertex Properties

Use vertex views to compute aggregate properties:

```cpp
// Compute sum of outgoing edge weights for each vertex
TNL::Containers::Array< float, Device > weightSums( graph.getVertexCount() );
auto weightSumsView = weightSums.getView();

TNL::Graphs::forAllVertices(
    graph,
    [=] __cuda_callable__ ( typename GraphType::VertexView vertex ) mutable {
        float sum = 0.0;
        for( int i = 0; i < vertex.getDegree(); i++ ) {
            sum += vertex.getEdgeWeight( i );
        }
        weightSumsView[ vertex.getVertexIndex() ] = sum;
    }
);
```

### Const Vertex Views

When working with const graphs, vertex views are also const:

```cpp
const auto& constGraph = graph;

TNL::Graphs::forAllVertices(
    constGraph,
    [] __cuda_callable__ ( typename GraphType::VertexView vertex ) {
        // vertex is const - can read but not modify
        int degree = vertex.getDegree();
        auto weight = vertex.getEdgeWeight( 0 );
        // vertex.getEdgeWeight( 0 ) = 5.0;  // ❌ Error: vertex is const
    }
);
```

A complete example is available in \ref GraphExample_VertexView.cpp.

## Reductions on Graphs

Graph reductions allow you to compute aggregate values across edges or vertices efficiently. TNL provides specialized reduction operations for graph structures.

### Reducing Over Edges of a Vertex

Compute aggregate values from edges using reductions:

```cpp
// Find maximum edge weight for each vertex
TNL::Containers::Array< float, Device > maxWeights( graph.getVertexCount() );
auto maxWeightsView = maxWeights.getView();

TNL::Graphs::forAllVertices(
    graph,
    [=] __cuda_callable__ ( auto vertex ) mutable {
        float maxWeight = 0.0;
        for( int i = 0; i < vertex.getDegree(); i++ ) {
            maxWeight = max( maxWeight, vertex.getEdgeWeight( i ) );
        }
        maxWeightsView[ vertex.getVertexIndex() ] = maxWeight;
    }
);
```

### Common Reduction Patterns

**Sum of edge weights**:
```cpp
float sum = 0.0;
for( int i = 0; i < vertex.getDegree(); i++ ) {
    sum += vertex.getEdgeWeight( i );
}
```

**Count edges satisfying a condition**:
```cpp
int count = 0;
for( int i = 0; i < vertex.getDegree(); i++ ) {
    if( vertex.getEdgeWeight( i ) > threshold ) {
        count++;
    }
}
```

**Find edge with maximum weight**:
```cpp
int maxIdx = -1;
float maxWeight = -INFINITY;
for( int i = 0; i < vertex.getDegree(); i++ ) {
    if( vertex.getEdgeWeight( i ) > maxWeight ) {
        maxWeight = vertex.getEdgeWeight( i );
        maxIdx = vertex.getTargetIndex( i );
    }
}
```

## Graph Views

Similar to matrix views, graph views (\ref TNL::Graphs::GraphView) are lightweight reference objects that allow passing graphs to GPU kernels:

```cpp
template< typename Device >
void processGraph()
{
    using GraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::DirectedGraph >;
    GraphType graph( 5, { {0, 1, 1.0}, {1, 2, 2.0}, {2, 3, 3.0} } );

    auto graphView = graph.getView();  // Get view

    // Pass view to parallel operations
    TNL::Graphs::forAllVertices(
        graphView,  // Use view instead of graph
        [] __cuda_callable__ ( auto vertex ) {
            // Process vertex...
        }
    );
}
```

Graph views are particularly useful when:
- Passing graphs to custom GPU kernels
- Working with graphs in lambda functions
- Creating shallow copies for parallel algorithms

## Undirected Graphs

For undirected graphs, edges are bidirectional. When constructing an undirected graph, you typically only specify each edge once:

```cpp
using UndirectedGraphType = TNL::Graphs::Graph< float, Device, int, TNL::Graphs::UndirectedGraph >;

UndirectedGraphType graph( 4,
    {  // Only specify edges once
       { 0, 1, 1.0 },
       { 0, 2, 2.0 },
       { 1, 2, 3.0 },
       { 2, 3, 4.0 }
    },
    TNL::Matrices::MatrixElementsEncoding::SymmetricMixed
);
```

The `SymmetricMixed` encoding ensures that the graph properly maintains symmetry (bidirectional edges).

## Performance Considerations

### Choosing the Right Matrix Format

For sparse graphs, different segment formats offer different performance characteristics:

- **CSR** (Compressed Sparse Row): General-purpose, good memory efficiency
- **Ellpack**: Better GPU performance for graphs with similar vertex degrees
- **ChunkedEllpack**: Balanced performance for varying vertex degrees
- **SlicedEllpack**: Excellent GPU performance with moderate memory overhead

Example with Ellpack format:
```cpp
using EllpackGraph = TNL::Graphs::Graph< float, Device, int,
                                         TNL::Graphs::DirectedGraph,
                                         TNL::Algorithms::Segments::Ellpack >;
```

## Summary

For complete working examples, see:
- \ref GraphExample_Constructors.cpp - Various graph construction methods
- \ref GraphExample_setEdges.cpp - Setting and modifying edges
- \ref GraphExample_VertexView.cpp - Working with vertex views and parallel traversal
- \ref GraphExample_forEdges.cpp - Direct edge traversal
- \ref GraphExample_forEdgesIf.cpp - Conditional edge traversal
- \ref GraphExample_forEdgesWithIndexes.cpp - Edge traversal for specific vertices

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
