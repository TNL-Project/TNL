// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs {

/**
 * \page GraphTraversalOverview Overview of Graph Traversal Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all traversal functions available for graph operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section GraphTraversalFunctionCategories Function Categories
 *
 * Graph traversal functions are organized along three main dimensions:
 *
 * \subsection GraphTraversalConstVsNonConst Const vs. Non-Const Graph
 *
 * | Category | Graph Modifiable? | Use Case |
 * |----------|-------------------|----------|
 * | **Non-const** | Yes | Can modify graph edges and structure |
 * | **Const** | No | Read-only access to graph edges |
 *
 * Note: Each traversal function has **both const and non-const overloads**.
 *
 * \subsection GraphTraversalEdgeVsVertex Edge-wise vs. Vertex-wise Traversal
 *
 * | Category | Operates On | Lambda Parameter | Use Case |
 * |----------|------------|------------------|----------|
 * | **Edge-wise** (`forEdges`, `forAllEdges`) | Individual edges | Edge indices & weights | Operate on each
 * edge separately |
 * | **Vertex-wise** (`forVertices`, `forAllVertices`) | Individual vertices | VertexView object | Operate on vertices
 * |
 *
 * \subsection GraphTraversalScopeAndConditional Scope and Conditional Variants
 *
 * Similar to other graph operations, traversal functions have different scope and conditional variants.
 * All traversal functions follow this naming pattern: `for[All]Vertices[If]` or `for[All]Edges[If]`
 *
 * | Scope | Vertices Processed | Parameters |
 * |-------|---------------|------------|
 * | **All** | All vertices | No range/array parameters |
 * | **Range** | Vertices in [begin, end) | `begin` and `end` indices |
 * | **Array** | Specific vertices | Array of vertex indices |
 * | **If** | Vertices filtered by a condition | Process vertices based on vertex-level properties |
 *
 * \section GraphTraversalEdgeFunctions Edge-wise Traversal Functions
 *
 * These functions iterate in parallel over individual edges connected to given set of vertices:
 *
 * \subsection GraphTraversalBasicEdgeFunctions Basic Edge Traversal
 *
 * | Function | Vertices Processed | Description | Overloads |
 * |----------|---------------|-------------|----------|
 * | \ref forAllEdges | All vertices | Process all edges connected to all vertices | const & non-const |
 * | \ref forEdges (range) | Vertices in [begin, end) | Process edges connected to vertices in given range vertex indexes  |
 * const & non-const |
 * | \ref forEdges (array) | Vertices in array | Process edges connected to specified vertices | const & non-const | | \ref
 * forAllEdgesIf | All vertices | Vertex-level condition | const & non-const | | \ref forEdgesIf | Vertices in [begin, end) |
 * Vertex-level condition | const & non-const |
 *
 * Note: Edge-wise traversal functions iterates over the given set of vertices and they traverse all edges connected to
 * each vertex in parallel. Therefore some edges may be traversed twice if both source and target vertices are included in the
 * set. Since the edges are traversed in parallel, one edge may be processed by **multiple threads at the same time**.
 *
 * **When to use:**
 * - Edge-level operations (weight updates, edge filtering)
 * - Traversals that need to modify edge indices or weights
 *
 * \section GraphTraversalVertexFunctions Vertex-wise Traversal Functions
 *
 * These functions iterate over vertices using \e VertexView:
 *
 * \subsection GraphTraversalBasicVertexFunctions Basic Vertex Traversal
 *
 * | Function | Vertices Processed | Description | Overloads |
 * |----------|---------------|-------------|----------|
 * | \ref forAllVertices | All vertices | Process all vertices | const & non-const |
 * | \ref forVertices (range) | Vertices in [begin, end) | Process vertices in range | const & non-const |
 * | \ref forVertices (array) | Vertices in array | Process specified vertices | const & non-const |
 * | \ref forAllVerticesIf | All vertices | Vertex-level condition | const & non-const |
 * | \ref forVerticesIf | Vertices in [begin, end) | Vertex-level condition | const & non-const |
 *
 * Note: Vertex-wise traversal functions process each vertex as a whole using \e VertexView. No vertex is processed by
 * multiple threads at the same time.
 *
 * **When to use:**
 * - Vertex-level operations
 *
 * \section GraphTraversalParameters Common Parameters
 *
 * All traversal functions share these common parameters:
 *
 * - `graph`  The graph to traverse (const or non-const)
 * - `function`  Lambda function to apply (see \ref TraversalFunction_NonConst, \ref TraversalFunction_Const, \ref
 * TraversalVertexFunction_NonConst, or \ref TraversalVertexFunction_Const)
 * - `launchConfig`  Configuration for parallel execution (optional)
 *
 * Additional parameters:
 * - **Scope variants**: `begin`, `end` (range) or `vertexIndexes` (array)
 * - **If variants**: `condition` lambda for filtering (see \ref TraversalConditionLambda)
 *
 * \section GraphTraversalUsageGuidelines Usage Guidelines
 *
 * **Performance considerations: **
 *
 * Edge-wise traversal allows parallel processing of edges incident to the same vertex; multiple threads may therefore operate
 * on edges associated with a single vertex.
 *
 * Vertex-wise traversal assigns at most one thread to each vertex and never maps multiple threads to the same vertex. This
 * traversal mode is therefore preferred when vertex-level context or per-vertex state updates are required. *
 *
 * \section GraphTraversalRelatedPages Related Pages
 *
 * - \ref GraphTraversalLambdas - Detailed lambda function signatures
 * - \ref GraphReductionOverview - Graph reduction operations
 */

/**
 * \page GraphTraversalLambdas Graph Traversal Lambda Functions Reference
 *
 * This page documents the lambda function signatures used in graph traversal operations.
 *
 * \section TraversalFunction_NonConst Traversal Function (Non-Const Graph)
 *
 * For **non-const graphs with sparse adjacency matrix** (\ref Matrices::SparseMatrix), the traversal function has full access
 * to modify edge indices and weights:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( IndexType sourceIdx, IndexType localIdx, IndexType& targetIdx, RealType& weight ) {
 * ... }
 * ```
 *
 * **Parameters:**
 * - `sourceIdx` - The index of the graph vertex to which the edge belongs
 * - `localIdx` - The rank/position of the edge within the graph vertex (0-based)
 * - `targetIdx` - Reference to the target vertex index (can be modified)
 * - `weight` - Reference to the edge weight (can be modified)
 *
 * For other types of adjacent matrices like dense (\ref TNL::Matrices::DenseMatrix ),
 * tridiagonal matrices (\ref TNL::Matrices::TridiagonalMatrix) or multidiagonal (\ref TNL::Matrices::MultidiagonalMatrix),
 * the index of the target vertex is defined implicitly and cannot be changed even for non-constant graphss. The signature
 * then reads as:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, RealType& weight ) {
 * ... }
 *
 * ```
 * **Parameters:**
 * - `sourceIdx` - The index of the graph vertex to which the edge belongs
 * - `localIdx` - The rank/position of the edge within the graph vertex (0-based)
 * - `targetIdx` - The index of the target vertex
 * - `weight` - Reference to the edge weight (can be modified)
 *
 * \section TraversalFunction_Const Traversal Function (Const Graph)
 *
 * For constant graphs, the traversal function has read-only access:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const RealType& weight
 * ) {
 * ... }
 * ```
 *
 * **Parameters:**
 * - `sourceIdx` - The index of the graph vertex to which the edge belongs
 * - `localIdx` - The rank/position of the edge within the graph vertex (0-based)
 * - `targetIdx` - The index of the target vertex (read-only)
 * - `weight` - Const reference to the edge weight (read-only)
 *
 * \section TraversalVertexFunction_NonConst Vertex Traversal Function (Non-Const Graph)
 *
 * When traversing particular vertices with vertex-level operations, the function receives the vertex index:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( typename Graph::VertexView vertex ) { ... }
 * ```
 *
 * **Parameters:**
 * - `vertex` - The view of graph vertex being processed
 *
 * \section TraversalVertexFunction_Const Vertex Traversal Function (Const Graph)
 *
 * Same signature for const graphs:
 *
 * ```cpp
 * auto function = [=] __cuda_callable__ ( typename Graph::ConstVertexView vertex ) { ... }
 * ```
 *
 * **Parameters:**
 * - `vertex` - The constant view of graph vertex being processed
 *
 * \section TraversalConditionLambda Condition Lambda
 *
 * For conditional traversal operations (`forEdgesIf`, `forVerticesIf`), a condition function determines
 * which vertices to process:
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType vertexIdx ) -> bool { ... }
 * ```
 *
 * **Parameters:**
 * - `vertexIdx` - The index of the graph vertex to check
 *
 * **Returns:**
 * - `true` if the vertex should be processed, `false` to skip it
 *
 * \section GraphTraversalLambdasRelatedPages Related Pages
 *
 * - \ref GraphTraversalOverview - Overview of graph traversal functions
 */

/**
 * \brief Iterates in parallel over all edges of **all** graph vertices and applies the specified lambda function.
 *
 * See also: \ref GraphTraversalOverview
 *
 *
 * \tparam Graph The type of the graph.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdges.cpp
 * \par Output
 * \include GraphExample_forEdges.out
 */
template< typename Graph, typename Function >
void
forAllEdges( Graph& graph,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of **all** graph vertices of **constant graph** and
 * applies the specified lambda function.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdges.cpp
 * \par Output
 * \include GraphExample_forEdges.out
 */
template< typename Graph, typename Function >
void
forAllEdges( const Graph& graph,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges in the given range of graph vertices and applies the specified lambda function.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices whose edges we want to process using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices whose edges we want to process using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdges.cpp
 * \par Output
 * \include GraphExample_forEdges.out
 */
template< typename Graph, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( Graph& graph,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of **constant graph** in the given range of graph vertices and applies the
 * specified lambda function.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices whose edges we want to process using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices whose edges we want to process using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdges.cpp
 * \par Output
 * \include GraphExample_forEdges.out
 */
template< typename Graph, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( const Graph& graph,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of graph vertices with the given indexes and applies the specified lambda
 * function.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertex indexes whose edges will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertex indexes whose edges will be processed using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertex indexes whose edges
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertex indexes whose edges
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forEdgesWithIndexes.out
 */
template< typename Graph, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( Graph& graph,
          const Array& vertexIndexes,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of graph vertices with the given indexes and applies the specified lambda
 * function. This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertex indexes whose edges will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertex indexes whose edges will be processed using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertex indexes whose edges
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertex indexes whose edges
 *    will be processed using the lambda function.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forEdgesWithIndexes.out
 */
template< typename Graph, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forEdges( const Graph& graph,
          const Array& vertexIndexes,
          IndexBegin begin,
          IndexEnd end,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of graph vertices with the given indexes and applies the specified lambda
 * function.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forEdgesWithIndexes.out
 */
template< typename Graph, typename Array, typename Function >
void
forEdges( Graph& graph,
          const Array& vertexIndexes,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of graph vertices with the given indexes and applies the specified lambda
 * function. This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forEdgesWithIndexes.out
 */
template< typename Graph, typename Array, typename Function >
void
forEdges( const Graph& graph,
          const Array& vertexIndexes,
          Function&& function,
          Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges in a given range of vertices based on a condition.
 *
 * See also: \ref GraphTraversalOverview
 *
 *
 * For each graph vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, all edges of the vertex are traversed,
 * and the specified lambda function is applied to each edge. If the condition lambda function returns
 * \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices whose edges will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices whose edges will be processed using the lambda function.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param condition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesIf.cpp
 * \par Output
 * \include GraphExample_forEdgesIf.out
 */
template< typename Graph, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forEdgesIf( Graph& graph,
            IndexBegin begin,
            IndexEnd end,
            Condition&& condition,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges in a given range of vertices based on a condition. This function is for
 * **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * For each graph vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, all edges of the vertex are traversed,
 * and the specified lambda function is applied to each edge. If the condition lambda function returns
 * \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices whose edges will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices whose edges will be processed using the lambda function.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertices whose edges
 *    will be processed using the lambda function.
 * \param condition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesIf.cpp
 * \par Output
 * \include GraphExample_forEdgesIf.out
 */
template< typename Graph, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forEdgesIf( const Graph& graph,
            IndexBegin begin,
            IndexEnd end,
            Condition&& condition,
            Function&& function,
            Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of **all** graph vertices based on a condition.
 *
 * See also: \ref GraphTraversalOverview
 *
 * For each graph vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, all edges of the vertex are traversed,
 * and the specified lambda function is applied to each edge. If the condition lambda function returns
 * \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param condition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesIf.cpp
 * \par Output
 * \include GraphExample_forEdgesIf.out
 */
template< typename Graph, typename Condition, typename Function >
void
forAllEdgesIf( Graph& graph,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all edges of **all** graph vertices based on a condition.
 *
 * See also: \ref GraphTraversalOverview
 *
 * This function is for **constant matrices**.
 *
 * For each graph vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, all edges of the vertex are traversed,
 * and the specified lambda function is applied to each edge. If the condition lambda function returns
 * \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each edge.
 *
 * \param graph The graph whose edges will be processed using the lambda function.
 * \param condition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each edge. See \ref TraversalFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forEdgesIf.cpp
 * \par Output
 * \include GraphExample_forEdgesIf.out
 */
template< typename Graph, typename Condition, typename Function >
void
forAllEdgesIf( const Graph& graph,
               Condition&& condition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over graph vertices within the specified range of vertex indexes
 * and applies the given lambda function to each vertex.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of graph vertices on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of graph vertices on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of graph vertices
 *    that will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of graph vertices
 *    that will be processed using the lambda function.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVertices.cpp
 * \par Output
 * \include GraphExample_forVertices.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forVertices( Graph& graph,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over graph vertices within the specified range of vertex indexes
 * and applies the given lambda function to each vertex. This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of graph vertices on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of graph vertices on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of graph vertices
 *    that will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of graph vertices
 *    that will be processed using the lambda function.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVertices.cpp
 * \par Output
 * \include GraphExample_forVertices.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forVertices( const Graph& graph,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** graph vertices and applies the given lambda function to each vertex.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVertices.cpp
 * \par Output
 * \include GraphExample_forVertices.out
 */
template< typename Graph, typename Function >
void
forAllVertices( Graph& graph,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** graph vertices and applies the given lambda function to each vertex.
 * This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 *
 * \tparam Graph The type of the graph.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVertices.cpp
 * \par Output
 * \include GraphExample_forVertices.out
 */
template< typename Graph, typename Function >
void
forAllVertices( const Graph& graph,
                Function&& function,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over graph vertices with the given indexes and applies the specified
 * lambda function to each vertex.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forVerticesWithIndexes.out
 */
template< typename Graph,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< IsArrayType< Array >::value
                                         && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forVertices( Graph& graph,
             const Array& vertexIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over graph vertices with the given indexes and applies the specified
 * lambda function to each vertex. This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forVerticesWithIndexes.out
 */
template< typename Graph,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< IsArrayType< Array >::value
                                         && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forVertices( const Graph& graph,
             const Array& vertexIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over graph vertices with the given indexes and applies the specified
 * lambda function to each vertex.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forVerticesWithIndexes.out
 */
template< typename Graph, typename Array, typename Function, typename T = std::enable_if_t< IsArrayType< Array >::value > >
void
forVertices( Graph& graph,
             const Array& vertexIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over graph vertices with the given indexes and applies the specified
 * lambda function to each vertex. This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the graph vertices to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param vertexIndexes The array containing the indexes of the graph vertices to iterate over.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forVerticesWithIndexes.out
 */
template< typename Graph, typename Array, typename Function, typename T = std::enable_if_t< IsArrayType< Array >::value > >
void
forVertices( const Graph& graph,
             const Array& vertexIndexes,
             Function&& function,
             Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );
/**
 * \brief Iterates in parallel over vertices within the given range of vertex indexes, applying a condition
 * to determine whether each vertex should be processed.
 *
 * See also: \ref GraphTraversalOverview
 *
 * For each vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the vertex.
 * If the condition lambda function returns \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam VertexCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param vertexCondition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename VertexCondition,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forVerticesIf( Graph& graph,
               IndexBegin begin,
               IndexEnd end,
               VertexCondition&& vertexCondition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over vertices within the given range of vertex indexes, applying a condition
 * to determine whether each vertex should be processed. This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * For each vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the vertex.
 * If the condition lambda function returns \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices on which the lambda function will be applied.
 * \tparam VertexCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed using the lambda function.
 * \param vertexCondition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename VertexCondition,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forVerticesIf( const Graph& graph,
               IndexBegin begin,
               IndexEnd end,
               VertexCondition&& vertexCondition,
               Function&& function,
               Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** graph vertices, applying a condition
 * to determine whether each vertex should be processed.
 *
 * See also: \ref GraphTraversalOverview
 *
 * For each vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the vertex.
 * If the condition lambda function returns \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param vertexCondition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_NonConst.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph, typename VertexCondition, typename Function >
void
forAllVerticesIf( Graph& graph,
                  VertexCondition&& vertexCondition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** graph vertices, applying a condition
 * to determine whether each vertex should be processed. This function is for **constant matrices**.
 *
 * See also: \ref GraphTraversalOverview
 *
 * For each vertex, a condition lambda function is evaluated based on the vertex index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the vertex.
 * If the condition lambda function returns \e false, the vertex is skipped.
 *
 * \tparam Graph The type of the graph.
 * \tparam VertexCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each vertex.
 *
 * \param graph The graph on which the lambda function will be applied.
 * \param vertexCondition Lambda function to check vertex condition. See \ref TraversalConditionLambda.
 * \param function Lambda function to be applied to each vertex. See \ref TraversalVertexFunction_Const.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Traverse/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph, typename VertexCondition, typename Function >
void
forAllVerticesIf( const Graph& graph,
                  VertexCondition&& vertexCondition,
                  Function&& function,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  //namespace TNL::Graphs

#include "traverse.hpp"
