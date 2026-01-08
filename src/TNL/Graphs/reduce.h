// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/LaunchConfiguration.h>

namespace TNL::Graphs {

/**
 * \page GraphReductionOverview Overview of Graph Reduction Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all reduction functions available for graph operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section GraphReductionWhatIsReduction What is a Graph Vertex Reduction?
 *
 * A **graph vertex reduction** performs a reduction operation (like sum, max, min) across the edges
 * connected to each vertex of a graph, producing one result per vertex. This is a fundamental operation for:
 * - computation of vertex degrees,
 * - finding maximum/minimum edge weights connected to each vertex,
 * - vertex-level statistics and analysis
 * - algorithms like PageRank, centrality measures, etc.
 *
 * \section GraphReductionFunctionCategories Function Categories
 *
 * Graph reduction functions are organized along three independent axes:
 *
 * \subsection GraphReductionConstVsNonConst Const vs. Non-Const Graph
 *
 * | Category | Graph Modifiable? | Use Case |
 * |----------|-------------------|----------|
 * | **Non-const** | Yes | Can modify graph edges during reduction |
 * | **Const** | No | Read-only access to graph edges |
 *
 * Note: Each reduction function has both const and non-const overloads. The possiblity
 * to modify the graph during reduction alows to fuse reduction with traversing operations
 * into a single pass. This can improve performance by reducing memory accesses.
 *
 * \subsection GraphReductionBasicVsArgument Basic vs. WithArgument Variants
 *
 * | Category | Tracks Position? | Use Case |
 * |----------|-----------------|----------|
 * | **Basic** | No | Only the reduced weight is needed (e.g., vertex sum, vertex max) |
 * | **WithArgument** | Yes | Need weight and target vertex index (e.g., max weight and where it occurs) |
 *
 * \subsection GraphReductionScopeAndConditionalVariants Scope and Conditional Variants (Which Vertices to Process)
 *
 * | Scope | Vertices Processed | Parameters |
 * |-------|---------------|------------|
 * | **All** | All vertices | No range/array parameters |
 * | **Range** | Vertices [begin, end) | `begin` and `end` indices |
 * | **Array** | Specific vertices | Array of vertex indices |
 * | **If** | Vertices filtered by a condition | Process vertices based on vertex-level properties |
 *
 * \section GraphReductionCompleteGraph Complete Function Graph
 *
 * All reduction functions follow this naming pattern:
 * `reduce[All]Vertices[WithArgument][If]`
 *
 * Each function has **both const and non-const overloads** (×2 multiplier).
 *
 * \subsection GraphReductionBasicFunctions Basic Reduction Functions
 *
 * | Function | Scope | Conditional | Tracks Position | Overloads |
 * |----------|-------|-------------|-----------------|----------|
 * | \ref reduceAllVertices | All | No | No | const & non-const |
 * | \ref reduceVertices (range) | Range [begin,end) | No | No | const & non-const |
 * | \ref reduceVertices (array) | Vertex array | No | No | const & non-const |
 * | \ref reduceAllVerticesIf | All | Yes | No | const & non-const |
 * | \ref reduceVerticesIf | Range [begin,end) | Yes | No | const & non-const |
 *
 * \subsection GraphReductionWithArgumentFunctions WithArgument Reduction Functions
 *
 * | Function | Scope | Conditional | Tracks Position | Overloads |
 * |----------|-------|-------------|-----------------|----------|
 * | \ref reduceAllVerticesWithArgument | All | No | Yes | const & non-const |
 * | \ref reduceVerticesWithArgument (range) | Range [begin,end) | No | Yes | const & non-const |
 * | \ref reduceVerticesWithArgument (array) | Vertex array | No | Yes | const & non-const |
 * | \ref reduceAllVerticesWithArgumentIf | All | Yes | Yes | const & non-const |
 * | \ref reduceVerticesWithArgumentIf | Range [begin,end) | Yes | Yes | const & non-const |
 *
 * \section GraphReductionParameters Common Parameters
 *
 * All reduction functions share these common parameters:
 *
 * - **graph**: The graph to reduce (const or non-const)
 * - **fetch**: Lambda that retrieves edge weights (see \ref FetchLambda_NonConst or \ref FetchLambda_Const)
 * - **reduction**: Lambda or function object that combines weights (see \ref ReductionLambda_Basic or \ref
 * ReductionLambda_WithArgument)
 * - **store**: Lambda that stores the reduction results (see \ref StoreLambda_Basic or variants)
 * - **identity**: The identity edge for the reduction (e.g., 0 for addition, 1 for multiplication)
 * - **launchConfig**: Configuration for parallel execution (optional)
 *
 * Additional parameters:
 * - **Scope variants**: `begin`, `end` (range) or `vertexIndexes` (array)
 * - **If variants**: `condition` lambda for vertex filtering (see \ref VertexConditionLambda)
 *
 * \section GraphReductionRelatedPages Related Pages
 *
 * - \ref GraphReductionLambdas - Detailed lambda function signatures
 * - \ref GraphTraversalOverview - Graph traversal operations
 */

/**
 * \page GraphReductionLambdas Graph Reduction Lambda Function Reference
 *
 * This page provides a comprehensive reference for all lambda function signatures used
 * in graph reduction operations.
 *
 * \tableofcontents
 *
 * \section FetchLambdas Fetch Lambda Functions
 *
 * The \e fetch lambda is used to extract and transform weights from graph edges during reduction.
 *
 * \subsection FetchLambda_NonConst For Non-Const Graphs
 *
 * For **non-const graphs** (\ref TNL::Graphs::Graph and \ref TNL::Graphs::GraphView) having a sparse
 * adjacency matrix (\ref TNL::Matrices::SparseMatrix, \ref TNL::Matrices::SparseMatrixView),
 * the signature of the \e fetch lambda is:
 *
 * ```cpp
 * auto fetch = [] __cuda_callable__ ( IndexType sourceIdx, IndexType& targetIdx, RealType& weight ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e sourceIdx - The index of the source graph vertex
 * - \e targetIdx - The index of the target graph vertex  (can be modified)
 * - \e weight - The weight of the graph edge (can be modified)
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * For graphs having other type of the adjacency matrix like dense matrix (\ref TNL::Matrices::DenseMatrix, \ref
 * TNL::Matrices::DenseMatrixView), tridiagonal graphs (\ref TNL::Matrices::TridiagonalMatrix, \ref
 * TNL::Matrices::TridiagonalMatrixView) or multidiagonal (\ref TNL::Matrices::MultidiagonalMatrix, \ref
 * TNL::Matrices::MultidiagonalMatrixView), the column index (representing the target vertex index) is defined implicitly and
 * cannot be changed even for non-constant graphs. The signature then reads as:
 *
 * ```cpp
 * auto fetch = [] __cuda_callable__ ( IndexType sourceIdx, IndexType targetIdx, RealType& weight ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e sourceIdx - The index of the source graph vertex
 * - \e targetIdx - The index of the target graph vertex (passed by value)
 * - \e weight - The weight of the graph edge (can be modified)
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * \subsection FetchLambda_Const For Const Graphs
 *
 * ```cpp
 * auto fetch = [] __cuda_callable__ ( IndexType sourceIdx, IndexType targetIdx, const RealType& weight ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e sourceIdx - The index of the source graph vertex
 * - \e targetIdx - The index of the target graph vertex (passed by value)
 * - \e weight - The weight of the graph edge (const reference)
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * \section ReductionLambdas Reduction Lambda Functions
 *
 * The \e reduction lambda defines how weights are combined during the reduction operation.
 *
 * \subsection ReductionLambda_Basic Basic Reduction (Without Arguments)
 *
 * ```cpp
 * auto reduction = [=] __cuda_callable__ ( const FetchValue& a, const FetchValue& b ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e a - First weight to be reduced
 * - \e b - Second weight to be reduced
 * - Returns: The result of reducing \e a and \e b
 *
 * \subsection ReductionLambda_WithArgument Reduction With Argument (Position Tracking)
 *
 * ```cpp
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) -> Result { ... }
 * ```
 *
 * **Parameters:**
 * - \e a - First weight to be reduced (mutable reference)
 * - \e b - Second weight to be reduced (const reference)
 * - \e aIdx - Index/position associated with weight \e a (mutable reference for tracking)
 * - \e bIdx - Index/position associated with weight \e b (const reference)
 *
 * Note: This variant is used when you need to track which edge produced the final result
 * (e.g., finding the maximum weight and its position).
 *
 * \section StoreLambdas Store Lambda Functions
 *
 * The \e store lambda is used to store the final reduction result for each vertex.
 *
 * \subsection StoreLambda_Basic Basic Store (Vertex Index Only)
 *
 * ```cpp
 * auto store = [=] __cuda_callable__ ( IndexType sourceIdx, const FetchValue& weight ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e sourceIdx - The index of the source graph vertex
 * - \e weight - The result of the reduction for this vertex
 *
 * \subsection StoreLambda_WithLocalIdx Store With Argument (Position Tracking)
 *
 * ```cpp
 * auto store = [=] __cuda_callable__ ( IndexType sourceIdx, IndexType localIdx, IndexType targetIdx, const Value& weight, bool
 * isolatedVertex ) {
 * ... }
 * ```
 *
 * **Parameters:**
 * - \e sourceIdx - The index of the source graph vertex
 * - \e localIdx - The local index of the edge within the vertex (when tracking positions)
 * - \e targetIdx - The index of the target vertex of given edge (when tracking positions)
 * - \e weight - The result of the reduction for this vertex
 * - \e isolatedVertex - Boolean flag indicating whether the vertex has no edges (true if empty). When true, localIdx and
 * targetIdx values are meaningless and should not be used.
 *
 * \subsection StoreLambda_WithIndexArray Store With Vertex Index Array Or Condition
 *
 * ```cpp
 * auto store = [=] __cuda_callable__ ( IndexType indexOfVertexIdx, IndexType sourceIdx, const FetchValue& weight ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e indexOfVertexIdx - The position within the \e vertexIndexes array or the rank in the set of vertices for which the
 * condition was true.
 * - \e sourceIdx - The actual index of the vertex
 * - \e weight - The result of the reduction for this vertex
 *
 * \subsection StoreLambda_WithIndexArrayAndLocalIdx Store With Vertex Index Array and With Argument
 *
 * ```cpp
 * auto store = [=] __cuda_callable__ ( IndexType indexOfVertexIdx, IndexType sourceIdx, IndexType localIdx, IndexType
 * targetIdx, const FetchValue& weight, bool isolatedVertex ) { ...
 * }
 * ```
 *
 * **Parameters:**
 * - \e indexOfVertexIdx - The position within the \e vertexIndexes array or the rank in the set of vertices for which the
 * condition was true.
 * - \e sourceIdx - The index of the source graph vertex
 * - \e localIdx - The position of the edge within the vertex
 * - \e targetIdx - The index of the target vertex of given edge
 * - \e weight - The result of the reduction for this vertex
 * - \e isolatedVertex - Boolean flag indicating whether the vertex has no edges (true if empty). When true, localIdx and
 * targetIdx values are meaningless and should not be used.
 *
 * \section ConditionLambdas Condition Lambda Functions
 *
 * The \e condition lambda determines which vertices should be processed (used in "If" variants).
 *
 * \subsection ConditionLambda Condition Check
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType sourceIdx ) -> bool { ... }
 * ```
 *
 * **Parameters:**
 * - \e sourceIdx - The index of the graph vertex
 * - Returns: \e true if the vertex should be processed, \e false otherwise
 *
 * \section ReductionFunctionObjects Reduction Function Objects
 *
 * Instead of lambda functions, reduction operations can also be specified using function objects
 * from \ref TNL::Algorithms::Segments::ReductionFunctionObjects or
 * \ref TNL::Algorithms::Segments::ReductionFunctionObjectsWithArgument.
 *
 * When using function objects:
 * - They must provide a static template method \e getIdentity to automatically deduce the identity value
 * - For WithArgument variants, they must be instances of \ref ReductionFunctionObjectsWithArgument
 * - Common examples: \e Min, \e Max, \e Sum, \e Product, \e MinWithArg, \e MaxWithArg
 *
 * \section GraphReductionLambdasRelatedPages Related Pages
 *
 * - \ref GraphReductionOverview - Overview of graph reduction functions
 */

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVertices( Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVertices( const Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   const FetchValue& identity,
                   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices
 * with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forAllEdges.cpp
 * \par Output
 * \include GraphExample_forAllEdges.out
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVertices( Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices
 * with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forAllEdges.cpp
 * \par Output
 * \include GraphExample_forAllEdges.out
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVertices( const Graph& graph,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   Store&& store,
                   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for the reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forEdges.cpp
 * \par Output
 * \include GraphExample_forEdges.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVertices( Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for the reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVertices( const Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes
 * with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forEdges.cpp
 * \par Output
 * \include GraphExample_forEdges.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVertices( Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes
 * with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forEdges.cpp
 * \par Output
 * \include GraphExample_forEdges.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVertices( const Graph& graph,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertex indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertex indexes where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of vertex indexes
 *    whose corresponding vertices will be processed for reduction.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithIndexArray.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVertices( Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithIndexArray.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVertices( const Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                const FetchValue& identity,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes
 * with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithIndexArray.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forVerticesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forVerticesWithIndexes.out
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVertices( Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes
 * with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithIndexArray.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forVerticesWithIndexes.cpp
 * \par Output
 * \include GraphExample_forVerticesWithIndexes.out
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVertices( const Graph& graph,
                const Array& vertexIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                Store&& store,
                Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forAllVerticesIf.cpp
 * \par Output
 * \include GraphExample_forAllVerticesIf.out
 */
template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Graph::IndexType
reduceAllVerticesIf( Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     const FetchValue& identity,
                     Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forAllVerticesIf.cpp
 * \par Output
 * \include GraphExample_forAllVerticesIf.out
 */
template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store, typename FetchValue >
typename Graph::IndexType
reduceAllVerticesIf( const Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     const FetchValue& identity,
                     Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition
 * with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forAllVerticesIf.cpp
 * \par Output
 * \include GraphExample_forAllVerticesIf.out
 */
template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesIf( Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition
 * with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forAllVerticesIf.cpp
 * \par Output
 * \include GraphExample_forAllVerticesIf.out
 */
template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesIf( const Graph& graph,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     Store&& store,
                     Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Graph::IndexType
reduceVerticesIf( Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  const FetchValue& identity,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition (const
 * version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation. See \ref ReductionLambda_Basic.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue >
typename Graph::IndexType
reduceVerticesIf( const Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  const FetchValue& identity,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition
 * with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesIf( Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition
 * with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for condition check. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_Basic.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_forVerticesIf.cpp
 * \par Output
 * \include GraphExample_forVerticesIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesIf( const Graph& graph,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  Store&& store,
                  Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices while
 *  returning also the position of the edge of interest.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgument.out
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVerticesWithArgument(
   Graph& graph,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices while
 *  returning also the position of the edge of interest (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgument.out
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store, typename FetchValue >
void
reduceAllVerticesWithArgument(
   const Graph& graph,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices while
 *  returning also the position of the edge of interest with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgument.out
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVerticesWithArgument(
   Graph& graph,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices while
 *  returning also the position of the edge of interest with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgument.out
 */
template< typename Graph, typename Fetch, typename Reduction, typename Store >
void
reduceAllVerticesWithArgument(
   const Graph& graph,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes while
 *  returning also the position of the edge of interest.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction operation with argument. See \ref ReductionLambda_WithArgument.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVerticesWithArgument(
   Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes while
 *  returning also the position of the edge of interest (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction operation with argument. See \ref ReductionLambda_WithArgument.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVerticesWithArgument(
   const Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes while
 *  returning also the position of the edge of interest with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction operation with argument. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVerticesWithArgument(
   Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes while
 *  returning also the position of the edge of interest with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction operation with argument. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
reduceVerticesWithArgument(
   const Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes while
 *  returning also the position of the edge of interest.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithIndexArrayAndLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgument.out
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVerticesWithArgument(
   Graph& graph,
   const Array& vertexIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes while
 *  returning also the position of the edge of interest (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Lambda function for reduction with argument tracking. See \ref ReductionLambda_WithArgument.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithIndexArrayAndLocalIdx.
 * \param identity The initial value for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgument.out
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVerticesWithArgument(
   const Graph& graph,
   const Array& vertexIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes while
 *  returning also the position of the edge of interest with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithIndexArrayAndLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgument.out
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVerticesWithArgument(
   Graph& graph,
   const Array& vertexIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within graph vertices specified by a given set of vertex indexes while
 *  returning also the position of the edge of interest with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Array The type of the array containing the indexes of the vertices to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param vertexIndexes The array containing the indexes of the vertices to iterate over.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithIndexArrayAndLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgument.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgument.out
 */
template< typename Graph,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
void
reduceVerticesWithArgument(
   const Graph& graph,
   const Array& vertexIndexes,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition while
 *  returning also the position of the edge of interest.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param identity The identity edge for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgumentIf.out
 */
template< typename Graph,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Graph::RealType >() ) ) >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf(
   Graph& graph,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition while
 *  returning also the position of the edge of interest (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param identity The identity edge for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgumentIf.out
 */
template< typename Graph,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Graph::RealType >() ) ) >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf(
   const Graph& graph,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition while
 *  returning also the position of the edge of interest with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgumentIf.out
 */
template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf(
   Graph& graph,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over all vertices based on a condition while
 *  returning also the position of the edge of interest with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceAllVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceAllVerticesWithArgumentIf.out
 */
template< typename Graph, typename Condition, typename Fetch, typename Reduction, typename Store >
typename Graph::IndexType
reduceAllVerticesWithArgumentIf(
   const Graph& graph,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition while
 *  returning also the position of the edge of interest.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param identity The identity edge for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgumentIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Graph::RealType >() ) ) >
typename Graph::IndexType
reduceVerticesWithArgumentIf(
   Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition while
 *  returning also the position of the edge of interest (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 * \tparam FetchValue The type returned by the \e Fetch lambda function.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param identity The identity edge for the reduction operation.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgumentIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store,
          typename FetchValue = decltype( std::declval< Fetch >()( 0, 0, std::declval< typename Graph::RealType >() ) ) >
typename Graph::IndexType
reduceVerticesWithArgumentIf(
   const Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   const FetchValue& identity,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition while
 *  returning also the position of the edge of interest with automatic identity deduction.
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_NonConst.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgumentIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesWithArgumentIf(
   Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each graph vertex over a given range of vertex indexes based on a condition while
 *  returning also the position of the edge of interest with automatic identity deduction (const version).
 *
 * See also: \ref GraphReductionOverview
 *
 * \tparam Graph The type of the graph.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of vertices where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam Store The type of the lambda function used for storing results from individual vertices.
 *
 * \param graph The graph on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of vertices where the reduction will be performed.
 * \param condition Lambda function for vertex condition checking. See \ref ConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref FetchLambda_Const.
 * \param reduction Function object for reduction with argument tracking. See \ref ReductionFunctionObjects.
 * \param store Lambda function for storing results with position tracking. See \ref StoreLambda_WithLocalIdx.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \return The number of processed vertices, i.e. vertices for which the condition was true.
 *
 * \par Example
 * \include Graphs/Reduce/GraphExample_reduceVerticesWithArgumentIf.cpp
 * \par Output
 * \include GraphExample_reduceVerticesWithArgumentIf.out
 */
template< typename Graph,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename Store >
typename Graph::IndexType
reduceVerticesWithArgumentIf(
   const Graph& graph,
   IndexBegin begin,
   IndexEnd end,
   Condition&& condition,
   Fetch&& fetch,
   Reduction&& reduction,
   Store&& store,
   Algorithms::Segments::LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  //namespace TNL::Graphs

#include "reduce.hpp"
