// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SubGraphTestBase.h"
#include <TNL/Graphs/traverse.h>

TYPED_TEST_SUITE( SubGraphTest, SubGraphTestTypes );

// =========================================================================
// Construction tests
// =========================================================================

// Original graph (see SubGraphTestBase.h):
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// SubGraph (no filter): identical to the original — all 5 vertices active,
// all 5 edges present.
template< typename GraphType >
void
test_makeSubGraph_no_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph( graph );
   EXPECT_EQ( sg.getVertexCount(), 5 );
   // All vertices are active.
   for( IndexType v = 0; v < 5; ++v )
      EXPECT_TRUE( sg.isActive( v ) );
   // All edges are present (filter accepts all).
   EXPECT_TRUE( sg.edgeExists( IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
   // edgeExists tests the edge filter only; it does NOT verify that the edge
   // is actually present in the adjacency matrix. With AcceptAllEdges, even
   // non-existent edges return true. Use forAllEdges + counter to verify
   // actual edge presence (done in SubGraphTest_Traverse.h).
}

TYPED_TEST( SubGraphTest, makeSubGraph_no_filter )
{
   test_makeSubGraph_no_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Vertex filter `v != 2`: vertex 2 is removed (inactive).
// Edges are unchanged at the filter level, but `forAllEdges` skips vertex 2.
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    X          3 --(5)--> 4     (X = vertex 2 inactive, no outgoing edge)
template< typename GraphType >
void
test_makeSubGraph_vertex_filter()
{
   using IndexType = typename GraphType::IndexType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );
   EXPECT_EQ( sg.getVertexCount(), 5 );
   EXPECT_TRUE( sg.isActive( 0 ) );
   EXPECT_TRUE( sg.isActive( 1 ) );
   EXPECT_FALSE( sg.isActive( 2 ) );
   EXPECT_TRUE( sg.isActive( 3 ) );
   EXPECT_TRUE( sg.isActive( 4 ) );
}

TYPED_TEST( SubGraphTest, makeSubGraph_vertex_filter )
{
   test_makeSubGraph_vertex_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Edge filter `w <= 3`: only edges with weight 1, 2, 3 survive.
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2          3          4     (edges (2,3,w=4) and (3,4,w=5) filtered out)
template< typename GraphType >
void
test_makeSubGraph_edge_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      TNL::Graphs::edgeOnly,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );
   EXPECT_TRUE( sg.edgeExists( IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_FALSE( sg.edgeExists( IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_FALSE( sg.edgeExists( IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, makeSubGraph_edge_filter )
{
   test_makeSubGraph_edge_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Both filters: vertex filter `v != 2` (vertex 2 inactive) AND edge filter `w <= 3`.
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    X          3          4     (vertex 2 inactive; edges (2,3,4) and (3,4,5) filtered out)
template< typename GraphType >
void
test_makeSubGraph_both_filters()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      },
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );
   EXPECT_FALSE( sg.isActive( IndexType( 2 ) ) );
   EXPECT_FALSE( sg.edgeExists( IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_TRUE( sg.isActive( IndexType( 0 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   // edge (0,2) passes the edge filter (w=2<=3) but target 2 is inactive -> edge does not exist
   EXPECT_FALSE( sg.edgeExists( IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( sg.edgeExists( IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_FALSE( sg.edgeExists( IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, makeSubGraph_both_filters )
{
   test_makeSubGraph_both_filters< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_makeSubGraph_indexed()
{
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using CounterVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   IndexVectorType indexes{ 0, 1, 3, 4 };
   auto sg = TNL::Graphs::makeSubGraph( graph, indexes );
   EXPECT_EQ( sg.getVertexCount(), 5 );
   // Edges between active vertices {0,1,3,4}: (0,1), (1,3), (3,4) -> 3.
   // Edge (0,2) is filtered because target 2 is inactive.
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   EXPECT_EQ( counter.getElement( 0 ), 3 );
}

TYPED_TEST( SubGraphTest, makeSubGraph_indexed )
{
   test_makeSubGraph_indexed< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_makeSubGraph_indexed_with_edge_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   using CounterVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   IndexVectorType indexes{ 0, 1, 3 };
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      indexes,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 2;
      } );
   EXPECT_EQ( sg.getVertexCount(), 5 );
   // vertices {0,1,3} active, edges with w<=2 between active: only (0,1,w=1).
   // Edge (0,2,w=2) is filtered because target 2 is inactive.
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   EXPECT_EQ( counter.getElement( 0 ), 1 );
}

TYPED_TEST( SubGraphTest, makeSubGraph_indexed_with_edge_filter )
{
   test_makeSubGraph_indexed_with_edge_filter< typename TestFixture::GraphType >();
}

// AGENT-TODO (resolved — documented limitation): Make a test of subgraph of subgraph.
//
// Subgraph-of-subgraph CANNOT be constructed via `makeSubGraph` with the
// current API. The SubGraph class declares two constructors:
//
//   1. SubGraph(const GraphType& graph, VertexFilter, EdgeFilter);
//   2. SubGraph(ConstGraphView graphView, VertexFilter, EdgeFilter);
//
// where `GraphType = std::decay_t<Graph_>` and `ConstGraphView = GraphType::ConstViewType`.
// For SubGraph, `ConstViewType` is an alias for `SubGraph<...>` itself (SubGraph IS its own
// view). Therefore, when `Graph_` is a SubGraph, both constructors take the same first
// argument type (`const SubGraph<...>&`), and overload resolution is AMBIGUOUS:
//
//   auto inner = makeSubGraph(graph, v != 2);
//   auto outer = makeSubGraph(inner, edgeOnly, w <= 3);  // ERROR: ambiguous constructor
//
// Even if construction were disambiguated (e.g. by SFINAE or by adding a tag), the
// filters would still NOT compose: `TraversingOperations<SubGraph<...>>` reads only
// `graph.getVertexFilter()` / `graph.getEdgeFilter()` of the OUTER SubGraph, and the
// underlying adjacency matrix view is forwarded unchanged from the inner graph. The
// inner SubGraph's filters would be stored but never consulted.
//
// Recommendation: to compose filters, build a single SubGraph with combined predicates:
//   auto composed = makeSubGraph(graph,
//       [&](Index v) { return innerVF(v) && outerVF(v); },
//       [&](Index s, Index t, Value w) { return innerEF(s,t,w) && outerEF(s,t,w); });

// =========================================================================
// Interface tests
// =========================================================================

template< typename GraphType >
void
test_interface()
{
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph( graph );
   EXPECT_EQ( sg.getVertexCount(), 5 );
   EXPECT_TRUE( sg.isDirected() );
   EXPECT_FALSE( sg.isUndirected() );
}

TYPED_TEST( SubGraphTest, interface )
{
   test_interface< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_getView_getConstView()
{
   using IndexType = typename GraphType::IndexType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );
   auto view = sg.getView();
   auto constView = sg.getConstView();
   EXPECT_EQ( view.getVertexCount(), 5 );
   EXPECT_EQ( constView.getVertexCount(), 5 );
   EXPECT_FALSE( view.isActive( 2 ) );
   EXPECT_FALSE( constView.isActive( 2 ) );
   EXPECT_TRUE( view.isActive( 0 ) );
   EXPECT_TRUE( constView.isActive( 0 ) );
}

TYPED_TEST( SubGraphTest, getView_getConstView )
{
   test_getView_getConstView< typename TestFixture::GraphType >();
}

// =========================================================================
// Free-function adapter tests
// =========================================================================

template< typename GraphType >
void
test_adapters_isActive()
{
   using IndexType = typename GraphType::IndexType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );
   // Graph: always true
   EXPECT_TRUE( TNL::Graphs::isActive( graph, IndexType( 0 ) ) );
   EXPECT_TRUE( TNL::Graphs::isActive( graph, IndexType( 2 ) ) );
   // SubGraph: delegates to filter
   EXPECT_TRUE( TNL::Graphs::isActive( sg, IndexType( 0 ) ) );
   EXPECT_FALSE( TNL::Graphs::isActive( sg, IndexType( 2 ) ) );
}

TYPED_TEST( SubGraphTest, adapters_isActive )
{
   test_adapters_isActive< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_adapters_edgeExists()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      TNL::Graphs::edgeOnly,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );
   // Graph: always true
   EXPECT_TRUE( TNL::Graphs::edgeExists( graph, IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_TRUE( TNL::Graphs::edgeExists( graph, IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
   // SubGraph: delegates to filter
   EXPECT_TRUE( TNL::Graphs::edgeExists( sg, IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_FALSE( TNL::Graphs::edgeExists( sg, IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, adapters_edgeExists )
{
   test_adapters_edgeExists< typename TestFixture::GraphType >();
}

// =========================================================================
// Materialize tests
// =========================================================================

namespace {

template< typename Graph, typename Index, typename Value >
bool
hasEdge( const Graph& g, Index src, Index tgt, Value w )
{
   using DeviceType = typename Graph::DeviceType;
   using CounterVector = TNL::Containers::Vector< Index, DeviceType, Index >;
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      g,
      [ = ] __cuda_callable__( Index s, Index, Index t, const Value& weight ) mutable
      {
         if( s == src && t == tgt && weight == w )
            TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   return counter.getElement( 0 ) > 0;
}

}  // namespace

template< typename GraphType >
void
test_materialize_no_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph( graph );

   auto result = sg.materialize();

   EXPECT_EQ( result.getVertexCount(), 5 );
   // All edges preserved (no filter).
   EXPECT_EQ( result.getEdgeCount(), 5 );
   EXPECT_TRUE( hasEdge( result, IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, materialize_no_filter )
{
   test_materialize_no_filter< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_materialize_vertex_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );

   auto result = sg.materialize();

   // Vertex count preserved (Variant A).
   EXPECT_EQ( result.getVertexCount(), 5 );
   // vertex 2 inactive -> edges touching it are filtered: (0,2) and (2,3).
   // Surviving edges: (0,1), (1,3), (3,4) -> 3.
   EXPECT_EQ( result.getEdgeCount(), 3 );
   EXPECT_TRUE( hasEdge( result, IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, materialize_vertex_filter )
{
   test_materialize_vertex_filter< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_materialize_edge_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      TNL::Graphs::edgeOnly,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );

   auto result = sg.materialize();

   EXPECT_EQ( result.getVertexCount(), 5 );
   // Edges with w<=3: (0,1), (0,2), (1,3) -> 3.
   EXPECT_EQ( result.getEdgeCount(), 3 );
   EXPECT_TRUE( hasEdge( result, IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, materialize_edge_filter )
{
   test_materialize_edge_filter< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_materialize_both_filters()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      },
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );

   auto result = sg.materialize();

   EXPECT_EQ( result.getVertexCount(), 5 );
   // vertex 2 inactive -> edges touching it are filtered: (0,2) and (2,3).
   // Combined with w<=3 filter: surviving edges are (0,1), (1,3) -> 2.
   EXPECT_EQ( result.getEdgeCount(), 2 );
   EXPECT_TRUE( hasEdge( result, IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, materialize_both_filters )
{
   test_materialize_both_filters< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_materialize_masked()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVectorType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   IndexVectorType indexes{ 0, 1, 3, 4 };  // vertex 2 inactive
   auto sg = TNL::Graphs::makeSubGraph( graph, indexes );

   auto result = sg.materialize();

   EXPECT_EQ( result.getVertexCount(), 5 );
   // vertex 2 inactive -> edges touching it are filtered: (0,2) and (2,3).
   // Surviving edges: (0,1), (1,3), (3,4) -> 3.
   EXPECT_EQ( result.getEdgeCount(), 3 );
   EXPECT_TRUE( hasEdge( result, IndexType( 0 ), IndexType( 1 ), ValueType( 1 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 0 ), IndexType( 2 ), ValueType( 2 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 1 ), IndexType( 3 ), ValueType( 3 ) ) );
   EXPECT_FALSE( hasEdge( result, IndexType( 2 ), IndexType( 3 ), ValueType( 4 ) ) );
   EXPECT_TRUE( hasEdge( result, IndexType( 3 ), IndexType( 4 ), ValueType( 5 ) ) );
}

TYPED_TEST( SubGraphTest, materialize_masked )
{
   test_materialize_masked< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_materialize_free_function()
{
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph( graph );
   auto result = TNL::Graphs::materialize( sg );
   static_assert(
      std::is_same_v<
         decltype( result ),
         TNL::Graphs::Graph<
            typename GraphType::ValueType,
            typename GraphType::DeviceType,
            typename GraphType::IndexType,
            TNL::Graphs::DirectedGraph > >,
      "Free-function materialize(SubGraph) must return Graph<...>." );
   EXPECT_EQ( result.getVertexCount(), 5 );
   EXPECT_EQ( result.getEdgeCount(), 5 );
}

TYPED_TEST( SubGraphTest, materialize_free_function )
{
   test_materialize_free_function< typename TestFixture::GraphType >();
}
