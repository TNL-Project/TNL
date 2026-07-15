// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SubGraphTestBase.h"
#include <TNL/Graphs/reduce.h>

// Original graph (see SubGraphTestBase.h):
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Each test sums (or max-reduces) edge weights per vertex. Expected results
// are documented per test below.

TYPED_TEST_SUITE( SubGraphTest, SubGraphTestTypes );

// SubGraph (no filter): sum of outgoing edge weights per vertex.
//    v0: 1+2=3, v1: 3, v2: 4, v3: 5, v4: 0 (no outgoing edges)
template< typename GraphType >
void
test_reduceAllVertices_no_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using ResultVector = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph( graph );
   ResultVector result( 5, 0 );
   auto resultView = result.getView();
   TNL::Graphs::reduceAllVertices(
      sg,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType row, ValueType val ) mutable
      {
         resultView[ row ] = val;
      },
      0,
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // v0: 1+2=3, v1: 3, v2: 4, v3: 5, v4: 0 (no outgoing edges)
   EXPECT_EQ( result.getElement( 0 ), 3 );
   EXPECT_EQ( result.getElement( 1 ), 3 );
   EXPECT_EQ( result.getElement( 2 ), 4 );
   EXPECT_EQ( result.getElement( 3 ), 5 );
   EXPECT_EQ( result.getElement( 4 ), 0 );
}

TYPED_TEST( SubGraphTest, reduceAllVertices_no_filter )
{
   test_reduceAllVertices_no_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Vertex filter `v != 2`: vertex 2 inactive → its row returns identity (0).
//    v0: 3, v1: 3, v2: 0 (inactive), v3: 5, v4: 0
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    X          3 --(5)--> 4
template< typename GraphType >
void
test_reduceAllVertices_vertex_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using ResultVector = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );
   ResultVector result( 5, -1 );
   auto resultView = result.getView();
   TNL::Graphs::reduceAllVertices(
      sg,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType row, ValueType val ) mutable
      {
         resultView[ row ] = val;
      },
      0,
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // v0: 1 (edge (0,2) filtered: target 2 inactive), v1: 3, v2: 0 (inactive, identity), v3: 5, v4: 0
   EXPECT_EQ( result.getElement( 0 ), 1 );
   EXPECT_EQ( result.getElement( 1 ), 3 );
   EXPECT_EQ( result.getElement( 2 ), 0 );
   EXPECT_EQ( result.getElement( 3 ), 5 );
   EXPECT_EQ( result.getElement( 4 ), 0 );
}

TYPED_TEST( SubGraphTest, reduceAllVertices_vertex_filter )
{
   test_reduceAllVertices_vertex_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Edge filter `w <= 3`: edges (2,3,w=4) and (3,4,w=5) filtered out.
//    v0: 1+2=3, v1: 3, v2: 0 (edge filtered), v3: 0 (edge filtered), v4: 0
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2          3          4
template< typename GraphType >
void
test_reduceAllVertices_edge_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using ResultVector = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      TNL::Graphs::edgeOnly,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );
   ResultVector result( 5, -1 );
   auto resultView = result.getView();
   TNL::Graphs::reduceAllVertices(
      sg,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType row, ValueType val ) mutable
      {
         resultView[ row ] = val;
      },
      0,
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // v0: 1+2=3, v1: 3, v2: 0 (edge w=4 filtered), v3: 0 (edge w=5 filtered), v4: 0
   EXPECT_EQ( result.getElement( 0 ), 3 );
   EXPECT_EQ( result.getElement( 1 ), 3 );
   EXPECT_EQ( result.getElement( 2 ), 0 );
   EXPECT_EQ( result.getElement( 3 ), 0 );
   EXPECT_EQ( result.getElement( 4 ), 0 );
}

TYPED_TEST( SubGraphTest, reduceAllVertices_edge_filter )
{
   test_reduceAllVertices_edge_filter< typename TestFixture::GraphType >();
}

// SubGraph (no filter): max of outgoing edge weights per vertex.
//    v0: max(1,2)=2, v1: 3, v2: 4, v3: 5, v4: identity (no edges)
// Vertex 4 has no outgoing edges → Max::getIdentity<int>() = INT_MIN.
template< typename GraphType >
void
test_reduceAllVertices_max_edge_weight()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using ResultVector = TNL::Containers::Vector< ValueType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph( graph );
   ResultVector result( 5, 0 );
   auto resultView = result.getView();
   TNL::Graphs::reduceAllVertices(
      sg,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w;
      },
      TNL::Max{},
      [ = ] __cuda_callable__( IndexType row, ValueType val ) mutable
      {
         resultView[ row ] = val;
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // v0: max(1,2)=2, v1: 3, v2: 4, v3: 5, v4: identity (no edges)
   EXPECT_EQ( result.getElement( 0 ), 2 );
   EXPECT_EQ( result.getElement( 1 ), 3 );
   EXPECT_EQ( result.getElement( 2 ), 4 );
   EXPECT_EQ( result.getElement( 3 ), 5 );
   // v4 has no edges -> Max::getIdentity<int>() = numeric_limits<int>::min()
   EXPECT_EQ( result.getElement( 4 ), std::numeric_limits< ValueType >::min() );
}

TYPED_TEST( SubGraphTest, reduceAllVertices_max_edge_weight )
{
   test_reduceAllVertices_max_edge_weight< typename TestFixture::GraphType >();
}
