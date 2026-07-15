// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SubGraphTestBase.h"
#include <TNL/Graphs/traverse.h>

// Original graph (see SubGraphTestBase.h):
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4

TYPED_TEST_SUITE( SubGraphTest, SubGraphTestTypes );

// SubGraph (no filter): identical to the original — all 5 edges traversed.
template< typename GraphType >
void
test_forAllEdges_no_filter()
{
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using CounterVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph( graph );
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   EXPECT_EQ( counter.getElement( 0 ), 5 );
}

TYPED_TEST( SubGraphTest, forAllEdges_no_filter )
{
   test_forAllEdges_no_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Vertex filter `v != 2`: vertex 2 is inactive → edges touching it are filtered.
// Traversed edges: (0,1), (1,3), (3,4) -> 3  (edge (0,2) filtered: target 2 inactive).
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    X          3 --(5)--> 4
template< typename GraphType >
void
test_forAllEdges_vertex_filter()
{
   using IndexType = typename GraphType::IndexType;
   using DeviceType = typename GraphType::DeviceType;
   using CounterVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      [] __cuda_callable__( IndexType v )
      {
         return v != 2;
      } );
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // vertex 2 inactive -> edges touching it are filtered: (0,2) and (2,3) -> 3 edges remain
   EXPECT_EQ( counter.getElement( 0 ), 3 );
}

TYPED_TEST( SubGraphTest, forAllEdges_vertex_filter )
{
   test_forAllEdges_vertex_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Edge filter `w <= 3`: edges (2,3,w=4) and (3,4,w=5) are skipped.
// Traversed edges: (0,1), (0,2), (1,3) -> 3.
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2          3          4
template< typename GraphType >
void
test_forAllEdges_edge_filter()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using CounterVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      TNL::Graphs::edgeOnly,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 3;
      } );
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // edges with w>3 skipped: (2,3,w=4) and (3,4,w=5) -> 3 edges remain
   EXPECT_EQ( counter.getElement( 0 ), 3 );
}

TYPED_TEST( SubGraphTest, forAllEdges_edge_filter )
{
   test_forAllEdges_edge_filter< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Both filters: vertex filter `v != 2` AND edge filter `w <= 3`.
// Vertex 2 inactive + edges with w>3 filtered.
// Traversed edges: (0,1), (1,3) -> 2  (edge (0,2) filtered: target 2 inactive).
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    X          3          4
template< typename GraphType >
void
test_forAllEdges_both_filters()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using CounterVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
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
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // vertex 2 inactive + w>3 filtered -> edges: (0,1),(1,3) -> 2
   // Edge (0,2,w=2) is filtered because target 2 is inactive.
   EXPECT_EQ( counter.getElement( 0 ), 2 );
}

TYPED_TEST( SubGraphTest, forAllEdges_both_filters )
{
   test_forAllEdges_both_filters< typename TestFixture::GraphType >();
}

// Original graph with both filters (vertex `v != 2`, edge `w <= 3`):
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    X          3          4
//
// User condition `v <= 1` restricts to source vertices 0 and 1:
//    0 --(1)--> 1
//    |
//   (2)
//    |
//    v
//    X
//
// Combined: vertex filter v!=2 AND edge filter w<=3 AND user condition v<=1
// -> edges: (0,1,w=1), (1,3,w=3) -> 2  (edge (0,2) filtered: target 2 inactive)
template< typename GraphType >
void
test_forAllEdgesIf()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using CounterVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
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
   CounterVector counter( 1, 0 );
   auto counterView = counter.getView();
   // user condition: only process source vertices 0 and 1
   TNL::Graphs::forAllEdgesIf(
      sg,
      [] __cuda_callable__( IndexType v )
      {
         return v <= 1;
      },
      [ = ] __cuda_callable__( IndexType, IndexType, IndexType, int ) mutable
      {
         TNL::Algorithms::AtomicOperations< DeviceType >::add( counterView[ 0 ], 1 );
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   // user condition v<=1 AND vertex filter v!=2 AND edge filter w<=3
   // -> edges: (0,1,w=1),(1,3,w=3) -> 2
   // Edge (0,2,w=2) is filtered because target 2 is inactive.
   EXPECT_EQ( counter.getElement( 0 ), 2 );
}

TYPED_TEST( SubGraphTest, forAllEdgesIf )
{
   test_forAllEdgesIf< typename TestFixture::GraphType >();
}

// Original graph:
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Edge filter `w <= 2`: only edges (0,1,w=1) and (0,2,w=2) survive.
// Collecting their target vertices into `targets[tgt] = src`:
//    targets[1] = 0, targets[2] = 0, targets[3] = -1, targets[4] = -1.
//
//    0 --(1)--> 1
//    |
//   (2)
//    |
//    v
//    2          3          4
template< typename GraphType >
void
test_forAllEdges_collect_targets()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexVector = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;
   const auto graph = makeTestGraph< GraphType >();
   auto sg = TNL::Graphs::makeSubGraph(
      graph,
      TNL::Graphs::edgeOnly,
      [] __cuda_callable__( IndexType, IndexType, ValueType w )
      {
         return w <= 2;
      } );
   // Collect target vertices of edges with w<=2: (0,1) and (0,2)
   IndexVector targets( 5, -1 );
   auto targetsView = targets.getView();
   TNL::Graphs::forAllEdges(
      sg,
      [ = ] __cuda_callable__( IndexType src, IndexType, IndexType tgt, int ) mutable
      {
         targetsView[ tgt ] = src;
      },
      TNL::Algorithms::Segments::LaunchConfiguration{} );
   EXPECT_EQ( targets.getElement( 1 ), 0 );
   EXPECT_EQ( targets.getElement( 2 ), 0 );
   EXPECT_EQ( targets.getElement( 3 ), -1 );
   EXPECT_EQ( targets.getElement( 4 ), -1 );
}

TYPED_TEST( SubGraphTest, forAllEdges_collect_targets )
{
   test_forAllEdges_collect_targets< typename TestFixture::GraphType >();
}
