#include <TNL/Graphs/Algorithms/connectedComponents.h>
#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

template< typename Matrix >
class GraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::UndirectedGraph >;
};

using GraphTestTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Sequential, int, TNL::Matrices::SymmetricMatrix >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::SymmetricMatrix >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::SymmetricMatrix >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::SymmetricMatrix >
#endif
   >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

template< typename GraphType >
GraphType
makeUndirectedGraph(
   typename GraphType::IndexType vertexCount,
   std::initializer_list<
      std::tuple< typename GraphType::IndexType, typename GraphType::IndexType, typename GraphType::ValueType > > edges )
{
   return GraphType( vertexCount, edges, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
}

TYPED_TEST( GraphTest, test_CC_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   GraphType graph;
   ComponentsType components;

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   EXPECT_EQ( components.getSize(), 0 );
}

TYPED_TEST( GraphTest, test_CC_star )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 0, 2, 1.0 },
         { 0, 3, 1.0 },
         { 0, 4, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 5 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_chain )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 3, 4, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 5 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_chain_alternating )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      6,
      {
         { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 4, 1.0 },
         { 3, 5, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 6 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 0, 1, 0, 1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_small )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      10,
      {
         { 2, 6, 1.0 },
         { 4, 6, 1.0 },
         { 4, 7, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 10 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 2, 3, 2, 5, 2, 2, 8, 9 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_medium )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      15,
      {
         {  0,  4, 1.0 },
         {  0, 10, 1.0 },
         {  0, 13, 1.0 },
         {  2,  5, 1.0 },
         {  2,  6, 1.0 },
         {  2,  8, 1.0 },
         {  3, 10, 1.0 },
         {  3, 13, 1.0 },
         {  4, 12, 1.0 },
         {  5,  7, 1.0 },
         {  8,  9, 1.0 },
         { 12, 14, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 15 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 2, 0, 0, 2, 2, 2, 2, 2, 0, 11, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_medium2 )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      15,
      {
         {  0,  3, 1.0 }, {  0,  4, 1.0 }, {  0, 10, 1.0 }, {  2,  4, 1.0 }, {  2,  6, 1.0 },
         {  3,  4, 1.0 }, {  3,  5, 1.0 }, {  3,  9, 1.0 }, {  3, 13, 1.0 }, {  4, 11, 1.0 },
         {  5, 11, 1.0 }, {  5, 14, 1.0 }, {  8, 12, 1.0 }, {  9, 14, 1.0 }, { 10, 12, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 15 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected( { 0, 1, 0, 0, 0, 0, 0, 7, 0, 0, 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_large )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      30,
      {
         {  1,  3, 1.0 }, {  1,  8, 1.0 }, {  2, 10, 1.0 }, {  3,  6, 1.0 }, {  3,  8, 1.0 }, {  5, 21, 1.0 },
         {  6, 11, 1.0 }, {  6, 14, 1.0 }, {  6, 22, 1.0 }, {  6, 23, 1.0 }, {  6, 25, 1.0 }, {  8, 11, 1.0 },
         {  8, 25, 1.0 }, {  8, 28, 1.0 }, { 10, 25, 1.0 }, { 11, 28, 1.0 }, { 12, 20, 1.0 }, { 14, 17, 1.0 },
         { 15, 24, 1.0 }, { 16, 29, 1.0 }, { 17, 21, 1.0 }, { 18, 19, 1.0 }, { 19, 26, 1.0 }, { 20, 24, 1.0 },
         { 21, 28, 1.0 }, { 24, 29, 1.0 },
      } );
   // clang-format on
   ComponentsType components( 30 );

   TNL::Graphs::Algorithms::connectedComponents( graph, components );

   ComponentsType expected(
      { 0, 1, 1, 1, 4, 1, 1, 7, 1, 9, 1, 1, 12, 13, 1, 12, 12, 1, 18, 18, 12, 1, 1, 1, 12, 1, 18, 27, 1, 12 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_indexed_small )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      10,
      {
         { 2, 6, 1.0 },
         { 4, 6, 1.0 },
         { 4, 7, 1.0 },
      } );
   // clang-format on

   // Induced subgraph on {2, 4, 6, 7} -- the component {2,4,6,7} stays connected.
   ComponentsType vertexIndexes( { 2, 4, 6, 7 } );
   ComponentsType components;

   TNL::Graphs::Algorithms::connectedComponents( graph, vertexIndexes, components );

   ComponentsType expected( { -1, -1, 2, -1, 2, -1, 2, 2, -1, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_indexed_alternating )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      6,
      {
         { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 4, 1.0 },
         { 3, 5, 1.0 },
      } );
   // clang-format on

   // Select only even vertices: {0, 2, 4} -- they form one connected component in the induced subgraph.
   ComponentsType vertexIndexes( { 0, 2, 4 } );
   ComponentsType components;

   TNL::Graphs::Algorithms::connectedComponents( graph, vertexIndexes, components );

   ComponentsType expected( { 0, -1, 0, -1, 0, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_indexed_all_vertices )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 3, 4, 1.0 },
      } );
   // clang-format on

   // All vertices active -> same as whole-graph CC.
   ComponentsType vertexIndexes( { 0, 1, 2, 3, 4 } );
   ComponentsType components;

   TNL::Graphs::Algorithms::connectedComponents( graph, vertexIndexes, components );

   ComponentsType expected( { 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_predicate_small )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      10,
      {
         { 2, 6, 1.0 },
         { 4, 6, 1.0 },
         { 4, 7, 1.0 },
      } );
   // clang-format on

   // Predicate: select vertices >= 2 and <= 7 -> {2,3,4,5,6,7}
   // In the induced subgraph, {2,4,6,7} is one component; {3} and {5} are isolated.
   ComponentsType components;
   auto predicate = [] __cuda_callable__( IndexType v )
   {
      return v >= 2 && v <= 7;
   };

   TNL::Graphs::Algorithms::connectedComponentsIf( graph, predicate, components );

   ComponentsType expected( { -1, -1, 2, 3, 2, 5, 2, 2, -1, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_predicate_even_vertices )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      6,
      {
         { 0, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 4, 1.0 },
         { 3, 5, 1.0 },
      } );
   // clang-format on

   // Predicate: even vertices -> {0, 2, 4}, all connected via edges (0,2) and (2,4).
   ComponentsType components;
   auto predicate = [] __cuda_callable__( IndexType v )
   {
      return v % 2 == 0;
   };

   TNL::Graphs::Algorithms::connectedComponentsIf( graph, predicate, components );

   ComponentsType expected( { 0, -1, 0, -1, 0, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_predicate_none_active )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
      } );
   // clang-format on

   // No vertex is active -> all should get -1.
   ComponentsType components;
   auto predicate = [] __cuda_callable__( IndexType )
   {
      return false;
   };

   TNL::Graphs::Algorithms::connectedComponentsIf( graph, predicate, components );

   ComponentsType expected( { -1, -1, -1, -1, -1 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_edge_predicate_weight_threshold )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   // Chain: 0--(1)--1--(2)--2--(3)--3--(4)--4
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 2.0 },
         { 2, 3, 3.0 },
         { 3, 4, 4.0 },
      } );
   // clang-format on

   // Allow only edges with weight <= 2.0 -> edges (0,1) and (1,2) are usable,
   // so {0,1,2} form one component, {3} and {4} are singletons.
   ComponentsType components;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight <= 2.0;
   };

   TNL::Graphs::Algorithms::connectedComponents( graph, edgePredicate, components );

   ComponentsType expected( { 0, 0, 0, 3, 4 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_edge_predicate_block_all )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 3, 4, 1.0 },
      } );
   // clang-format on

   // Block all edges -> every vertex is its own component.
   ComponentsType components;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return false;
   };

   TNL::Graphs::Algorithms::connectedComponents( graph, edgePredicate, components );

   ComponentsType expected( { 0, 1, 2, 3, 4 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_edge_predicate_identity )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 1.0 },
         { 2, 3, 1.0 },
         { 3, 4, 1.0 },
      } );
   // clang-format on

   // Allow all edges -> same as whole-graph CC.
   ComponentsType components;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return true;
   };

   TNL::Graphs::Algorithms::connectedComponents( graph, edgePredicate, components );

   ComponentsType expected( { 0, 0, 0, 0, 0 } );
   ASSERT_EQ( components, expected );
}

TYPED_TEST( GraphTest, test_CC_vertex_and_edge_predicate )
{
   using GraphType = typename TestFixture::GraphType;
   using DeviceType = typename GraphType::DeviceType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, DeviceType, IndexType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      6,
      {
         { 0, 1, 1.0 },
         { 1, 2, 2.0 },
         { 2, 3, 1.0 },
         { 3, 4, 2.0 },
         { 4, 5, 1.0 },
      } );
   // clang-format on

   // Vertex predicate: active vertices 0..4 (exclude 5).
   // Edge predicate: allow weight <= 1.0 -> edges (0,1), (2,3), (4,5).
   // In the induced subgraph {0,1,2,3,4}: usable edges are (0,1) and (2,3).
   // So components are {0,1}, {2,3}, {4}.
   ComponentsType components;
   auto vertexPredicate = [] __cuda_callable__( IndexType v )
   {
      return v < 5;
   };
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight <= 1.0;
   };

    TNL::Graphs::Algorithms::connectedComponentsIf( graph, vertexPredicate, edgePredicate, components );

    ComponentsType expected( { 0, 0, 2, 2, 4, -1 } );
    ASSERT_EQ( components, expected );
}

template< typename GraphType >
GraphType
makeUndirectedGraphA()
{
   using Real = typename GraphType::ValueType;
   // clang-format off
   // 10 vertices, same topology as graph A for BFS/SSSP.
   // Edges with weight 2 are "expensive" and can be filtered.
   //
   //     0---1---2
   //     |   |   |
   //     3---4---5
   //     |   |   |
   //     6---7---8---9
   //
   // Weight-2 edges: 1-4, 4-5.  All others have weight 1.
   return GraphType(
      10,
      {
         { 0, 1, Real( 1 ) }, { 0, 3, Real( 1 ) },
         { 1, 2, Real( 1 ) }, { 1, 4, Real( 2 ) },
         { 2, 5, Real( 1 ) },
         { 3, 4, Real( 1 ) }, { 3, 6, Real( 1 ) },
         { 4, 5, Real( 2 ) }, { 4, 7, Real( 1 ) },
         { 5, 8, Real( 1 ) },
         { 6, 7, Real( 1 ) },
         { 7, 8, Real( 1 ) },
         { 8, 9, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeUndirectedSubgraphB()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,3,4,6,7,9} -> remapped to {0,1,2,3,4,5,6}
   // clang-format off
   return GraphType(
      7,
      {
         { 0, 1, Real( 1 ) }, { 0, 2, Real( 1 ) },
         { 1, 3, Real( 2 ) },
         { 2, 3, Real( 1 ) }, { 2, 4, Real( 1 ) },
         { 3, 5, Real( 1 ) },
         { 4, 5, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeUndirectedSubgraphD()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,2,3,5,6,7,8,9} -> remapped to {0,1,2,3,4,5,6,7,8}
   // Cut-vertex 4 removed: graph splits into {0,1,2,3} and {5,6,7,8,9}
   // clang-format off
   return GraphType(
      9,
      {
         { 0, 1, Real( 1 ) }, { 0, 3, Real( 1 ) },
         { 1, 2, Real( 1 ) },
         { 2, 4, Real( 1 ) },
         { 3, 5, Real( 1 ) },
         { 4, 7, Real( 1 ) },
         { 5, 6, Real( 1 ) },
         { 6, 7, Real( 1 ) },
         { 7, 8, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeUndirectedSubgraphC()
{
   using Real = typename GraphType::ValueType;
   // All 10 vertices, edges with weight >= 2 removed ({1,4} and {4,5}).
   // clang-format off
   return GraphType(
      10,
      {
         { 0, 1, Real( 1 ) }, { 0, 3, Real( 1 ) },
         { 1, 2, Real( 1 ) },
         { 2, 5, Real( 1 ) },
         { 3, 4, Real( 1 ) }, { 3, 6, Real( 1 ) },
         { 4, 7, Real( 1 ) },
         { 5, 8, Real( 1 ) },
         { 6, 7, Real( 1 ) },
         { 7, 8, Real( 1 ) },
         { 8, 9, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeUndirectedSubgraphE_edgeOnly()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,3,4,6,7}, edges with weight >= 2 also removed.
   // Remap: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5
   // clang-format off
   return GraphType(
      6,
      {
         { 0, 1, Real( 1 ) },
         { 0, 2, Real( 1 ) },
         { 2, 3, Real( 1 ) }, { 2, 4, Real( 1 ) },
         { 3, 5, Real( 1 ) },
         { 4, 5, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeUndirectedSubgraphE_bridge()
{
   using Real = typename GraphType::ValueType;
   // All 10 vertices, bridge edge {8,9} removed.
   // Vertex 9 becomes an isolated component.
   // clang-format off
   return GraphType(
      10,
      {
         { 0, 1, Real( 1 ) }, { 0, 3, Real( 1 ) },
         { 1, 2, Real( 1 ) }, { 1, 4, Real( 2 ) },
         { 2, 5, Real( 1 ) },
         { 3, 4, Real( 1 ) }, { 3, 6, Real( 1 ) },
         { 4, 5, Real( 2 ) }, { 4, 7, Real( 1 ) },
         { 5, 8, Real( 1 ) },
         { 6, 7, Real( 1 ) },
         { 7, 8, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename VectorA, typename VectorB >
void
expectPartitionEquiv(
   const VectorA& compA,
   const VectorB& compB,
   const std::vector< int >& oldToNew,
   int origSize )
{
   for( int u = 0; u < origSize; u++ ) {
      if( oldToNew[ u ] < 0 )
         continue;
      for( int v = u + 1; v < origSize; v++ ) {
         if( oldToNew[ v ] < 0 )
            continue;
         bool sameCompA = compA.getElement( u ) == compA.getElement( v );
         bool sameCompB = compB.getElement( oldToNew[ u ] ) == compB.getElement( oldToNew[ v ] );
         ASSERT_EQ( sameCompA, sameCompB ) << "vertices " << u << " and " << v;
      }
   }
}

TYPED_TEST( GraphTest, test_CC_subgraph_vertex_removal_predicate )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeUndirectedGraphA< GraphType >();
   const auto subgraphB = makeUndirectedSubgraphB< GraphType >();

   const auto excludeVertices = [=] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };

   ComponentsType compA, compB;
   TNL::Graphs::Algorithms::connectedComponentsIf( graphA, excludeVertices, compA );
   TNL::Graphs::Algorithms::connectedComponents( subgraphB, compB );

   // oldToNew: -1 for removed vertices
   const std::vector< int > oldToNew = { 0, 1, -1, 2, 3, -1, 4, 5, -1, 6 };
   expectPartitionEquiv( compA, compB, oldToNew, 10 );
}

TYPED_TEST( GraphTest, test_CC_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeUndirectedGraphA< GraphType >();
   const auto subgraphB = makeUndirectedSubgraphB< GraphType >();

   const ComponentsType vertexIndexes( { 0, 1, 3, 4, 6, 7, 9 } );

   ComponentsType compA, compB;
   TNL::Graphs::Algorithms::connectedComponents( graphA, vertexIndexes, compA );
   TNL::Graphs::Algorithms::connectedComponents( subgraphB, compB );

   const std::vector< int > oldToNew = { 0, 1, -1, 2, 3, -1, 4, 5, -1, 6 };
   expectPartitionEquiv( compA, compB, oldToNew, 10 );
}

TYPED_TEST( GraphTest, test_CC_subgraph_vertex_removal_disconnected )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeUndirectedGraphA< GraphType >();
   const auto subgraphD = makeUndirectedSubgraphD< GraphType >();

   const auto excludeFour = [=] __cuda_callable__( IndexType v )
   {
      return v != 4;
   };

   ComponentsType compA, compD;
   TNL::Graphs::Algorithms::connectedComponentsIf( graphA, excludeFour, compA );
   TNL::Graphs::Algorithms::connectedComponents( subgraphD, compD );

   const std::vector< int > oldToNew = { 0, 1, 2, 3, -1, 4, 5, 6, 7, 8 };
   expectPartitionEquiv( compA, compD, oldToNew, 10 );
}

TYPED_TEST( GraphTest, test_CC_subgraph_edge_removal_wholeGraph )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeUndirectedGraphA< GraphType >();
   const auto subgraphC = makeUndirectedSubgraphC< GraphType >();

   const auto blockWeight2 = [=] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < ValueType( 2 );
   };

   ComponentsType compA, compC;
   TNL::Graphs::Algorithms::connectedComponents( graphA, blockWeight2, compA );
   TNL::Graphs::Algorithms::connectedComponents( subgraphC, compC );

   const std::vector< int > identity = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
   expectPartitionEquiv( compA, compC, identity, 10 );
}

TYPED_TEST( GraphTest, test_CC_subgraph_edge_removal_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeUndirectedGraphA< GraphType >();
   const auto subgraphE2 = makeUndirectedSubgraphE_edgeOnly< GraphType >();

   const ComponentsType vertexIndexes( { 0, 1, 3, 4, 6, 7 } );
   const auto blockWeight2 = [=] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < ValueType( 2 );
   };

   ComponentsType compA, compE2;
   TNL::Graphs::Algorithms::connectedComponents( graphA, vertexIndexes, blockWeight2, compA );
   TNL::Graphs::Algorithms::connectedComponents( subgraphE2, compE2 );

   const std::vector< int > oldToNew = { 0, 1, -1, 2, 3, -1, 4, 5, -1, -1 };
   expectPartitionEquiv( compA, compE2, oldToNew, 10 );
}

TYPED_TEST( GraphTest, test_CC_subgraph_edge_removal_bridge )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ComponentsType = TNL::Containers::Vector< IndexType, typename GraphType::DeviceType, IndexType >;

   const auto graphA = makeUndirectedGraphA< GraphType >();
   const auto subgraphBridge = makeUndirectedSubgraphE_bridge< GraphType >();

   const auto blockEdge89 = [=] __cuda_callable__( IndexType src, IndexType tgt, ValueType )
   {
      return ! ( ( src == 8 && tgt == 9 ) || ( src == 9 && tgt == 8 ) );
   };

   ComponentsType compA, compBridge;
   TNL::Graphs::Algorithms::connectedComponents( graphA, blockEdge89, compA );
   TNL::Graphs::Algorithms::connectedComponents( subgraphBridge, compBridge );

   const std::vector< int > identity = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };
   expectPartitionEquiv( compA, compBridge, identity, 10 );
}

#include "../../main.h"
