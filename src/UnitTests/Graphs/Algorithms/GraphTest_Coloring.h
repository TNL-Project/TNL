// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#include <TNL/Containers/Vector.h>
#include <TNL/Graphs/Algorithms/graphColoring.h>
#include <TNL/Graphs/Algorithms/maximalIndependentSet.h>
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
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int >
#endif
   >;

TYPED_TEST_SUITE( GraphTest, GraphTestTypes );

template< typename GraphType >
using ColoringVector =
   TNL::Containers::Vector< typename GraphType::IndexType, typename GraphType::DeviceType, typename GraphType::IndexType >;

template< typename GraphType >
using VertexIndexVector = ColoringVector< GraphType >;

template< typename Colors >
auto
getColorCount( const Colors& colors )
{
   using ColorType = typename Colors::ValueType;

   if( colors.getSize() == 0 )
      return static_cast< ColorType >( 0 );

   const ColorType maxColor = TNL::max( colors );
   return maxColor < static_cast< ColorType >( 0 ) ? static_cast< ColorType >( 0 ) : maxColor + 1;
}

template< typename Colors >
void
expectColorCountAtMost( const Colors& colors, const typename Colors::ValueType maxColors )
{
   EXPECT_LE( getColorCount( colors ), maxColors );
}

template< typename GraphType >
void
expectZeroColorClassIsMaximalIndependentSet( const GraphType& graph, const ColoringVector< GraphType >& colors )
{
   using IndexType = typename GraphType::IndexType;

   ColoringVector< GraphType > zeroColorClass( colors.getSize() );
   zeroColorClass = TNL::equalTo( colors, static_cast< IndexType >( 0 ) );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, zeroColorClass ) );
}

template< typename GraphType, typename VertexIndexes >
void
expectZeroColorClassIsMaximalIndependentSet(
   const GraphType& graph,
   const VertexIndexes& vertexIndexes,
   const ColoringVector< GraphType >& colors )
{
   using IndexType = typename GraphType::IndexType;

   ColoringVector< GraphType > zeroColorClass( colors.getSize() );
   zeroColorClass = TNL::equalTo( colors, static_cast< IndexType >( 0 ) );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSet( graph, vertexIndexes, zeroColorClass ) );
}

template< typename GraphType, typename VertexPredicate >
void
expectZeroColorClassIsMaximalIndependentSetIf(
   const GraphType& graph,
   VertexPredicate&& vertexPredicate,
   const ColoringVector< GraphType >& colors )
{
   using IndexType = typename GraphType::IndexType;

   ColoringVector< GraphType > zeroColorClass( colors.getSize() );
   zeroColorClass = TNL::equalTo( colors, static_cast< IndexType >( 0 ) );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isMaximalIndependentSetIf( graph, vertexPredicate, zeroColorClass ) );
}

template< typename GraphType >
GraphType
makeUndirectedGraph(
   typename GraphType::IndexType vertexCount,
   std::initializer_list<
      std::tuple< typename GraphType::IndexType, typename GraphType::IndexType, typename GraphType::ValueType > > edges )
{
   return GraphType( vertexCount, edges, TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
}

template< typename GraphType >
void
expectComputedColoringIsProper( const GraphType& graph )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoring( graph, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_EQ( TNL::min( colors ), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, colors );
}

template< typename GraphType, typename VertexIndexes >
void
expectComputedColoringIsProper( const GraphType& graph, const VertexIndexes& vertexIndexes )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoring( graph, vertexIndexes, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, vertexIndexes, colors );
}

template< typename GraphType, typename VertexPredicate >
void
expectComputedColoringIsProperIf( const GraphType& graph, VertexPredicate&& vertexPredicate )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoringIf( graph, vertexPredicate, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, vertexPredicate, colors ) );
   expectZeroColorClassIsMaximalIndependentSetIf( graph, vertexPredicate, colors );
}

template< typename GraphType >
void
expectComputedLubiColoringIsProper( const GraphType& graph )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoringLubi( graph, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_EQ( TNL::min( colors ), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, colors );
}

template< typename GraphType, typename VertexIndexes >
void
expectComputedLubiColoringIsProper( const GraphType& graph, const VertexIndexes& vertexIndexes )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoringLubi( graph, vertexIndexes, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, vertexIndexes, colors );
}

template< typename GraphType, typename VertexPredicate >
void
expectComputedLubiColoringIsProperIf( const GraphType& graph, VertexPredicate&& vertexPredicate )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoringLubiIf( graph, vertexPredicate, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, vertexPredicate, colors ) );
   expectZeroColorClassIsMaximalIndependentSetIf( graph, vertexPredicate, colors );
}

TYPED_TEST( GraphTest, test_isProperlyColored_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

   GraphType graph;
   ColorsType colors;

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_star_true )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

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
   const ColorsType colors( { 1, 0, 0, 0, 0 } );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_star_false )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

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
   const ColorsType colors( { 1, 0, 0, 0, 1 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_chain_true )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

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
   const ColorsType colors( { 0, 1, 0, 1, 0 } );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_negative_label_false )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

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
   const ColorsType colors( { 0, 1, -1, 1, 0 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_withIndexes_true )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   const ColorsType colors( { -1, 0, 1, 0, -1 } );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_withIndexes_false )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   const ColorsType colors( { 0, 0, 1, 0, -1 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColoredIf_true )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

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
   const ColorsType colors( { -1, 0, 1, 0, -1 } );
   const auto middleVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex >= 1 && vertex <= 3;
   };

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, middleVertices, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColoredIf_false )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

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
   const ColorsType colors( { -1, 0, 0, 1, -1 } );
   const auto middleVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex >= 1 && vertex <= 3;
   };

   EXPECT_FALSE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, middleVertices, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_medium_false )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      8,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 6, 1.0 },
         { 1, 3, 1.0 }, { 1, 4, 1.0 }, { 2, 5, 1.0 }, { 2, 6, 1.0 },
         { 3, 4, 1.0 }, { 4, 5, 1.0 }, { 4, 7, 1.0 }, { 5, 6, 1.0 },
         { 6, 7, 1.0 },
      } );
   // clang-format on
   const ColorsType colors( { 0, 1, 1, 2, 0, 0, 2, 1 } );

   EXPECT_FALSE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
}

TYPED_TEST( GraphTest, test_graphColoring_empty )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

   GraphType graph;
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoring( graph, colors );

   EXPECT_EQ( colors.getSize(), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
}

TYPED_TEST( GraphTest, test_graphColoring_star )
{
   using GraphType = typename TestFixture::GraphType;

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

   expectComputedColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoring_result_isProperlyColored )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      8,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 },
         { 1, 3, 1.0 }, { 1, 5, 1.0 }, { 1, 6, 1.0 }, { 1, 7, 1.0 },
         { 2, 3, 1.0 }, { 2, 7, 1.0 }, { 3, 4, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 },
      } );
   // clang-format on
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoring( graph, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, colors );
}

TYPED_TEST( GraphTest, test_graphColoring_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoring( graph, vertexIndexes, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
   EXPECT_EQ( colors.getElement( 0 ), -1 );
   EXPECT_EQ( colors.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_graphColoring_star_usesAtMostTwoColors )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

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
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoring( graph, colors );

   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

TYPED_TEST( GraphTest, test_graphColoring_withIndexes_usesAtMostTwoColors )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoring( graph, vertexIndexes, colors );

   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

TYPED_TEST( GraphTest, test_graphColoringIf )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

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
   const auto middleVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex >= 1 && vertex <= 3;
   };
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoringIf( graph, middleVertices, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, middleVertices, colors ) );
   EXPECT_EQ( colors.getElement( 0 ), -1 );
   EXPECT_EQ( colors.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_result_isProperlyColored )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      8,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 },
         { 1, 3, 1.0 }, { 1, 5, 1.0 }, { 1, 6, 1.0 }, { 1, 7, 1.0 },
         { 2, 3, 1.0 }, { 2, 7, 1.0 }, { 3, 4, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 },
      } );
   // clang-format on
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoringLubi( graph, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, colors );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_withIndexes )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoringLubi( graph, vertexIndexes, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
   EXPECT_EQ( colors.getElement( 0 ), -1 );
   EXPECT_EQ( colors.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_star_usesAtMostTwoColors )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;

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
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoringLubi( graph, colors );

   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_withIndexes_usesAtMostTwoColors )
{
   using GraphType = typename TestFixture::GraphType;
   using ColorsType = ColoringVector< GraphType >;
   using VertexIndexVectorType = VertexIndexVector< GraphType >;

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
   const VertexIndexVectorType vertexIndexes( { 1, 2, 3 } );
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoringLubi( graph, vertexIndexes, colors );

   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

TYPED_TEST( GraphTest, test_graphColoringLubiIf )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

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
   const auto middleVertices = [ = ] __cuda_callable__( IndexType vertex )
   {
      return vertex >= 1 && vertex <= 3;
   };
   ColorsType colors;

   TNL::Graphs::Algorithms::graphColoringLubiIf( graph, middleVertices, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, middleVertices, colors ) );
   EXPECT_EQ( colors.getElement( 0 ), -1 );
   EXPECT_EQ( colors.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_graphColoring_chain )
{
   using GraphType = typename TestFixture::GraphType;

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

   expectComputedColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoring_small )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 0, 3, 1.0 },
         { 1, 2, 1.0 },
         { 1, 3, 1.0 },
         { 2, 3, 1.0 },
         { 2, 4, 1.0 },
      } );
   // clang-format on

   expectComputedColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoring_medium )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      8,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 },
         { 1, 3, 1.0 }, { 1, 5, 1.0 }, { 1, 6, 1.0 }, { 1, 7, 1.0 },
         { 2, 3, 1.0 }, { 2, 7, 1.0 }, { 3, 4, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 },
      } );
   // clang-format on

   expectComputedColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoring_large )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      15,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 4, 1.0 }, { 0, 5, 1.0 }, { 0, 7, 1.0 },
         { 1, 3, 1.0 }, { 1, 8, 1.0 }, { 1, 12, 1.0 }, { 2, 5, 1.0 }, { 2, 10, 1.0 },
         { 2, 13, 1.0 }, { 3, 5, 1.0 }, { 3, 12, 1.0 }, { 3, 13, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 }, { 4, 7, 1.0 }, { 4, 11, 1.0 }, { 4, 13, 1.0 }, { 5, 11, 1.0 },
         { 5, 12, 1.0 }, { 5, 14, 1.0 }, { 6, 8, 1.0 }, { 6, 9, 1.0 }, { 6, 10, 1.0 },
         { 6, 12, 1.0 }, { 6, 14, 1.0 }, { 7, 9, 1.0 }, { 8, 10, 1.0 }, { 8, 14, 1.0 },
         { 10, 14, 1.0 }, { 13, 14, 1.0 },
      } );
   // clang-format on

   expectComputedColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_star )
{
   using GraphType = typename TestFixture::GraphType;

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

   expectComputedLubiColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_chain )
{
   using GraphType = typename TestFixture::GraphType;

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

   expectComputedLubiColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_medium )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      8,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 3, 1.0 }, { 0, 4, 1.0 },
         { 1, 3, 1.0 }, { 1, 5, 1.0 }, { 1, 6, 1.0 }, { 1, 7, 1.0 },
         { 2, 3, 1.0 }, { 2, 7, 1.0 }, { 3, 4, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 },
      } );
   // clang-format on

   expectComputedLubiColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoringLubi_large )
{
   using GraphType = typename TestFixture::GraphType;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      15,
      {
         { 0, 1, 1.0 }, { 0, 2, 1.0 }, { 0, 4, 1.0 }, { 0, 5, 1.0 }, { 0, 7, 1.0 },
         { 1, 3, 1.0 }, { 1, 8, 1.0 }, { 1, 12, 1.0 }, { 2, 5, 1.0 }, { 2, 10, 1.0 },
         { 2, 13, 1.0 }, { 3, 5, 1.0 }, { 3, 12, 1.0 }, { 3, 13, 1.0 }, { 4, 5, 1.0 },
         { 4, 6, 1.0 }, { 4, 7, 1.0 }, { 4, 11, 1.0 }, { 4, 13, 1.0 }, { 5, 11, 1.0 },
         { 5, 12, 1.0 }, { 5, 14, 1.0 }, { 6, 8, 1.0 }, { 6, 9, 1.0 }, { 6, 10, 1.0 },
         { 6, 12, 1.0 }, { 6, 14, 1.0 }, { 7, 9, 1.0 }, { 8, 10, 1.0 }, { 8, 14, 1.0 },
         { 10, 14, 1.0 }, { 13, 14, 1.0 },
      } );
   // clang-format on

   expectComputedLubiColoringIsProper( graph );
}

#include "../../main.h"
