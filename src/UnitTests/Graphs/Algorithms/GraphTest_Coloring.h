// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

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
expectComputedLubyColoringIsProper( const GraphType& graph )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoringLuby( graph, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_EQ( TNL::min( colors ), 0 );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, colors );
}

template< typename GraphType, typename VertexIndexes >
void
expectComputedLubyColoringIsProper( const GraphType& graph, const VertexIndexes& vertexIndexes )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoringLuby( graph, vertexIndexes, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, vertexIndexes, colors );
}

template< typename GraphType, typename VertexPredicate >
void
expectComputedLubyColoringIsProperIf( const GraphType& graph, VertexPredicate&& vertexPredicate )
{
   ColoringVector< GraphType > colors;
   TNL::Graphs::Algorithms::graphColoringLubyIf( graph, vertexPredicate, colors );

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

template< typename GraphType >
void
test_isProperlyColoredIf_true_impl()
{
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

TYPED_TEST( GraphTest, test_isProperlyColoredIf_true )
{
   test_isProperlyColoredIf_true_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_isProperlyColoredIf_false_impl()
{
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

TYPED_TEST( GraphTest, test_isProperlyColoredIf_false )
{
   test_isProperlyColoredIf_false_impl< typename TestFixture::GraphType >();
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

template< typename GraphType >
void
test_graphColoringIf_impl()
{
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

TYPED_TEST( GraphTest, test_graphColoringIf )
{
   test_graphColoringIf_impl< typename TestFixture::GraphType >();
}

TYPED_TEST( GraphTest, test_graphColoringLuby_result_isProperlyColored )
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

   TNL::Graphs::Algorithms::graphColoringLuby( graph, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, colors ) );
   expectZeroColorClassIsMaximalIndependentSet( graph, colors );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_withIndexes )
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

   TNL::Graphs::Algorithms::graphColoringLuby( graph, vertexIndexes, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, vertexIndexes, colors ) );
   EXPECT_EQ( colors.getElement( 0 ), -1 );
   EXPECT_EQ( colors.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_star_usesAtMostTwoColors )
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

   TNL::Graphs::Algorithms::graphColoringLuby( graph, colors );

   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_withIndexes_usesAtMostTwoColors )
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

   TNL::Graphs::Algorithms::graphColoringLuby( graph, vertexIndexes, colors );

   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

template< typename GraphType >
void
test_graphColoringLubyIf_impl()
{
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

   TNL::Graphs::Algorithms::graphColoringLubyIf( graph, middleVertices, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, middleVertices, colors ) );
   EXPECT_EQ( colors.getElement( 0 ), -1 );
   EXPECT_EQ( colors.getElement( 4 ), -1 );
}

TYPED_TEST( GraphTest, test_graphColoringLubyIf )
{
   test_graphColoringLubyIf_impl< typename TestFixture::GraphType >();
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

TYPED_TEST( GraphTest, test_graphColoringLuby_star )
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

   expectComputedLubyColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_chain )
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

   expectComputedLubyColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_medium )
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

   expectComputedLubyColoringIsProper( graph );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_large )
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

   expectComputedLubyColoringIsProper( graph );
}

template< typename GraphType, typename EdgePredicate >
void
expectComputedColoringWithEdgePredicateIsProper( const GraphType& graph, EdgePredicate&& edgePredicate )
{
   using ColorsType = ColoringVector< GraphType >;

   ColorsType colors;
   TNL::Graphs::Algorithms::graphColoring( graph, edgePredicate, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, edgePredicate, colors ) );
}

template< typename GraphType, typename EdgePredicate >
void
expectComputedLubyColoringWithEdgePredicateIsProper( const GraphType& graph, EdgePredicate&& edgePredicate )
{
   using ColorsType = ColoringVector< GraphType >;

   ColorsType colors;
   TNL::Graphs::Algorithms::graphColoringLuby( graph, edgePredicate, colors );

   EXPECT_EQ( colors.getSize(), graph.getVertexCount() );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, edgePredicate, colors ) );
}

template< typename GraphType >
void
test_graphColoring_edge_predicate_weight_threshold_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ColorsType = ColoringVector< GraphType >;

   // clang-format off
   // Chain with varying weights: 0--(1)--1--(2)--2--(1)--3--(3)--4
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 2.0 },
         { 2, 3, 1.0 },
         { 3, 4, 3.0 },
      } );
   // clang-format on

   // Allow only edges with weight <= 1.0 -> usable edges: (0,1), (2,3).
   // Vertices 0 and 1 conflict; vertices 2 and 3 conflict; vertex 4 is isolated.
   // So we need at most 2 colors (alternating 0-1 and 2-3, and 4 can be color 0).
   ColorsType colors;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight <= 1.0;
   };

   TNL::Graphs::Algorithms::graphColoring( graph, edgePredicate, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, edgePredicate, colors ) );
   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

TYPED_TEST( GraphTest, test_graphColoring_edge_predicate_weight_threshold )
{
   test_graphColoring_edge_predicate_weight_threshold_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoring_edge_predicate_block_all_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
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

   // Block all edges -> no conflicts, 1 color suffices.
   ColorsType colors;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return false;
   };

   TNL::Graphs::Algorithms::graphColoring( graph, edgePredicate, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, edgePredicate, colors ) );
   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 1 ) );
}

TYPED_TEST( GraphTest, test_graphColoring_edge_predicate_block_all )
{
   test_graphColoring_edge_predicate_block_all_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoring_edge_predicate_identity_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;

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

   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return true;
   };

   expectComputedColoringWithEdgePredicateIsProper( graph, edgePredicate );
}

TYPED_TEST( GraphTest, test_graphColoring_edge_predicate_identity )
{
   test_graphColoring_edge_predicate_identity_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoring_vertex_and_edge_predicate_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ColorsType = ColoringVector< GraphType >;

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
   // Edge predicate: allow weight <= 1.0 -> usable edges: (0,1), (2,3).
   ColorsType colors;
   auto vertexPredicate = [] __cuda_callable__( IndexType v )
   {
      return v < 5;
   };
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight <= 1.0;
   };

   TNL::Graphs::Algorithms::graphColoringIf( graph, vertexPredicate, edgePredicate, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graph, vertexPredicate, edgePredicate, colors ) );
   EXPECT_EQ( colors.getElement( 5 ), -1 );
}

TYPED_TEST( GraphTest, test_graphColoring_vertex_and_edge_predicate )
{
   test_graphColoring_vertex_and_edge_predicate_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoringLuby_edge_predicate_block_all_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
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

   // Block all edges -> no conflicts, 1 color suffices.
   ColorsType colors;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType )
   {
      return false;
   };

   TNL::Graphs::Algorithms::graphColoringLuby( graph, edgePredicate, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, edgePredicate, colors ) );
   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 1 ) );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_edge_predicate_block_all )
{
   test_graphColoringLuby_edge_predicate_block_all_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoringLuby_edge_predicate_weight_threshold_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ColorsType = ColoringVector< GraphType >;

   // clang-format off
   // Chain with varying weights: 0--(1)--1--(2)--2--(1)--3--(3)--4
   const GraphType graph = makeUndirectedGraph< GraphType >(
      5,
      {
         { 0, 1, 1.0 },
         { 1, 2, 2.0 },
         { 2, 3, 1.0 },
         { 3, 4, 3.0 },
      } );
   // clang-format on

   ColorsType colors;
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight <= 1.0;
   };

   TNL::Graphs::Algorithms::graphColoringLuby( graph, edgePredicate, colors );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, edgePredicate, colors ) );
   expectColorCountAtMost( colors, static_cast< typename ColorsType::ValueType >( 2 ) );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_edge_predicate_weight_threshold )
{
   test_graphColoringLuby_edge_predicate_weight_threshold_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_isProperlyColored_edge_predicate_blocked_edge_no_conflict_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ColorsType = ColoringVector< GraphType >;

   // clang-format off
   const GraphType graph = makeUndirectedGraph< GraphType >(
      3,
      {
         { 0, 1, 2.0 },
         { 1, 2, 1.0 },
      } );
   // clang-format on

   // Both vertices 0 and 1 have the same color, but the edge (0,1) has weight 2.0.
   // If we block weight >= 2.0, this coloring should be proper.
   const ColorsType colors( { 0, 0, 1 } );
   auto edgePredicate = [] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < 2.0;
   };

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graph, edgePredicate, colors ) );
}

TYPED_TEST( GraphTest, test_isProperlyColored_edge_predicate_blocked_edge_no_conflict )
{
   test_isProperlyColored_edge_predicate_blocked_edge_no_conflict_impl< typename TestFixture::GraphType >();
}

// clang-format off
// 10 vertices, same topology as graph A for BFS/SSSP/CC.
// Edges with weight 2 are "expensive".
//
//     0---1---2
//     |   |   |
//     3---4---5
//     |   |   |
//     6---7---8---9
//
// Weight-2 edges: 1-4, 4-5.  All others have weight 1.
// clang-format on

template< typename GraphType >
GraphType
makeColoringGraphA()
{
   using Real = typename GraphType::ValueType;
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
         { 8, 9, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
GraphType
makeColoringSubgraphB()
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
makeColoringSubgraphD()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,2,3,5,6,7,8,9} -> remapped to {0,1,2,3,4,5,6,7,8}
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
makeColoringSubgraphC()
{
   using Real = typename GraphType::ValueType;
   // All 10 vertices, edges with weight >= 2 removed.
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
makeColoringSubgraphE2()
{
   using Real = typename GraphType::ValueType;
   // Vertices {0,1,3,4,6,7}, edges with weight >= 2 also removed.
   // Remap: 0->0, 1->1, 3->2, 4->3, 6->4, 7->5
   // clang-format off
   return GraphType(
      6,
      {
         { 0, 1, Real( 1 ) }, { 0, 2, Real( 1 ) },
         { 2, 3, Real( 1 ) }, { 2, 4, Real( 1 ) },
         { 3, 5, Real( 1 ) },
         { 4, 5, Real( 1 ) },
      },
      TNL::Matrices::MatrixElementsEncoding::SymmetricMixed );
   // clang-format on
}

template< typename GraphType >
void
test_graphColoring_subgraph_vertex_removal_predicate_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

   const auto graphA = makeColoringGraphA< GraphType >();
   const auto subgraphB = makeColoringSubgraphB< GraphType >();

   const auto excludeVertices = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };

   ColorsType colorsA, colorsB;
   TNL::Graphs::Algorithms::graphColoringIf( graphA, excludeVertices, colorsA );
   TNL::Graphs::Algorithms::graphColoring( subgraphB, colorsB );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graphA, excludeVertices, colorsA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( subgraphB, colorsB ) );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   for( int i = 0; i < (int) newToOld.size(); i++ )
      ASSERT_NE( colorsA.getElement( newToOld[ i ] ), -1 ) << "vertex " << newToOld[ i ];
   EXPECT_EQ( getColorCount( colorsA ), getColorCount( colorsB ) );
}

TYPED_TEST( GraphTest, test_graphColoring_subgraph_vertex_removal_predicate )
{
   test_graphColoring_subgraph_vertex_removal_predicate_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoringLuby_subgraph_vertex_removal_predicate_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

   const auto graphA = makeColoringGraphA< GraphType >();
   const auto subgraphB = makeColoringSubgraphB< GraphType >();

   const auto excludeVertices = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 2 && v != 5 && v != 8;
   };

   ColorsType colorsA, colorsB;
   TNL::Graphs::Algorithms::graphColoringLubyIf( graphA, excludeVertices, colorsA );
   TNL::Graphs::Algorithms::graphColoringLuby( subgraphB, colorsB );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graphA, excludeVertices, colorsA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( subgraphB, colorsB ) );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   for( int i = 0; i < (int) newToOld.size(); i++ )
      ASSERT_NE( colorsA.getElement( newToOld[ i ] ), -1 ) << "vertex " << newToOld[ i ];
   EXPECT_EQ( getColorCount( colorsA ), getColorCount( colorsB ) );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_subgraph_vertex_removal_predicate )
{
   test_graphColoringLuby_subgraph_vertex_removal_predicate_impl< typename TestFixture::GraphType >();
}

TYPED_TEST( GraphTest, test_graphColoring_subgraph_vertex_removal_indexed )
{
   using GraphType = typename TestFixture::GraphType;
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

   const auto graphA = makeColoringGraphA< GraphType >();
   const auto subgraphB = makeColoringSubgraphB< GraphType >();

   const ColorsType vertexIndexes( { 0, 1, 3, 4, 6, 7, 9 } );

   ColorsType colorsA, colorsB;
   TNL::Graphs::Algorithms::graphColoring( graphA, vertexIndexes, colorsA );
   TNL::Graphs::Algorithms::graphColoring( subgraphB, colorsB );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graphA, vertexIndexes, colorsA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( subgraphB, colorsB ) );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7, 9 };
   for( int i = 0; i < (int) newToOld.size(); i++ )
      ASSERT_NE( colorsA.getElement( newToOld[ i ] ), -1 ) << "vertex " << newToOld[ i ];
   EXPECT_EQ( getColorCount( colorsA ), getColorCount( colorsB ) );
}

template< typename GraphType >
void
test_graphColoring_subgraph_vertex_removal_disconnected_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ColorsType = ColoringVector< GraphType >;

   const auto graphA = makeColoringGraphA< GraphType >();
   const auto subgraphD = makeColoringSubgraphD< GraphType >();

   const auto excludeFour = [ = ] __cuda_callable__( IndexType v )
   {
      return v != 4;
   };

   ColorsType colorsA, colorsD;
   TNL::Graphs::Algorithms::graphColoringIf( graphA, excludeFour, colorsA );
   TNL::Graphs::Algorithms::graphColoring( subgraphD, colorsD );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColoredIf( graphA, excludeFour, colorsA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( subgraphD, colorsD ) );

   const std::vector< int > newToOld = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
   for( int i = 0; i < (int) newToOld.size(); i++ )
      ASSERT_NE( colorsA.getElement( newToOld[ i ] ), -1 ) << "vertex " << newToOld[ i ];
   EXPECT_EQ( getColorCount( colorsA ), getColorCount( colorsD ) );
}

TYPED_TEST( GraphTest, test_graphColoring_subgraph_vertex_removal_disconnected )
{
   test_graphColoring_subgraph_vertex_removal_disconnected_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoring_subgraph_edge_removal_wholeGraph_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ColorsType = ColoringVector< GraphType >;

   const auto graphA = makeColoringGraphA< GraphType >();
   const auto subgraphC = makeColoringSubgraphC< GraphType >();

   const auto blockWeight2 = [ = ] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < ValueType( 2 );
   };

   ColorsType colorsA, colorsC;
   TNL::Graphs::Algorithms::graphColoring( graphA, blockWeight2, colorsA );
   TNL::Graphs::Algorithms::graphColoring( subgraphC, colorsC );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graphA, blockWeight2, colorsA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( subgraphC, colorsC ) );
}

TYPED_TEST( GraphTest, test_graphColoring_subgraph_edge_removal_wholeGraph )
{
   test_graphColoring_subgraph_edge_removal_wholeGraph_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoringLuby_subgraph_edge_removal_wholeGraph_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ColorsType = ColoringVector< GraphType >;

   const auto graphA = makeColoringGraphA< GraphType >();
   const auto subgraphC = makeColoringSubgraphC< GraphType >();

   const auto blockWeight2 = [ = ] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < ValueType( 2 );
   };

   ColorsType colorsA, colorsC;
   TNL::Graphs::Algorithms::graphColoringLuby( graphA, blockWeight2, colorsA );
   TNL::Graphs::Algorithms::graphColoringLuby( subgraphC, colorsC );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graphA, blockWeight2, colorsA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( subgraphC, colorsC ) );

   EXPECT_EQ( getColorCount( colorsA ), getColorCount( colorsC ) );
}

TYPED_TEST( GraphTest, test_graphColoringLuby_subgraph_edge_removal_wholeGraph )
{
   test_graphColoringLuby_subgraph_edge_removal_wholeGraph_impl< typename TestFixture::GraphType >();
}

template< typename GraphType >
void
test_graphColoring_subgraph_edge_removal_withIndexes_impl()
{
   using IndexType = typename GraphType::IndexType;
   using ValueType = typename GraphType::ValueType;
   using ColorsType = ColoringVector< GraphType >;

   const auto graphA = makeColoringGraphA< GraphType >();
   const auto subgraphE2 = makeColoringSubgraphE2< GraphType >();

   const ColorsType vertexIndexes( { 0, 1, 3, 4, 6, 7 } );
   const auto blockWeight2 = [ = ] __cuda_callable__( IndexType, IndexType, ValueType weight )
   {
      return weight < ValueType( 2 );
   };

   ColorsType colorsA, colorsE2;
   TNL::Graphs::Algorithms::graphColoring( graphA, vertexIndexes, blockWeight2, colorsA );
   TNL::Graphs::Algorithms::graphColoring( subgraphE2, colorsE2 );

   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( graphA, vertexIndexes, blockWeight2, colorsA ) );
   EXPECT_TRUE( TNL::Graphs::Algorithms::isProperlyColored( subgraphE2, colorsE2 ) );

   const std::vector< int > newToOld = { 0, 1, 3, 4, 6, 7 };
   for( int i = 0; i < (int) newToOld.size(); i++ )
      ASSERT_NE( colorsA.getElement( newToOld[ i ] ), -1 ) << "vertex " << newToOld[ i ];
}

TYPED_TEST( GraphTest, test_graphColoring_subgraph_edge_removal_withIndexes )
{
   test_graphColoring_subgraph_edge_removal_withIndexes_impl< typename TestFixture::GraphType >();
}

#include "../../main.h"
