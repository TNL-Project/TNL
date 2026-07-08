#pragma once

#include <TNL/Graphs/Algorithms/details/lambdaTraits.hpp>
#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <type_traits>

#include <gtest/gtest.h>

/**
 * \brief Compile-time tests for the centralized lambda-trait metafunctions.
 */
template< typename Matrix >
class LambdaTraitsTest : public ::testing::Test
{
protected:
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::DirectedGraph >;
};

using LambdaTraitsTestTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Sequential, int >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int >
#endif
   >;

TYPED_TEST_SUITE( LambdaTraitsTest, LambdaTraitsTestTypes );

TYPED_TEST( LambdaTraitsTest, isEdgePredicate_accepts_valid_signature )
{
   using Graph = typename TestFixture::GraphType;
   auto good = []( typename Graph::IndexType, typename Graph::IndexType, typename Graph::ValueType ) -> bool
   {
      return true;
   };
   static_assert(
      TNL::Graphs::Algorithms::detail::isEdgePredicate_v< decltype( good ), Graph >,
      "Valid edge predicate (source, target, weight) -> bool must be accepted." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isEdgePredicate_rejects_non_callable )
{
   using Graph = typename TestFixture::GraphType;
   int notCallable = 0;
   static_assert(
      ! TNL::Graphs::Algorithms::detail::isEdgePredicate_v< decltype( notCallable ), Graph >,
      "Non-callable object must be rejected as edge predicate." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isEdgePredicate_rejects_wrong_arity )
{
   using Graph = typename TestFixture::GraphType;
   auto bad = []( typename Graph::IndexType, typename Graph::IndexType ) -> bool
   {
      return true;
   };
   static_assert(
      ! TNL::Graphs::Algorithms::detail::isEdgePredicate_v< decltype( bad ), Graph >,
      "Edge predicate with wrong arity must be rejected." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isVertexPredicate_accepts_valid_signature )
{
   using Graph = typename TestFixture::GraphType;
   auto good = []( typename Graph::IndexType ) -> bool
   {
      return true;
   };
   static_assert(
      TNL::Graphs::Algorithms::detail::isVertexPredicate_v< decltype( good ), Graph >,
      "Valid vertex predicate (vertex) -> bool must be accepted." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isVertexPredicate_rejects_wrong_arity )
{
   using Graph = typename TestFixture::GraphType;
   auto bad = []( typename Graph::IndexType, typename Graph::IndexType ) -> bool
   {
      return true;
   };
   static_assert(
      ! TNL::Graphs::Algorithms::detail::isVertexPredicate_v< decltype( bad ), Graph >,
      "Vertex predicate with wrong arity must be rejected." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isEdgeWeightCallable_accepts_valid_signature )
{
   using Graph = typename TestFixture::GraphType;
   auto good = []( typename Graph::IndexType, typename Graph::IndexType, typename Graph::ValueType ) ->
      typename Graph::ValueType
   {
      return {};
   };
   static_assert(
      TNL::Graphs::Algorithms::detail::isEdgeWeightCallable_v< decltype( good ), Graph >,
      "Valid edge-weight callable (source, target, weight) -> ValueType must be accepted." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isEdgeWeightCallable_rejects_non_callable )
{
   using Graph = typename TestFixture::GraphType;
   int notCallable = 0;
   static_assert(
      ! TNL::Graphs::Algorithms::detail::isEdgeWeightCallable_v< decltype( notCallable ), Graph >,
      "Non-callable object must be rejected as edge-weight callable." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isBfsVisitor_accepts_valid_signature )
{
   using Graph = typename TestFixture::GraphType;
   auto good = []( typename Graph::IndexType, typename Graph::IndexType ) -> void {};
   static_assert(
      TNL::Graphs::Algorithms::detail::isBfsVisitor_v< decltype( good ), Graph >,
      "Valid BFS visitor (node, distance) -> void must be accepted." );
   SUCCEED();
}

TYPED_TEST( LambdaTraitsTest, isBfsVisitor_rejects_wrong_arity )
{
   using Graph = typename TestFixture::GraphType;
   auto bad = []( typename Graph::IndexType, typename Graph::IndexType, typename Graph::ValueType ) -> void {};
   static_assert(
      ! TNL::Graphs::Algorithms::detail::isBfsVisitor_v< decltype( bad ), Graph >,
      "BFS visitor with wrong arity must be rejected." );
   SUCCEED();
}
