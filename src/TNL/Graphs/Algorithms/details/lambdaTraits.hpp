// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>

namespace TNL::Graphs::Algorithms::detail {

/**
 * \brief Trait: callable is an edge predicate.
 *
 * An edge predicate decides whether an edge may be traversed. It must be
 * invocable with `(IndexType source, IndexType target, ValueType weight)` and
 * return `bool`.
 */
template< typename EdgePredicate, typename Graph >
struct IsEdgePredicate
: std::bool_constant<
     std::
        is_invocable_r_v< bool, EdgePredicate, typename Graph::IndexType, typename Graph::IndexType, typename Graph::ValueType > >
{};

template< typename EdgePredicate, typename Graph >
constexpr bool isEdgePredicate_v = IsEdgePredicate< EdgePredicate, Graph >::value;

/**
 * \brief Trait: callable is a vertex predicate.
 *
 * A vertex predicate decides whether a vertex belongs to the induced
 * subgraph. It must be invocable with `(IndexType vertex)` and return `bool`.
 */
template< typename VertexPredicate, typename Graph >
struct IsVertexPredicate : std::bool_constant< std::is_invocable_r_v< bool, VertexPredicate, typename Graph::IndexType > >
{};

template< typename VertexPredicate, typename Graph >
constexpr bool isVertexPredicate_v = IsVertexPredicate< VertexPredicate, Graph >::value;

/**
 * \brief Trait: callable is an SSSP edge-weight transformation.
 *
 * Unlike an edge predicate, the edge-weight callable returns a
 * `ValueType` instead of `bool`. Returning +/- infinity marks the edge as
 * non-traversable. It must be invocable with
 * `(IndexType source, IndexType target, ValueType weight)`.
 */
template< typename EdgeWeightCallable, typename Graph >
struct IsEdgeWeightCallable : std::bool_constant< std::is_invocable_r_v<
                                 typename Graph::ValueType,
                                 EdgeWeightCallable,
                                 typename Graph::IndexType,
                                 typename Graph::IndexType,
                                 typename Graph::ValueType > >
{};

template< typename EdgeWeightCallable, typename Graph >
constexpr bool isEdgeWeightCallable_v = IsEdgeWeightCallable< EdgeWeightCallable, Graph >::value;

/**
 * \brief Trait: callable is a BFS visitor.
 *
 * A BFS visitor is invoked upon visiting each node. It must be invocable with
 * `(IndexType node, IndexType distance)` and return `void`.
 */
template< typename Visitor, typename Graph >
struct IsBfsVisitor
: std::bool_constant< std::is_invocable_r_v< void, Visitor, typename Graph::IndexType, typename Graph::IndexType > >
{};

template< typename Visitor, typename Graph >
constexpr bool isBfsVisitor_v = IsBfsVisitor< Visitor, Graph >::value;

}  // namespace TNL::Graphs::Algorithms::detail
