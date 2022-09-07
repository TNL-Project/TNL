// Copyright (c) 2004-2022 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>

namespace TNL {
namespace Meshes {
namespace Templates {

/*
 * A compiler-friendly implementation of the templated for-cycle, because
 * the template specializations count is O(Value) bounded.
 */
template< int >
struct DescendingFor;

template< int Value >
struct DescendingFor
{
public:
   template< typename Func, typename... FuncArgs >
   inline static void
   exec( Func&& func, FuncArgs&&... args )
   {
      static_assert( Value > 0, "Couldn't descend for negative values" );

      func( std::integral_constant< int, Value >(), std::forward< FuncArgs >( args )... );

      DescendingFor< Value - 1 >::exec( std::forward< Func >( func ), std::forward< FuncArgs >( args )... );
   }
};

template<>
struct DescendingFor< 0 >
{
public:
   template< typename Func, typename... FuncArgs >
   inline static void
   exec( Func&& func, FuncArgs&&... args )
   {
      func( std::integral_constant< int, 0 >(), std::forward< FuncArgs >( args )... );
   }
};

}  // namespace Templates
}  // namespace Meshes
}  // namespace TNL
