// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

namespace TNL::Graphs {

template< typename MatrixType, typename GraphType_ >
struct Graph;

//! \brief This checks if given type is matrix.
[[nodiscard]] constexpr std::false_type
isGraph( ... )
{
   return {};
}

template< typename MatrixType, typename GraphType_ >
[[nodiscard]] constexpr std::true_type
isGraph( const Graph< MatrixType, GraphType_ >& )
{
   return {};
}

}  // namespace TNL::Graphs
