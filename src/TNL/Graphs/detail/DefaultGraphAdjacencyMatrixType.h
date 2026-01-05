// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/SparseMatrix.h>
#include "../GraphOrientation.h"

namespace TNL::Graphs::detail {

template< typename Value,
          typename Device,
          typename Index,
          template< typename, typename, typename > class Segments,
          typename Orientation >
struct DefaultGraphAdjacencyMatrix
{
   using type = TNL::Matrices::SparseMatrix< Value, Device, Index, TNL::Matrices::GeneralMatrix, Segments >;
};

template< typename Value, typename Device, typename Index, template< typename, typename, typename > class Segments >
struct DefaultGraphAdjacencyMatrix< Value, Device, Index, Segments, UndirectedGraph >
{
   using type = TNL::Matrices::SparseMatrix< Value, Device, Index, TNL::Matrices::SymmetricMatrix, Segments >;
};

template< typename Value,
          typename Device,
          typename Index,
          template< typename, typename, typename > class Segments,
          typename Orientation >
using DefaultGraphAdjacencyMatrix_t = typename DefaultGraphAdjacencyMatrix< Value, Device, Index, Segments, Orientation >::type;
}  // namespace TNL::Graphs::detail
