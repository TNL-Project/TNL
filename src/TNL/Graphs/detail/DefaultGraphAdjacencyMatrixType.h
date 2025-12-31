// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/SparseMatrix.h>
#include "../GraphOrientation.h"

namespace TNL::Graphs::detail {

template< typename Value, typename Device, typename Index, typename Orientation >
struct DefaultGraphAdjacencyMatrix
{
   using type =
      TNL::Matrices::SparseMatrix< Value, Device, Index, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >;
};

template< typename Value, typename Device, typename Index >
struct DefaultGraphAdjacencyMatrix< Value, Device, Index, UndirectedGraph >
{
   using type =
      TNL::Matrices::SparseMatrix< Value, Device, Index, TNL::Matrices::SymmetricMatrix, TNL::Algorithms::Segments::CSR >;
};

template< typename Value, typename Device, typename Index, typename Orientation >
using DefaultGraphAdjacencyMatrix_t = typename DefaultGraphAdjacencyMatrix< Value, Device, Index, Orientation >::type;

}  // namespace TNL::Graphs::detail
