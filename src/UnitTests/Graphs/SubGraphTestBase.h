// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Graphs/Graph.h>
#include <TNL/Matrices/SparseMatrix.h>

#include <gtest/gtest.h>

template< typename Matrix >
class SubGraphTest : public ::testing::Test
{
protected:
   using MatrixType = Matrix;
   using GraphType = TNL::Graphs::
      Graph< typename Matrix::RealType, typename Matrix::DeviceType, typename Matrix::IndexType, TNL::Graphs::DirectedGraph >;
};

using SubGraphTestTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Sequential, int >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int >
#elif defined( __CUDACC__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int >
#elif defined( __HIP__ )
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int >
#endif
   >;

// 5-vertex directed graph used by all SubGraph tests:
//
//    0 --(1)--> 1
//    |          |
//   (2)        (3)
//    |          |
//    v          v
//    2 --(4)--> 3 --(5)--> 4
//
// Adjacency:
//   0 -> 1 (w=1), 0 -> 2 (w=2), 1 -> 3 (w=3), 2 -> 3 (w=4), 3 -> 4 (w=5)
template< typename GraphType >
GraphType
makeTestGraph()
{
   return GraphType( 5, { { 0, 1, 1 }, { 0, 2, 2 }, { 1, 3, 3 }, { 2, 3, 4 }, { 3, 4, 5 } } );
}
