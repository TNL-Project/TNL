// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <gtest/gtest.h>

// Types for which MatrixTraverseTest is instantiated - SparseMatrix with CSR
using SparseMatrixCSRTypes = ::testing::Types<
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Host, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >
#if defined( __CUDACC__ )
   ,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Cuda, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >
#elif defined( __HIP__ )
   ,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, int, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< int, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< long, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< float, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >,
   TNL::Matrices::SparseMatrix< double, TNL::Devices::Hip, long, TNL::Matrices::GeneralMatrix, TNL::Algorithms::Segments::CSR >
#endif
   >;

#include "MatrixTraverseTest.h"

INSTANTIATE_TYPED_TEST_SUITE_P( SparseMatrixCSR, MatrixTraverseTest, SparseMatrixCSRTypes );
