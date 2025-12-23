// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/DenseMatrix.h>
#include <gtest/gtest.h>

// Types for which MatrixReduceTest is instantiated - DenseMatrix
using DenseMatrixTypes = ::testing::Types< TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, int >,
                                           TNL::Matrices::DenseMatrix< long, TNL::Devices::Host, int >,
                                           TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int >,
                                           TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int >,
                                           TNL::Matrices::DenseMatrix< int, TNL::Devices::Host, long >,
                                           TNL::Matrices::DenseMatrix< long, TNL::Devices::Host, long >,
                                           TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, long >,
                                           TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, long >
#if defined( __CUDACC__ )
                                           ,
                                           TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, int >,
                                           TNL::Matrices::DenseMatrix< long, TNL::Devices::Cuda, int >,
                                           TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int >,
                                           TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int >,
                                           TNL::Matrices::DenseMatrix< int, TNL::Devices::Cuda, long >,
                                           TNL::Matrices::DenseMatrix< long, TNL::Devices::Cuda, long >,
                                           TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, long >,
                                           TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
                                           ,
                                           TNL::Matrices::DenseMatrix< int, TNL::Devices::Hip, int >,
                                           TNL::Matrices::DenseMatrix< long, TNL::Devices::Hip, int >,
                                           TNL::Matrices::DenseMatrix< float, TNL::Devices::Hip, int >,
                                           TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, int >,
                                           TNL::Matrices::DenseMatrix< int, TNL::Devices::Hip, long >,
                                           TNL::Matrices::DenseMatrix< long, TNL::Devices::Hip, long >,
                                           TNL::Matrices::DenseMatrix< float, TNL::Devices::Hip, long >,
                                           TNL::Matrices::DenseMatrix< double, TNL::Devices::Hip, long >
#endif
                                           >;

#include "MatrixReduceTest.h"

INSTANTIATE_TYPED_TEST_SUITE_P( DenseMatrix, MatrixReduceTest, DenseMatrixTypes );
