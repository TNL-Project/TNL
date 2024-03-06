
#pragma once
#include <functional>
#include <iostream>
#include <sstream>
#include <TNL/Algorithms/Segments/ElementsOrganization.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/MatrixType.h>
#include <TNL/Algorithms/contains.h>
#include <TNL/Config/parseCommandLine.h>
#include <TNL/Benchmarks/Benchmarks.h>
#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Matrices/MatrixOperations.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Backend/Types.h>
#include <TNL/Backend/Macros.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/Stream.h>
#include <TNL/Backend/StreamPool.h>
#include <TNL/Backend/DeviceInfo.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Backend/LaunchHelpers.h>

#include <TNL/Backend/KernelLaunch.h>
#include <cmath>
#include <vector>

#include <gtest/gtest.h>

template< typename Matrix >
class DenseMatrixTranspositionTest : public ::testing::Test
{
protected:
   using DenseMatrixType = Matrix;
};
// Define MatrixTypes for the test suite
using MatrixTypes = ::testing::Types<
   TNL::Matrices::
      DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
   TNL::Matrices::
      DenseMatrix< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Host, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >
#ifdef __CUDACC__
   ,
   TNL::Matrices::
      DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
   TNL::Matrices::
      DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
#endif
   >;

TYPED_TEST_SUITE( DenseMatrixTranspositionTest, MatrixTypes );

TYPED_TEST( DenseMatrixTranspositionTest, Transposition1 )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 5, 6, 6 }, { 6, 5, 7 }, { 7, 2, 1 }, { 3, 3, 9 } };

   DenseMatrixType resultMatrix;
   resultMatrix.setDimensions( matrix.getColumns(), matrix.getRows() );

   DenseMatrixType checkMatrix{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   resultMatrix.getTransposition( matrix );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixTranspositionTest, Transposition2 )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;
   DenseMatrixType matrix{ { 4, 3, 1, 2 }, { 3, 4, 4, 3 }, { 2, 2, 7, 8 }, { 6, 8, 9, 0 },
                           { 7, 6, 5, 4 }, { 8, 1, 2, 0 }, { 4, 5, 6, 7 } };

   DenseMatrixType resultMatrix;
   resultMatrix.setDimensions( matrix.getColumns(), matrix.getRows() );

   DenseMatrixType checkMatrix{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   resultMatrix.getTransposition( matrix );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

#include "../main.h"
