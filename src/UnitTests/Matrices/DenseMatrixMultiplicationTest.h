
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
class DenseMatrixMultiplicationTest : public ::testing::Test
{
protected:
   using DenseMatrixType = Matrix;
};

template< typename Matrix >
class DenseMatrixMultiplicationTransposedTest : public ::testing::Test
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

using MatrixTypesCuda = ::testing::Types<
#ifdef __CUDACC__
   TNL::Matrices::
      DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< double, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
   TNL::Matrices::
      DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::ColumnMajorOrder >,
   TNL::Matrices::DenseMatrix< float, TNL::Devices::Cuda, int, TNL::Algorithms::Segments::ElementsOrganization::RowMajorOrder >,
#endif
   >;

TYPED_TEST_SUITE( DenseMatrixMultiplicationTest, MatrixTypes );

TYPED_TEST( DenseMatrixMultiplicationTest, IdentityProduct )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 1, 4, 5, 6, 3 }, { 8, 1, 0, 4, 5 }, { 5, 0, 1, 0, 5 }, { 4, 5, 2, 1, 7 }, { 7, 8, 0, 9, 9 } };

   DenseMatrixType matrix2{ { 1, 0, 0, 0, 0 }, { 0, 1, 0, 0, 0 }, { 0, 0, 1, 0, 0 }, { 0, 0, 0, 1, 0 }, { 0, 0, 0, 0, 1 } };

   DenseMatrixType resultMatrix;
   resultMatrix.setDimensions( matrix1.getRows(), matrix2.getColumns() );

   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == matrix1;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, NormalProduct )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   DenseMatrixType matrix2{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 51, 76, 95, 141, 118, 60, 113 },
                                { 47, 55, 60, 94, 94, 57, 82 },
                                { 64, 77, 105, 101, 125, 57, 128 } };

   resultMatrix.setDimensions( matrix1.getRows(), matrix2.getColumns() );
   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTest, NormalProductWrong )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   DenseMatrixType matrix2{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 52, 76, 95, 141, 118, 60, 113 },
                                { 47, 55, 60, 94, 94, 57, 82 },
                                { 64, 77, 105, 101, 125, 57, 128 } };

   resultMatrix.setDimensions( matrix1.getRows(), matrix2.getColumns() );
   resultMatrix.getMatrixProduct( matrix1, matrix2, 1.0 );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, false );
}

TYPED_TEST_SUITE( DenseMatrixMultiplicationTransposedTest, MatrixTypesCuda );
TYPED_TEST( DenseMatrixMultiplicationTransposedTest, TransposedTimesNormal )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 6 }, { 6, 5, 7 }, { 7, 2, 1 }, { 3, 3, 9 } };

   DenseMatrixType matrix2{
      { 4, 3, 2, 6, 7, 8, 4 }, { 3, 4, 2, 8, 6, 1, 5 }, { 1, 4, 7, 9, 5, 2, 6 }, { 2, 3, 8, 0, 4, 0, 7 }
   };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 51, 76, 95, 141, 118, 60, 113 },
                                { 47, 55, 60, 94, 94, 57, 82 },
                                { 64, 77, 105, 101, 125, 57, 128 } };

   resultMatrix.setDimensions( matrix1.getColumns(), matrix2.getColumns() );

   resultMatrix.getMatrixProduct(
      matrix1, matrix2, 1.0, TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::None );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTransposedTest, NormalTimesTransposed )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 7, 3 }, { 6, 5, 2, 3 }, { 6, 7, 1, 9 } };

   DenseMatrixType matrix2{ { 4, 3, 1, 2 }, { 3, 4, 4, 3 }, { 2, 2, 7, 8 }, { 6, 8, 9, 0 },
                            { 7, 6, 5, 4 }, { 8, 1, 2, 0 }, { 4, 5, 6, 7 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 51, 76, 95, 141, 118, 60, 113 },
                                { 47, 55, 60, 94, 94, 57, 82 },
                                { 64, 77, 105, 101, 125, 57, 128 } };

   resultMatrix.setDimensions( matrix1.getRows(), matrix2.getRows() );
   resultMatrix.getMatrixProduct(
      matrix1, matrix2, 1.0, TNL::Matrices::TransposeState::None, TNL::Matrices::TransposeState::Transpose );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}

TYPED_TEST( DenseMatrixMultiplicationTransposedTest, TransposedTimesTransposed )
{
   using DenseMatrixType = typename TestFixture::DenseMatrixType;

   DenseMatrixType matrix1{ { 5, 6, 6 }, { 6, 5, 7 }, { 7, 2, 1 }, { 3, 3, 9 } };

   DenseMatrixType matrix2{ { 4, 3, 1, 2 }, { 3, 4, 4, 3 }, { 2, 2, 7, 8 }, { 6, 8, 9, 0 },
                            { 7, 6, 5, 4 }, { 8, 1, 2, 0 }, { 4, 5, 6, 7 } };

   DenseMatrixType resultMatrix;

   DenseMatrixType checkMatrix{ { 51, 76, 95, 141, 118, 60, 113 },
                                { 47, 55, 60, 94, 94, 57, 82 },
                                { 64, 77, 105, 101, 125, 57, 128 } };

   resultMatrix.setDimensions( matrix1.getColumns(), matrix2.getRows() );

   resultMatrix.getMatrixProduct(
      matrix1, matrix2, 1.0, TNL::Matrices::TransposeState::Transpose, TNL::Matrices::TransposeState::Transpose );

   bool check = resultMatrix == checkMatrix;
   ASSERT_EQ( check, true );
}
#include "../main.h"
