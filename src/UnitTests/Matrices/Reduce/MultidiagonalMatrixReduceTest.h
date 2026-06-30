// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Functional.h>
#include <TNL/Matrices/MultidiagonalMatrix.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Containers/Vector.h>
#include <gtest/gtest.h>

// Types for which MultidiagonalMatrixReduceTest exercises the free reduction functions.
using MultidiagonalMatrixReduceTypes = ::testing::Types<
#if ! defined( __CUDACC__ ) && ! defined( __HIP__ )
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, int >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, int >,
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Host, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Host, long >
#elif defined( __CUDACC__ )
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, int >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, int >,
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Cuda, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Cuda, long >
#elif defined( __HIP__ )
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, int >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, int >,
   TNL::Matrices::MultidiagonalMatrix< float, TNL::Devices::Hip, long >,
   TNL::Matrices::MultidiagonalMatrix< double, TNL::Devices::Hip, long >
#endif
   >;

template< typename MatrixType >
void
setupTestMatrix( MatrixType& matrix )
{
   using IndexType = typename MatrixType::IndexType;
   using RealType = typename MatrixType::RealType;

   const IndexType rows = 6;
   const IndexType cols = 6;
   TNL::Containers::Vector< IndexType, TNL::Devices::Host, IndexType > offsets{ -2, -1, 0, 1, 2 };
   matrix.setDimensions( rows, cols, offsets );

   for( IndexType row = 0; row < rows; row++ ) {
      for( IndexType localIdx = 0; localIdx < 5; localIdx++ ) {
         const IndexType column = row + ( localIdx - 2 );
         if( column >= 0 && column < cols )
            matrix.setElement( row, column, (RealType) ( row * 10 + localIdx ) );
      }
   }
}

template< typename MatrixType >
void
test_reduceRows()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceRows(
      view,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceRows(
      constView,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );
}

template< typename MatrixType >
void
test_reduceRows_AutoIdentity()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceRows(
      view,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceRows(
      constView,
      1,
      4,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 0 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 0 );
   EXPECT_EQ( rowSums.getElement( 5 ), 0 );
}

template< typename MatrixType >
void
test_reduceAllRows()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceAllRows(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceAllRows(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      },
      (RealType) 0 );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );
}

template< typename MatrixType >
void
test_reduceAllRows_AutoIdentity()
{
   using RealType = typename MatrixType::RealType;
   using IndexType = typename MatrixType::IndexType;
   using VectorType = TNL::Containers::Vector< RealType, typename MatrixType::DeviceType, IndexType >;

   MatrixType matrix;
   setupTestMatrix( matrix );
   auto view = matrix.getView();

   VectorType rowSums( matrix.getRows() );
   rowSums.setValue( 0 );
   auto rowSumsView = rowSums.getView();

   TNL::Matrices::reduceAllRows(
      view,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );

   const auto constView = view;
   rowSums.setValue( 0 );

   TNL::Matrices::reduceAllRows(
      constView,
      [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, const RealType& value ) -> RealType
      {
         return value;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( IndexType rowIdx, const RealType& value ) mutable
      {
         rowSumsView[ rowIdx ] = value;
      } );

   EXPECT_EQ( rowSums.getElement( 0 ), 9 );
   EXPECT_EQ( rowSums.getElement( 1 ), 50 );
   EXPECT_EQ( rowSums.getElement( 2 ), 110 );
   EXPECT_EQ( rowSums.getElement( 3 ), 160 );
   EXPECT_EQ( rowSums.getElement( 4 ), 166 );
   EXPECT_EQ( rowSums.getElement( 5 ), 153 );
}
