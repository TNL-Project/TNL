// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/GPU.h>

#include "DenseSparseOperations.h"

namespace TNL::Matrices {

template< typename Matrix1, typename Matrix2 >
void
copySparseToDenseMatrix( Matrix1& A, const Matrix2& B )
{
   using Index = typename Matrix1::IndexType;
   using Real = typename Matrix1::RealType;
   using Device = typename Matrix1::DeviceType;
   using RealAllocatorType = typename Matrix1::RealAllocatorType;
   using RHSIndexType = typename Matrix2::IndexType;
   using RHSRealType = typename Matrix2::RealType;
   using RHSDeviceType = typename Matrix2::DeviceType;
   using RHSRealAllocatorType = typename Matrix2::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowLengths;
   B.getCompressedRowLengths( rowLengths );
   A.setDimensions( B.getRows(), B.getColumns() );

   if constexpr( std::is_same_v< Device, RHSDeviceType > ) {
      auto A_view = A.getView();
      auto f = [ = ] __cuda_callable__(
                  RHSIndexType rowIdx, RHSIndexType localIdx_, RHSIndexType columnIdx, const RHSRealType& value ) mutable
      {
         TNL_ASSERT_LT( rowIdx, A_view.getRows(), "Row index is larger than number of matrix rows." );
         TNL_ASSERT_LT( columnIdx, A_view.getColumns(), "Column index is larger than number of matrix columns." );
         if( value != 0.0 && columnIdx != paddingIndex< Index > )
            A_view( rowIdx, columnIdx ) = value;
      };
      B.forAllElements( f );
   }
   else {
      const Index maxRowLength = max( rowLengths );
      const Index bufferRowsCount = 128;
      const std::size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > matrixColumnsBuffer( bufferSize );
      Containers::Vector< Real, Device, Index, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< Index, Device, Index > thisColumnsBuffer( bufferSize );
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto matrixColumnsBuffer_view = matrixColumnsBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();
      auto thisColumnsBuffer_view = thisColumnsBuffer.getView();

      Index baseRow = 0;
      const Index rowsCount = A.getRows();
      while( baseRow < rowsCount ) {
         const Index lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex< Index >;
         matrixColumnsBuffer_view = paddingIndex< Index >;
         auto B_view = B.getConstView();
         auto f1 = [ = ] __cuda_callable__( RHSIndexType rowIdx ) mutable
         {
            auto row = B_view.getRow( rowIdx );
            for( RHSIndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
               const RHSIndexType columnIndex = row.getColumnIndex( localIdx );
               const RHSRealType value = row.getValue( localIdx );
               if( columnIndex != paddingIndex< Index > && columnIndex >= 0
                   && columnIndex < B_view.getColumns() ) {  // columnIndex >= 0 && columnIndex < A_view.getColumns() is
                                                             // necessary because of the tridiagonal and
                  // multidiagonal matrices
                  const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
                  matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
                  matrixValuesBuffer_view[ bufferIdx ] = value;
               }
            }
         };
         Algorithms::parallelFor< RHSDeviceType >( baseRow, lastRow, f1 );

         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;
         thisColumnsBuffer_view = matrixColumnsBuffer_view;

         // Copy matrix elements from the buffer to the matrix
         auto A_view = A.getView();
         using MultiIndex = Containers::StaticArray< 2, Index >;
         auto f2 = [ = ] __cuda_callable__( const MultiIndex& i ) mutable
         {
            const Index& bufferColumnIdx = i[ 0 ];
            const Index& bufferRowIdx = i[ 1 ];
            const Index bufferIdx = bufferRowIdx * maxRowLength + bufferColumnIdx;
            const Index columnIdx = thisColumnsBuffer_view[ bufferIdx ];
            if( columnIdx != paddingIndex< Index > ) {
               TNL_ASSERT_LT( baseRow + bufferRowIdx, A_view.getRows(), "Row index is larger than number of matrix rows." );
               TNL_ASSERT_LT( columnIdx, A_view.getColumns(), "Column index is larger than number of matrix columns." );
               A_view( baseRow + bufferRowIdx, columnIdx ) = thisValuesBuffer_view[ bufferIdx ];
            }
         };
         MultiIndex begin = { 0, 0 };
         MultiIndex end = { maxRowLength, min( bufferRowsCount, A.getRows() - baseRow ) };
         Algorithms::parallelFor< Device >( begin, end, f2 );
         baseRow += bufferRowsCount;
      }
   }
}

template< typename Matrix1, typename Matrix2 >
void
copyDenseToSparseMatrix( Matrix1& A, const Matrix2& B )
{
   using Index = typename Matrix1::IndexType;
   using Real = typename Matrix1::RealType;
   using Device = typename Matrix1::DeviceType;
   using RealAllocatorType = typename Matrix1::RealAllocatorType;
   using IndexAllocatorType = typename Matrix1::IndexAllocatorType;
   using RHSIndexType = typename Matrix2::IndexType;
   using RHSRealType = typename Matrix2::RealType;
   using RHSDeviceType = typename Matrix2::DeviceType;
   using RHSRealAllocatorType = typename Matrix2::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowLengths;
   B.getCompressedRowLengths( rowLengths );
   A.setLike( B );
   A.setRowCapacities( rowLengths );
   Containers::Vector< Index, Device, Index > rowLocalIndexes( B.getRows() );
   rowLocalIndexes = 0;

   auto columns_view = A.getColumnIndexes().getView();
   auto values_view = A.getValues().getView();
   auto rowLocalIndexes_view = rowLocalIndexes.getView();
   columns_view = paddingIndex< Index >;

   if constexpr( std::is_same_v< Device, RHSDeviceType > ) {
      const auto segments_view = A.getSegments().getView();
      auto B_view = B.getConstView();
      auto f = [ = ] __cuda_callable__( RHSIndexType rowIdx ) mutable
      {
         auto row = B_view.getRow( rowIdx );
         for( RHSIndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
            const RHSIndexType columnIndex = row.getColumnIndex( localIdx );
            const RHSRealType value = row.getValue( localIdx );
            if( value != 0.0 ) {
               Index thisGlobalIdx = segments_view.getGlobalIndex( rowIdx, rowLocalIndexes_view[ rowIdx ]++ );
               columns_view[ thisGlobalIdx ] = columnIndex;
               if( ! Matrix1::isBinary() )
                  values_view[ thisGlobalIdx ] = value;
            }
         }
      };
      Algorithms::parallelFor< RHSDeviceType >( 0, B.getRows(), f );
   }
   else {
      const Index maxRowLength = B.getColumns();
      const Index bufferRowsCount = 4096;
      const std::size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< Real, Device, Index, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< Index, Device, Index, IndexAllocatorType > thisColumnsBuffer( bufferSize );
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();

      Index baseRow = 0;
      const Index rowsCount = A.getRows();
      while( baseRow < rowsCount ) {
         const Index lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex< Index >;

         // Copy matrix elements into buffer
         auto f1 = [ = ] __cuda_callable__(
                      RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value ) mutable
         {
            const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
            matrixValuesBuffer_view[ bufferIdx ] = value;
         };
         B.forElements( baseRow, lastRow, f1 );

         // Copy the input matrix buffer to the output matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;

         // Copy matrix elements from the buffer to the matrix and ignoring
         // zero matrix elements.
         const Index matrix_columns = A.getColumns();
         auto A_view = A.getView();
         constexpr Index padding_index = paddingIndex< Index >;  // this is just to avoid nvcc error: identifier
                                                                 // "TNL::Matrices::paddingIndex<int> " is undefined in device
                                                                 // code From src/UnitTests/Matrices/SparseMatrixCopyTest.cu

         auto f2 = [ = ] __cuda_callable__( const Index rowIdx ) mutable
         {
            auto row = A_view.getRow( rowIdx );
            for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
               Real inValue = 0;
               Index column = rowLocalIndexes_view[ rowIdx ];
               while( inValue == Real{ 0 } && column < matrix_columns ) {
                  const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength + column++;
                  inValue = thisValuesBuffer_view[ bufferIdx ];
               }
               rowLocalIndexes_view[ rowIdx ] = column;
               if( inValue == Real{ 0 } ) {
                  row.setColumnIndex( localIdx, padding_index );
                  row.setValue( localIdx, 0 );
               }
               else {
                  row.setColumnIndex( localIdx, column - 1 );
                  row.setValue( localIdx, inValue );
               }
            }
         };
         Algorithms::parallelFor< Device >( baseRow, lastRow, f2 );
         baseRow += bufferRowsCount;
      }
   }
}

}  // namespace TNL::Matrices
