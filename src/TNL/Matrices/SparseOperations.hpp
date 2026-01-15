// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <memory>  // std::unique_ptr

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Pointers/DevicePointer.h>

#include "MatrixBase.h"

//#if defined( __CUDACC__ ) && ( __CUDACC_VER_MAJOR__ < 12 )
#define USE_NVCC_WORKAROUND  // This is workaround for nvcc which is not able to compile copySparseToSparseMatrix function
                             // due to the lambda functions in the code. This issue appears at least with
                             // nvcc build cuda_11.8.r11.8/compiler.31833905_0 and g++ 11.3.0.
//#endif

namespace TNL::Matrices {

#ifdef USE_NVCC_WORKAROUND
template< typename Matrix, typename Index, typename IndexVector, typename ValueVector >
void
copyMatrixElementsToBuffers( const Matrix& m,
                             Index baseRow,
                             Index lastRow,
                             Index maxRowLength,
                             IndexVector& columnsBuffer,
                             ValueVector& valuesBuffer,
                             bool generalToSymmetric = false )
{
   using RHSIndexType = typename Matrix::IndexType;
   using RHSRealType = typename Matrix::RealType;

   auto matrixColumnsBuffer_view = columnsBuffer.getView();
   auto matrixValuesBuffer_view = valuesBuffer.getView();

   // Copy matrix elements into buffer
   auto f1 = [ = ] __cuda_callable__(
                RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value ) mutable
   {
      if( generalToSymmetric && columnIndex > rowIdx )
         return;
      if( columnIndex != paddingIndex< Index > ) {
         TNL_ASSERT_LT( localIdx, maxRowLength, "" );
         const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
         matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
         matrixValuesBuffer_view[ bufferIdx ] = value;
      }
   };
   m.forElements( baseRow, lastRow, f1 );
}

template< typename Matrix, typename Index, typename IndexVectorView, typename ValueVectorView, typename RowLengthsVector >
void
copyBuffersToMatrixElements( Matrix& m,
                             const IndexVectorView& thisColumnsBuffer_view,
                             const ValueVectorView& thisValuesBuffer_view,
                             Index baseRow,
                             Index lastRow,
                             Index maxRowLength,
                             RowLengthsVector& thisRowLengths,
                             IndexVectorView rowLocalIndexes_view )
{
   using Real = typename Matrix::RealType;
   constexpr Index padding_index = paddingIndex< Index >;  // this is just to avoid nvcc error: identifier
                                                           // "TNL::Matrices::paddingIndex<int> " is undefined in device
                                                           // code From src/UnitTests/Matrices/SparseMatrixCopyTest.cu

   auto thisRowLengths_view = thisRowLengths.getView();
   auto f2 = [ = ] __cuda_callable__( typename Matrix::RowView row ) mutable
   {
      const Index rowIdx = row.getRowIndex();
      for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
         Real inValue = 0;
         std::size_t bufferIdx = 0;
         Index bufferLocalIdx = rowLocalIndexes_view[ rowIdx ];
         while( inValue == Real{ 0 } && localIdx < thisRowLengths_view[ rowIdx ] ) {
            bufferIdx = ( rowIdx - baseRow ) * maxRowLength + bufferLocalIdx++;
            inValue = thisValuesBuffer_view[ bufferIdx ];
         }
         rowLocalIndexes_view[ rowIdx ] = bufferLocalIdx;
         if( inValue == Real{ 0 } ) {
            row.setColumnIndex( localIdx, padding_index );
            row.setValue( localIdx, 0 );
         }
         else {
            row.setColumnIndex( localIdx, thisColumnsBuffer_view[ bufferIdx ] );  // column - 1;
            row.setValue( localIdx, inValue );
         }
      }
   };
   m.forRows( baseRow, lastRow, f2 );
}
#endif

template< typename TargetMatrix, typename SourceMatrix >
void
copySparseToSparseMatrix( TargetMatrix& A, const SourceMatrix& B )
{
   using Index = typename TargetMatrix::IndexType;
   using Real = typename TargetMatrix::RealType;
   using Device = typename TargetMatrix::DeviceType;
   using RealAllocatorType = typename TargetMatrix::RealAllocatorType;
   using SourceIndexType = typename SourceMatrix::IndexType;
   using SourceRealType = typename SourceMatrix::RealType;
   using SourceDeviceType = typename SourceMatrix::DeviceType;
   using SourceRealAllocatorType = typename SourceMatrix::RealAllocatorType;

   if constexpr( ! TargetMatrix::MatrixType::isSymmetric() && SourceMatrix::MatrixType::isSymmetric() ) {
      copySymmetricSparseToGeneralSparseMatrix( A, B );
      return;
   }

   constexpr bool generalToSymmetric = TargetMatrix::MatrixType::isSymmetric() && ! SourceMatrix::MatrixType::isSymmetric();

   // Set row capacities for matrix A
   Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > rowCapacities;
   if( generalToSymmetric ) {
      rowCapacities.setSize( B.getRows() );
      auto rowCapacities_view = rowCapacities.getView();
      B.reduceAllRows(
         [ = ] __cuda_callable__(
            SourceIndexType rowIdx, SourceIndexType columnIdx, const SourceRealType& value ) mutable -> SourceIndexType
         {
            if( columnIdx <= rowIdx )
               return 1;
            else
               return 0;
         },
         TNL::Plus{},
         [ = ] __cuda_callable__( SourceIndexType rowIdx, SourceIndexType value ) mutable
         {
            rowCapacities_view[ rowIdx ] = value;
         },
         SourceIndexType( 0 ) );
      A.setDimensions( B.getRows(), B.getColumns() );
      A.setRowCapacities( rowCapacities );
   }
   else {
      B.getRowCapacities( rowCapacities );
      A.setDimensions( B.getRows(), B.getColumns() );
      A.setRowCapacities( rowCapacities );
   }

   auto columns_view = A.getColumnIndexes().getView();
   auto values_view = A.getValues().getView();
   columns_view = paddingIndex< Index >;

   if constexpr( std::is_same_v< Device, SourceDeviceType > ) {
      const auto segments_view = A.getSegments().getView();
      const auto b_view = B.getConstView();

      auto f = [ = ] __cuda_callable__( SourceIndexType rowIdx ) mutable
      {
         auto row = b_view.getRow( rowIdx );
         Index aLocalIdx( 0 );
         for( SourceIndexType bLocalIdx = 0; bLocalIdx < row.getSize(); bLocalIdx++ ) {
            if( row.getValue( bLocalIdx ) != 0.0 && row.getColumnIndex( bLocalIdx ) != paddingIndex< SourceIndexType > ) {
               const SourceIndexType columnIdx = row.getColumnIndex( bLocalIdx );
               if( generalToSymmetric )  // TODO: make this constexpr when nvcc allows it
                  if( columnIdx > rowIdx )
                     continue;

               Index thisGlobalIdx = segments_view.getGlobalIndex( rowIdx, aLocalIdx++ );
               TNL_ASSERT_GE( thisGlobalIdx,
                              0,
                              "Global index must be non-negative. Negative values may appear due to Index type overflow." );
               columns_view[ thisGlobalIdx ] = columnIdx;
               if( ! TargetMatrix::isBinary() )
                  values_view[ thisGlobalIdx ] = row.getValue( bLocalIdx );
            }
         }
      };
      Algorithms::parallelFor< SourceDeviceType >( 0, B.getRows(), f );  // TODO: Rewrite this using forRows
   }
   else {  // Not the same device
      Containers::Vector< Index, Device, Index > rowLocalIndexes( B.getRows() );
      rowLocalIndexes = 0;
      auto rowLocalIndexes_view = rowLocalIndexes.getView();
      const Index maxRowLength = max( rowCapacities );
      const Index bufferRowsCount = 4096;
      const std::size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< SourceRealType, SourceDeviceType, SourceIndexType, SourceRealAllocatorType > matrixValuesBuffer(
         bufferSize );
      Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > matrixColumnsBuffer( bufferSize );
      Containers::Vector< Real, Device, Index, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< Index, Device, Index > thisColumnsBuffer( bufferSize );
      Containers::Vector< Index, Device, Index > thisRowLengths;
      Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > sourceRowLengths;
      B.getCompressedRowLengths( sourceRowLengths );
      thisRowLengths = sourceRowLengths;
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto matrixColumnsBuffer_view = matrixColumnsBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();
      auto thisColumnsBuffer_view = thisColumnsBuffer.getView();
      matrixValuesBuffer_view = 0.0;

      Index baseRow = 0;
      const Index rowsCount = A.getRows();
      while( baseRow < rowsCount ) {
         const Index lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex< Index >;
         matrixColumnsBuffer_view = paddingIndex< Index >;

#ifdef USE_NVCC_WORKAROUND
         copyMatrixElementsToBuffers(
            B, baseRow, lastRow, maxRowLength, matrixColumnsBuffer, matrixValuesBuffer, generalToSymmetric );
#else
         // Copy matrix elements into buffer
         auto f1 = [ = ] __cuda_callable__( const typename SourceMatrix::ConstRowView& row ) mutable
         {
            const SourceIndexType rowIdx = row.getRowIndex();
            for( SourceIndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
               const SourceIndexType columnIndex = row.getColumnIndex( localIdx );
               if( generalToSymmetric )  // TODO: make this constexpr when nvcc allows it
                  if( columnIndex > rowIdx )
                     continue;
               const SourceRealType& value = row.getValue( localIdx );
               if( columnIndex != paddingIndex< SourceIndexType > ) {
                  TNL_ASSERT_LT( rowIdx - baseRow, bufferRowsCount, "" );
                  TNL_ASSERT_LT( localIdx, maxRowLength, "" );
                  const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
                  TNL_ASSERT_LT( bufferIdx, (Index) bufferSize, "" );
                  matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
                  matrixValuesBuffer_view[ bufferIdx ] = value;
               }
            }
         };
         B.forRows( baseRow, lastRow, f1 );
#endif
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;
         thisColumnsBuffer_view = matrixColumnsBuffer_view;

#ifdef USE_NVCC_WORKAROUND
         copyBuffersToMatrixElements( A,
                                      thisColumnsBuffer_view,
                                      thisValuesBuffer_view,
                                      baseRow,
                                      lastRow,
                                      maxRowLength,
                                      thisRowLengths,
                                      rowLocalIndexes_view );
#else
         // Copy matrix elements from the buffer to the matrix and ignoring
         // zero matrix elements
         const Index padding_index = paddingIndex< Index >;  // this is just to avoid nvcc error: identifier
                                                             // "TNL::Matrices::paddingIndex<int> " is undefined in device
                                                             // code From src/UnitTests/Matrices/SparseMatrixCopyTest.cu
         const auto thisRowLengths_view = thisRowLengths.getConstView();
         auto f2 = [ = ] __cuda_callable__( typename TargetMatrix::RowView row ) mutable
         {
            const Index rowIdx = row.getRowIndex();
            for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
               Real inValue( 0 );
               std::size_t bufferIdx( 0 );
               Index bufferLocalIdx = rowLocalIndexes_view[ rowIdx ];
               while( inValue == Real{ 0 } && localIdx < thisRowLengths_view[ rowIdx ] ) {
                  bufferIdx = ( rowIdx - baseRow ) * maxRowLength + bufferLocalIdx++;
                  TNL_ASSERT_LT( bufferIdx, bufferSize, "" );
                  inValue = thisValuesBuffer_view[ bufferIdx ];
               }
               rowLocalIndexes_view[ rowIdx ] = bufferLocalIdx;
               if( inValue == Real{ 0 } ) {
                  row.setColumnIndex( localIdx, padding_index );
                  row.setValue( localIdx, 0 );
               }
               else {
                  row.setColumnIndex( localIdx, thisColumnsBuffer_view[ bufferIdx ] );  // column - 1;
                  row.setValue( localIdx, inValue );
               }
            }
         };
         A.forRows( baseRow, lastRow, f2 );
#endif
         baseRow += bufferRowsCount;
      }
   }
}

#ifdef USE_NVCC_WORKAROUND
template< typename Matrix, typename Index, typename IndexVector, typename ValueVector >
void
copyMatrixElementsToBuffers_SymmetricToGeneral( const Matrix& m,
                                                Index baseRow,
                                                Index lastRow,
                                                Index maxRowLength,
                                                IndexVector& columnsBuffer,
                                                ValueVector& valuesBuffer,
                                                IndexVector& bufferRowLocalIndexes )
{
   using RealType = typename Matrix::RealType;
   using DeviceType = typename Matrix::DeviceType;
   using IndexType = typename Matrix::IndexType;

   auto matrixColumnsBuffer_view = columnsBuffer.getView();
   auto matrixValuesBuffer_view = valuesBuffer.getView();
   auto bufferRowLocalIndexes_view = bufferRowLocalIndexes.getView();

   auto f1 = [ = ] __cuda_callable__( const typename Matrix::ConstRowView& row ) mutable
   {
      const IndexType rowIdx = row.getRowIndex();
      for( IndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
         TNL_ASSERT_LT( localIdx, row.getSize(), "" );
         const IndexType columnIndex = row.getColumnIndex( localIdx );
         const RealType& value = row.getValue( localIdx );
         if( rowIdx >= baseRow && rowIdx < lastRow )
            if( columnIndex != paddingIndex< IndexType > ) {
               TNL_ASSERT_LT( localIdx, maxRowLength, "" );
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength
                                         + Algorithms::AtomicOperations< DeviceType >::add(
                                              bufferRowLocalIndexes_view[ rowIdx - baseRow ], (Index) 1 );
               matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
               matrixValuesBuffer_view[ bufferIdx ] = value;
            }
         if( columnIndex != paddingIndex< IndexType > && rowIdx != columnIndex && columnIndex >= baseRow
             && columnIndex < lastRow )
         {
            const IndexType bufferIdx = ( columnIndex - baseRow ) * maxRowLength
                                      + Algorithms::AtomicOperations< DeviceType >::add(
                                           bufferRowLocalIndexes_view[ columnIndex - baseRow ], (Index) 1 );
            matrixColumnsBuffer_view[ bufferIdx ] = rowIdx;
            matrixValuesBuffer_view[ bufferIdx ] = value;
         }
      }
   };
   m.forAllRows( f1 );
}
#endif

template< typename TargetMatrix, typename SourceMatrix >
void
copySymmetricSparseToGeneralSparseMatrix( TargetMatrix& A, const SourceMatrix& B )
{
   static_assert( SourceMatrix::isSymmetric(), "SourceMatrix must be symmetric." );
   static_assert( ! TargetMatrix::isSymmetric(), "TargetMatrix must be general (not symmetric)." );

   using Index = typename TargetMatrix::IndexType;
   using Real = typename TargetMatrix::RealType;
   using Device = typename TargetMatrix::DeviceType;
   using RealAllocatorType = typename TargetMatrix::RealAllocatorType;
   using SourceIndexType = typename SourceMatrix::IndexType;
   using SourceRealType = typename SourceMatrix::RealType;
   using SourceDeviceType = typename SourceMatrix::DeviceType;
   using SourceRealAllocatorType = typename SourceMatrix::RealAllocatorType;

   // Set row capacities for matrix A
   Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > rowCapacities( B.getRows(), 0 );
   auto rowCapacities_view = rowCapacities.getView();
   B.reduceAllRows(
      [ = ] __cuda_callable__(
         SourceIndexType rowIdx, SourceIndexType columnIdx, const SourceRealType& value ) mutable -> SourceIndexType
      {
         if( rowIdx != columnIdx )
            Algorithms::AtomicOperations< SourceDeviceType >::add( rowCapacities_view[ columnIdx ], (SourceIndexType) 1 );
         return 1;
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( SourceIndexType rowIdx, SourceIndexType value ) mutable
      {
         Algorithms::AtomicOperations< SourceDeviceType >::add( rowCapacities_view[ rowIdx ], value );
      } );
   A.setDimensions( B.getRows(), B.getColumns() );
   A.setRowCapacities( rowCapacities );

   auto columns_view = A.getColumnIndexes().getView();
   auto values_view = A.getValues().getView();
   columns_view = paddingIndex< Index >;

   if constexpr( std::is_same_v< Device, SourceDeviceType > ) {
      const auto segments_view = A.getSegments().getView();
      Containers::Vector< Index, Device, Index > aLocalIndexes( A.getRows(), 0 );
      auto aLocalIndexes_view = aLocalIndexes.getView();

      auto f = [ = ] __cuda_callable__( typename SourceMatrix::ConstRowView row ) mutable
      {
         const SourceIndexType rowIdx = row.getRowIndex();
         for( SourceIndexType bLocalIdx = 0; bLocalIdx < row.getSize(); bLocalIdx++ ) {
            if( row.getValue( bLocalIdx ) != 0.0 && row.getColumnIndex( bLocalIdx ) != paddingIndex< SourceIndexType > ) {
               const SourceIndexType columnIdx = row.getColumnIndex( bLocalIdx );
               const SourceRealType value = row.getValue( bLocalIdx );

               Index thisGlobalIdx = segments_view.getGlobalIndex(
                  rowIdx, Algorithms::AtomicOperations< Device >::add( aLocalIndexes_view[ rowIdx ], (Index) 1 ) );
               TNL_ASSERT_GE( thisGlobalIdx,
                              0,
                              "Global index must be non-negative. Negative values may appear due to Index type overflow." );
               columns_view[ thisGlobalIdx ] = columnIdx;
               if( ! TargetMatrix::isBinary() )
                  values_view[ thisGlobalIdx ] = value;
               if( rowIdx != columnIdx ) {  // Transpose element
                  thisGlobalIdx = segments_view.getGlobalIndex(
                     columnIdx, Algorithms::AtomicOperations< Device >::add( aLocalIndexes_view[ columnIdx ], (Index) 1 ) );
                  TNL_ASSERT_GE( thisGlobalIdx,
                                 0,
                                 "Global index must be non-negative. Negative values may appear due to Index type overflow." );
                  columns_view[ thisGlobalIdx ] = rowIdx;
                  if( ! TargetMatrix::isBinary() )
                     values_view[ thisGlobalIdx ] = value;
               }
            }
         }
      };
      B.forAllRows( f );
   }
   else {  // Not the same device
      Containers::Vector< Index, Device, Index > rowLocalIndexes( B.getRows() );
      rowLocalIndexes = 0;
      auto rowLocalIndexes_view = rowLocalIndexes.getView();
      const Index maxRowLength = max( rowCapacities );
      const Index bufferRowsCount = 4096;
      const std::size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< SourceRealType, SourceDeviceType, SourceIndexType, SourceRealAllocatorType > matrixValuesBuffer(
         bufferSize );
      Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > matrixColumnsBuffer( bufferSize );
      Containers::Vector< Real, Device, Index, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< Index, Device, Index > thisColumnsBuffer( bufferSize );
      Containers::Vector< Index, Device, Index > thisRowLengths;
      Containers::Vector< Index, SourceDeviceType, Index > bufferRowLocalIndexes( bufferRowsCount, 0 );
      thisRowLengths = rowCapacities;
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto matrixColumnsBuffer_view = matrixColumnsBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();
      auto thisColumnsBuffer_view = thisColumnsBuffer.getView();
      auto bufferRowLocalIndexes_view = bufferRowLocalIndexes.getView();
      matrixValuesBuffer_view = 0.0;

      Index baseRow = 0;
      const Index rowsCount = A.getRows();
      while( baseRow < rowsCount ) {
         const Index lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = paddingIndex< Index >;
         matrixColumnsBuffer_view = paddingIndex< Index >;

#ifdef USE_NVCC_WORKAROUND
         copyMatrixElementsToBuffers_SymmetricToGeneral(
            B, baseRow, lastRow, maxRowLength, matrixColumnsBuffer, matrixValuesBuffer, bufferRowLocalIndexes );
#else
         // Copy matrix elements into buffer
         auto B_view = B.getConstView();
         auto f1 = [ = ] __cuda_callable__( const typename SourceMatrix::ConstRowView& row ) mutable
         {
            const SourceIndexType rowIdx = row.getRowIndex();
            for( SourceIndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
               TNL_ASSERT_LT( localIdx, row.getSize(), "" );
               const SourceIndexType columnIndex = row.getColumnIndex( localIdx );
               const SourceRealType& value = row.getValue( localIdx );
               if( rowIdx >= baseRow && rowIdx < lastRow )
                  if( columnIndex != paddingIndex< SourceIndexType > ) {
                     TNL_ASSERT_LT( rowIdx - baseRow, bufferRowsCount, "" );
                     TNL_ASSERT_LT( localIdx, maxRowLength, "" );
                     const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength
                                           + Algorithms::AtomicOperations< SourceDeviceType >::add(
                                                bufferRowLocalIndexes_view[ rowIdx - baseRow ], (Index) 1 );
                     TNL_ASSERT_LT( bufferIdx, (Index) bufferSize, "" );
                     matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
                     matrixValuesBuffer_view[ bufferIdx ] = value;
                  }
               if( columnIndex != paddingIndex< SourceIndexType > && rowIdx != columnIndex && columnIndex >= baseRow
                   && columnIndex < lastRow )
               {
                  TNL_ASSERT_LT( columnIndex - baseRow, bufferRowsCount, "" );
                  const Index bufferIdx = ( columnIndex - baseRow ) * maxRowLength
                                        + Algorithms::AtomicOperations< SourceDeviceType >::add(
                                             bufferRowLocalIndexes_view[ columnIndex - baseRow ], (Index) 1 );
                  TNL_ASSERT_LT( bufferIdx, (Index) bufferSize, "" );
                  matrixColumnsBuffer_view[ bufferIdx ] = rowIdx;
                  matrixValuesBuffer_view[ bufferIdx ] = value;
               }
            }
         };
         B.forAllRows( f1 );
#endif
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;
         thisColumnsBuffer_view = matrixColumnsBuffer_view;

#ifdef USE_NVCC_WORKAROUND
         copyBuffersToMatrixElements( A,
                                      thisColumnsBuffer_view,
                                      thisValuesBuffer_view,
                                      baseRow,
                                      lastRow,
                                      maxRowLength,
                                      thisRowLengths,
                                      rowLocalIndexes_view );
#else
         // Copy matrix elements from the buffer to the matrix and ignoring
         // zero matrix elements
         const auto thisRowLengths_view = thisRowLengths.getConstView();
         const auto padding_index = paddingIndex< Index >;  // TODO:: Fix - SparseOperations.hpp(532): error: identifier
                                                            // "TNL::Matrices::paddingIndex<int> " is undefined in device code
                                                            // From Documentation/Examples/Matrices/MatrixWriterReaderExample.cu

         auto f2 = [ = ] __cuda_callable__( typename TargetMatrix::RowView row ) mutable
         {
            const Index rowIdx = row.getRowIndex();
            for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
               Real inValue( 0 );
               std::size_t bufferIdx( 0 );
               Index bufferLocalIdx = rowLocalIndexes_view[ rowIdx ];
               while( inValue == Real{ 0 } && localIdx < thisRowLengths_view[ rowIdx ] ) {
                  bufferIdx = ( rowIdx - baseRow ) * maxRowLength + bufferLocalIdx++;
                  TNL_ASSERT_LT( bufferIdx, bufferSize, "" );
                  inValue = thisValuesBuffer_view[ bufferIdx ];
               }
               rowLocalIndexes_view[ rowIdx ] = bufferLocalIdx;
               if( inValue == Real{ 0 } ) {
                  row.setColumnIndex( localIdx, padding_index );
                  row.setValue( localIdx, 0 );
               }
               else {
                  row.setColumnIndex( localIdx, thisColumnsBuffer_view[ bufferIdx ] );
                  row.setValue( localIdx, inValue );
               }
            }
         };
         A.forRows( baseRow, lastRow, f2 );
#endif
         baseRow += bufferRowsCount;
      }
   }
}

template< typename Vector, typename Matrix >
__global__
void
SparseMatrixSetRowLengthsVectorKernel( Vector* rowLengths,
                                       const Matrix* matrix,
                                       typename Matrix::IndexType rows,
                                       typename Matrix::IndexType cols )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using IndexType = typename Matrix::IndexType;

   IndexType rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   while( rowIdx < rows ) {
      const auto row = matrix->getRow( rowIdx );
      IndexType length = 0;
      for( IndexType c_j = 0; c_j < row.getSize(); c_j++ )
         if( row.getColumnIndex( c_j ) < cols )
            length++;
         else
            break;
      rowLengths[ rowIdx ] = length;
      rowIdx += gridSize;
   }
#endif
}

template< typename Matrix1, typename Matrix2 >
__global__
void
SparseMatrixCopyKernel( Matrix1* A,
                        const Matrix2* B,
                        const typename Matrix2::IndexType* rowLengths,
                        typename Matrix2::IndexType rows )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using IndexType = typename Matrix2::IndexType;

   IndexType rowIdx = blockIdx.x * blockDim.x + threadIdx.x;
   const IndexType gridSize = blockDim.x * gridDim.x;

   while( rowIdx < rows ) {
      const auto length = rowLengths[ rowIdx ];
      const auto rowB = B->getRow( rowIdx );
      auto rowA = A->getRow( rowIdx );
      for( IndexType c = 0; c < length; c++ )
         rowA.setElement( c, rowB.getColumnIndex( c ), rowB.getValue( c ) );
      rowIdx += gridSize;
   }
#endif
}

// copy on the same device
template< typename Matrix1, typename Matrix2 >
std::enable_if_t< std::is_same_v< typename Matrix1::DeviceType, typename Matrix2::DeviceType > >
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   static_assert( std::is_same_v< typename Matrix1::RealType, typename Matrix2::RealType >,
                  "The matrices must have the same RealType." );
   static_assert( std::is_same_v< typename Matrix1::DeviceType, typename Matrix2::DeviceType >,
                  "The matrices must be allocated on the same device." );
   static_assert( std::is_same_v< typename Matrix1::IndexType, typename Matrix2::IndexType >,
                  "The matrices must have the same IndexType." );

   using DeviceType = typename Matrix1::DeviceType;
   using IndexType = typename Matrix1::IndexType;

   const IndexType rows = B.getRows();
   const IndexType cols = B.getColumns();

   A.setDimensions( rows, cols );

   if constexpr( std::is_same_v< DeviceType, Devices::Host > ) {
      // set row lengths
      typename Matrix1::RowCapacitiesType rowLengths;
      rowLengths.setSize( rows );
#ifdef HAVE_OPENMP
   #pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < rows; i++ ) {
         const auto row = B.getRow( i );
         IndexType length = 0;
         for( IndexType c_j = 0; c_j < row.getSize(); c_j++ )
            if( row.getColumnIndex( c_j ) < cols )
               length++;
            else
               break;
         rowLengths[ i ] = length;
      }
      A.setRowCapacities( rowLengths );

#ifdef HAVE_OPENMP
   #pragma omp parallel for if( Devices::Host::isOMPEnabled() )
#endif
      for( IndexType i = 0; i < rows; i++ ) {
         const auto length = rowLengths[ i ];
         const auto rowB = B.getRow( i );
         auto rowA = A.getRow( i );
         for( IndexType c = 0; c < length; c++ )
            rowA.setElement( c, rowB.getColumnIndex( c ), rowB.getValue( c ) );
      }
   }

   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > ) {
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      const IndexType desGridSize = 32 * Backend::getDeviceMultiprocessors( Backend::getDevice() );
      launch_config.gridSize.x = min( desGridSize, Backend::getNumberOfBlocks( rows, launch_config.blockSize.x ) );

      typename Matrix1::RowCapacitiesType rowLengths;
      rowLengths.setSize( rows );

      Pointers::DevicePointer< Matrix1 > Apointer( A );
      const Pointers::DevicePointer< const Matrix2 > Bpointer( B );

      // set row lengths
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      constexpr auto kernelRowLengths =
         SparseMatrixSetRowLengthsVectorKernel< typename Matrix1::RowCapacitiesType::ValueType, Matrix2 >;
      Backend::launchKernelSync( kernelRowLengths,
                                 launch_config,
                                 rowLengths.getData(),
                                 &Bpointer.template getData< TNL::Devices::Cuda >(),
                                 rows,
                                 cols );
      Apointer->setRowCapacities( rowLengths );

      // copy rows
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      constexpr auto kernelCopy = SparseMatrixCopyKernel< Matrix1, Matrix2 >;
      Backend::launchKernelSync( kernelCopy,
                                 launch_config,
                                 &Apointer.template modifyData< TNL::Devices::Cuda >(),
                                 &Bpointer.template getData< TNL::Devices::Cuda >(),
                                 rowLengths.getData(),
                                 rows );
   }
}

// cross-device copy (host -> gpu)
template< typename Matrix1, typename Matrix2 >
std::enable_if_t< ! std::is_same_v< typename Matrix1::DeviceType, typename Matrix2::DeviceType >
                  && std::is_same_v< typename Matrix2::DeviceType, Devices::Host > >
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   using CudaMatrix2 = typename Matrix2::template Self< typename Matrix2::RealType, Devices::Cuda >;
   CudaMatrix2 B_tmp;
   B_tmp = B;
   copySparseMatrix_impl( A, B_tmp );
}

// cross-device copy (gpu -> host)
template< typename Matrix1, typename Matrix2 >
std::enable_if_t< ! std::is_same_v< typename Matrix1::DeviceType, typename Matrix2::DeviceType >
                  && std::is_same_v< typename Matrix2::DeviceType, Devices::Cuda > >
copySparseMatrix_impl( Matrix1& A, const Matrix2& B )
{
   using CudaMatrix1 = typename Matrix1::template Self< typename Matrix1::RealType, Devices::Cuda >;
   CudaMatrix1 A_tmp;
   copySparseMatrix_impl( A_tmp, B );
   A = A_tmp;
}

template< typename Matrix1, typename Matrix2 >
void
copySparseMatrix( Matrix1& A, const Matrix2& B )
{
   copySparseMatrix_impl( A, B );
}

template< typename Matrix, typename AdjacencyMatrix >
void
copyAdjacencyStructure( const Matrix& A, AdjacencyMatrix& B, bool has_symmetric_pattern, bool ignore_diagonal )
{
   static_assert( std::is_same_v< typename Matrix::DeviceType, Devices::Host >,
                  "The function is not implemented for CUDA matrices - it would require atomic insertions "
                  "of elements into the sparse format." );
   static_assert( std::is_same_v< typename Matrix::DeviceType, typename AdjacencyMatrix::DeviceType >,
                  "The matrices must be allocated on the same device." );
   static_assert( std::is_same_v< typename Matrix::IndexType, typename AdjacencyMatrix::IndexType >,
                  "The matrices must have the same IndexType." );
   //static_assert( std::is_same_v< typename AdjacencyMatrix::RealType, bool >,
   //               "The RealType of the adjacency matrix must be bool." );

   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   if( A.getRows() != A.getColumns() ) {
      throw std::logic_error( "The matrix is not square: " + std::to_string( A.getRows() ) + " rows, "
                              + std::to_string( A.getColumns() ) + " columns." );
   }

   const IndexType N = A.getRows();
   B.setDimensions( N, N );

   // set row lengths
   typename AdjacencyMatrix::RowCapacitiesType rowLengths;
   rowLengths.setSize( N );
   rowLengths.setValue( 0 );
   for( IndexType i = 0; i < A.getRows(); i++ ) {
      const auto row = A.getRow( i );
      IndexType length = 0;
      for( int c_j = 0; c_j < row.getSize(); c_j++ ) {
         const IndexType j = row.getColumnIndex( c_j );
         if( j >= A.getColumns() )
            break;
         length++;
         if( ! has_symmetric_pattern && i != j )
            if( A.getElement( j, i ) == RealType{ 0 } )
               rowLengths[ j ]++;
      }
      if( ignore_diagonal )
         length--;
      rowLengths[ i ] += length;
   }
   B.setRowCapacities( rowLengths );

   // set non-zeros
   for( IndexType i = 0; i < A.getRows(); i++ ) {
      const auto row = A.getRow( i );
      for( int c_j = 0; c_j < row.getSize(); c_j++ ) {
         const IndexType j = row.getColumnIndex( c_j );
         if( j >= A.getColumns() )
            break;
         if( ! ignore_diagonal || i != j )
            if( A.getElement( i, j ) != 0 ) {
               B.setElement( i, j, true );
               if( ! has_symmetric_pattern )
                  B.setElement( j, i, true );
            }
      }
   }
}

template< typename Matrix1, typename Matrix2, typename PermutationArray >
void
reorderSparseMatrix( const Matrix1& matrix1, Matrix2& matrix2, const PermutationArray& perm, const PermutationArray& iperm )
{
   // TODO: implement on GPU
   static_assert( std::is_same_v< typename Matrix1::DeviceType, Devices::Host >
                     || std::is_same_v< typename Matrix1::DeviceType, Devices::Sequential >,
                  "matrix reordering is implemented only for host" );
   static_assert( std::is_same_v< typename Matrix2::DeviceType, Devices::Host >
                     || std::is_same_v< typename Matrix2::DeviceType, Devices::Sequential >,
                  "matrix reordering is implemented only for host" );
   static_assert( std::is_same_v< typename PermutationArray::DeviceType, Devices::Host >
                     || std::is_same_v< typename PermutationArray::DeviceType, Devices::Sequential >,
                  "matrix reordering is implemented only for host" );

   using IndexType = typename Matrix1::IndexType;

   matrix2.setDimensions( matrix1.getRows(), matrix1.getColumns() );

   // set row lengths
   typename Matrix2::RowCapacitiesType rowLengths;
   rowLengths.setSize( matrix1.getRows() );
   for( IndexType i = 0; i < matrix1.getRows(); i++ ) {
      const auto row = matrix1.getRow( perm[ i ] );
      IndexType length = 0;
      for( IndexType j = 0; j < row.getSize(); j++ )
         if( row.getColumnIndex( j ) < matrix1.getColumns() )
            length++;
      rowLengths[ i ] = length;
   }
   matrix2.setRowCapacities( rowLengths );

   // set row elements
   for( IndexType i = 0; i < matrix2.getRows(); i++ ) {
      const IndexType rowLength = rowLengths[ i ];

      // extract sparse row
      const auto row1 = matrix1.getRow( perm[ i ] );

      // permute
      std::unique_ptr< typename Matrix2::IndexType[] > columns{ new typename Matrix2::IndexType[ rowLength ] };
      std::unique_ptr< typename Matrix2::RealType[] > values{ new typename Matrix2::RealType[ rowLength ] };
      for( IndexType j = 0; j < rowLength; j++ ) {
         columns[ j ] = iperm[ row1.getColumnIndex( j ) ];
         values[ j ] = row1.getValue( j );
      }

      // sort
      std::unique_ptr< IndexType[] > indices{ new IndexType[ rowLength ] };
      for( IndexType j = 0; j < rowLength; j++ )
         indices[ j ] = j;
      auto comparator = [ &columns ]( IndexType a, IndexType b )
      {
         return columns[ a ] < columns[ b ];
      };
      std::sort( indices.get(), indices.get() + rowLength, comparator );

      // set the row
      auto row2 = matrix2.getRow( i );
      for( IndexType j = 0; j < rowLength; j++ )
         row2.setElement( j, columns[ indices[ j ] ], values[ indices[ j ] ] );
   }
}

template< typename Matrix >
void
compressSparseMatrix( Matrix& A )
{
   using Real = typename Matrix::RealType;
   using Device = typename Matrix::DeviceType;
   using Index = typename Matrix::IndexType;
   using RowView = typename Matrix::RowView;

   if( all( notEqualTo( A.getValues(), Real{ 0 } ) ) )
      return;

   Containers::Vector< Index, Device, Index > row_capacities( A.getRows() );
   auto row_capacities_view = row_capacities.getView();
   A.reduceAllRows(
      [ = ] __cuda_callable__( Index rowIdx, Index columnIdx, const Real& value ) -> Index
      {
         return columnIdx != paddingIndex< Index > && ( value != Real{ 0 } );
      },
      std::plus<>{},
      [ = ] __cuda_callable__( Index rowIdx, Index value ) mutable
      {
         row_capacities_view[ rowIdx ] = value;
      },
      0 );
   Matrix aux_matrix( A.getRows(), A.getColumns() );
   aux_matrix.setRowCapacities( row_capacities );

   auto aux_matrix_view = aux_matrix.getView();
   A.forAllRows(
      [ = ] __cuda_callable__( RowView & row ) mutable
      {
         auto aux_matrix_row = aux_matrix_view.getRow( row.getRowIndex() );
         Index localIdx = 0;
         for( auto element : row )
            if( element.value() != 0.0 )
               aux_matrix_row.setElement( localIdx++, element.columnIndex(), element.value() );
      } );
   A = aux_matrix;
}

template< typename Array1, typename Array2, typename PermutationArray >
void
reorderArray( const Array1& src, Array2& dest, const PermutationArray& perm )
{
   static_assert( std::is_same_v< typename Array1::DeviceType, typename Array2::DeviceType >,
                  "Arrays must reside on the same device." );
   static_assert( std::is_same_v< typename Array1::DeviceType, typename PermutationArray::DeviceType >,
                  "Arrays must reside on the same device." );
   if( src.getSize() != perm.getSize() )
      throw std::invalid_argument( "reorderArray: source array and permutation must have the same size." );
   if( dest.getSize() != perm.getSize() )
      throw std::invalid_argument( "reorderArray: destination array and permutation must have the same size." );

   using DeviceType = typename Array1::DeviceType;
   using IndexType = typename Array1::IndexType;

   auto kernel = [] __cuda_callable__( IndexType i,
                                       const typename Array1::ValueType* src,
                                       typename Array2::ValueType* dest,
                                       const typename PermutationArray::ValueType* perm )
   {
      dest[ i ] = src[ perm[ i ] ];
   };

   Algorithms::parallelFor< DeviceType >( 0, src.getSize(), kernel, src.getData(), dest.getData(), perm.getData() );
}

}  // namespace TNL::Matrices
