// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include <memory>  // std::unique_ptr

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Containers/StaticArray.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Pointers/DevicePointer.h>

#include "MatrixBase.h"
#include "reduce.h"

namespace TNL::Matrices {

namespace detail {

/**
 * \brief Copies elements of the sparse matrix `B` to the sparse matrix `A`
 * allocated on a different device.
 *
 * The matrix `A` must already have the dimensions and row capacities set,
 * and all its column indexes must be set to the padding index.
 * The elements are transferred in chunks of rows packed into compact buffers,
 * so the amount of transferred data is proportional
 * to the number of matrix elements.
 * The size of the buffers is bounded by `chunkSize`
 * or by the length of the longest row.
 * Zero elements of `B` are not copied.
 *
 * \tparam TargetMatrix The type of the target sparse matrix.
 * \tparam SourceMatrix The type of the source sparse matrix.
 *
 * \param A The target matrix.
 * \param B The source matrix.
 * \param generalToSymmetric If `true`, only the lower part of `B` is copied.
 * \param symmetricToGeneral If `true`, `B` stores the lower part
 *                           of a symmetric matrix
 *                           and the upper part of `A` is reconstructed from it.
 * \param chunkSize The maximal number of elements transferred at once,
 *                  unless a row of `B` is longer.
 */
template< typename TargetMatrix, typename SourceMatrix >
void
copySparseMatrixElementsAcrossDevices(
   TargetMatrix& A,
   const SourceMatrix& B,
   bool generalToSymmetric,
   bool symmetricToGeneral,
   typename SourceMatrix::IndexType chunkSize = 1 << 22 )
{
   using Index = typename TargetMatrix::IndexType;
   using Real = typename TargetMatrix::RealType;
   using Device = typename TargetMatrix::DeviceType;
   using RealAllocatorType = typename TargetMatrix::RealAllocatorType;
   using SourceIndexType = typename SourceMatrix::IndexType;
   using SourceRealType = typename SourceMatrix::RealType;
   using SourceDeviceType = typename SourceMatrix::DeviceType;
   using SourceRealAllocatorType = typename SourceMatrix::RealAllocatorType;

   const SourceIndexType rowsCount = B.getRows();
   if( rowsCount == 0 )
      return;

   // Count the elements copied from each row of B and compute offsets of the rows in the packed buffers
   Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > sourceOffsets( rowsCount + 1 );
   auto sourceOffsets_view = sourceOffsets.getView();
   const auto b_view = B.getConstView();
   reduceAllRows(
      b_view,
      [ = ] __cuda_callable__(
         SourceIndexType rowIdx, SourceIndexType columnIdx, const SourceRealType& value ) -> SourceIndexType
      {
         return value != SourceRealType{ 0 } && ( ! generalToSymmetric || columnIdx <= rowIdx );
      },
      TNL::Plus{},
      [ = ] __cuda_callable__( SourceIndexType rowIdx, SourceIndexType count ) mutable
      {
         sourceOffsets_view[ rowIdx ] = count;
      },
      SourceIndexType( 0 ) );
   sourceOffsets.setElement( rowsCount, 0 );
   const SourceIndexType maxRowLength = max( sourceOffsets );
   Algorithms::inplaceExclusiveScan( sourceOffsets );

   const SourceIndexType elementsCount = sourceOffsets.getElement( rowsCount );
   if( elementsCount == 0 )
      return;

   // The offsets are needed on the host for splitting the rows into chunks and on the target device for unpacking
   const Containers::Vector< SourceIndexType, Devices::Host, SourceIndexType > hostOffsets( sourceOffsets );
   const Containers::Vector< Index, Device, Index > targetOffsets( sourceOffsets );
   const auto targetOffsets_view = targetOffsets.getConstView();

   // Allocate the buffers, every row must fit into them
   const SourceIndexType bufferSize = min( max( chunkSize, maxRowLength ), elementsCount );
   const SourceIndexType valuesBufferSize = TargetMatrix::isBinary() ? 0 : bufferSize;
   Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > sourceColumnsBuffer( bufferSize );
   Containers::Vector< SourceRealType, SourceDeviceType, SourceIndexType, SourceRealAllocatorType > sourceValuesBuffer(
      valuesBufferSize );
   Containers::Vector< Index, Device, Index > targetColumnsBuffer( bufferSize );
   Containers::Vector< Real, Device, Index, RealAllocatorType > targetValuesBuffer( valuesBufferSize );
   auto sourceColumnsBuffer_view = sourceColumnsBuffer.getView();
   auto sourceValuesBuffer_view = sourceValuesBuffer.getView();
   const auto targetColumnsBuffer_view = targetColumnsBuffer.getConstView();
   const auto targetValuesBuffer_view = targetValuesBuffer.getConstView();

   // Local indexes of the next free elements in rows of A, needed only when reconstructing the upper part of A
   Containers::Vector< Index, Device, Index > aLocalIndexes( symmetricToGeneral ? A.getRows() : 0, 0 );
   auto aLocalIndexes_view = aLocalIndexes.getView();
   const auto segments_view = A.getSegments().getView();
   auto columns_view = A.getColumnIndexes().getView();
   auto values_view = A.getValues().getView();

   SourceIndexType firstRow = 0;
   while( firstRow < rowsCount ) {
      // Find the largest chunk of rows starting at firstRow which fits into the buffers
      const SourceIndexType* offsets = hostOffsets.getData();
      const SourceIndexType firstElement = offsets[ firstRow ];
      // this is just to avoid overflow of firstElement + bufferSize
      const SourceIndexType lastElement = firstElement < elementsCount - bufferSize ? firstElement + bufferSize : elementsCount;
      const SourceIndexType lastRow =
         std::upper_bound( offsets + firstRow + 1, offsets + rowsCount + 1, lastElement ) - offsets - 1;
      const SourceIndexType chunkElementsCount = offsets[ lastRow ] - firstElement;

      // Pack the elements of the chunk on the source device
      auto pack = [ = ] __cuda_callable__( SourceIndexType rowIdx ) mutable
      {
         const auto row = b_view.getRow( rowIdx );
         SourceIndexType bufferIdx = sourceOffsets_view[ rowIdx ] - firstElement;
         for( SourceIndexType localIdx = 0; localIdx < row.getSize(); localIdx++ ) {
            const SourceIndexType columnIdx = row.getColumnIndex( localIdx );
            if( row.getValue( localIdx ) != SourceRealType{ 0 }
                && columnIdx != paddingIndex< SourceIndexType > && ( ! generalToSymmetric || columnIdx <= rowIdx ) )
            {
               sourceColumnsBuffer_view[ bufferIdx ] = columnIdx;
               if( ! TargetMatrix::isBinary() )
                  sourceValuesBuffer_view[ bufferIdx ] = row.getValue( localIdx );
               bufferIdx++;
            }
         }
      };
      Algorithms::parallelFor< SourceDeviceType >( firstRow, lastRow, pack );

      // Transfer the packed elements to the target device
      targetColumnsBuffer.getView( 0, chunkElementsCount ) = sourceColumnsBuffer.getConstView( 0, chunkElementsCount );
      if( ! TargetMatrix::isBinary() )
         targetValuesBuffer.getView( 0, chunkElementsCount ) = sourceValuesBuffer.getConstView( 0, chunkElementsCount );

      // Unpack the elements of the chunk on the target device
      auto unpack = [ = ] __cuda_callable__( Index rowIdx ) mutable
      {
         const Index begin = targetOffsets_view[ rowIdx ] - firstElement;
         const Index end = targetOffsets_view[ rowIdx + 1 ] - firstElement;
         for( Index bufferIdx = begin; bufferIdx < end; bufferIdx++ ) {
            const Index columnIdx = targetColumnsBuffer_view[ bufferIdx ];
            const Index localIdx =
               symmetricToGeneral
                  ? Algorithms::AtomicOperations< Device >::add( aLocalIndexes_view[ rowIdx ], static_cast< Index >( 1 ) )
                  : bufferIdx - begin;
            Index globalIdx = segments_view.getGlobalIndex( rowIdx, localIdx );
            TNL_ASSERT_GE(
               globalIdx, 0, "Global index must be non-negative. Negative values may appear due to Index type overflow." );
            columns_view[ globalIdx ] = columnIdx;
            if( ! TargetMatrix::isBinary() )
               values_view[ globalIdx ] = targetValuesBuffer_view[ bufferIdx ];
            if( symmetricToGeneral && rowIdx != columnIdx ) {  // Transpose element
               globalIdx = segments_view.getGlobalIndex(
                  columnIdx,
                  Algorithms::AtomicOperations< Device >::add( aLocalIndexes_view[ columnIdx ], static_cast< Index >( 1 ) ) );
               TNL_ASSERT_GE(
                  globalIdx, 0, "Global index must be non-negative. Negative values may appear due to Index type overflow." );
               columns_view[ globalIdx ] = rowIdx;
               if( ! TargetMatrix::isBinary() )
                  values_view[ globalIdx ] = targetValuesBuffer_view[ bufferIdx ];
            }
         }
      };
      Algorithms::parallelFor< Device >( firstRow, lastRow, unpack );

      firstRow = lastRow;
   }
}

}  // namespace detail

template< typename TargetMatrix, typename SourceMatrix >
void
copySparseToSparseMatrix( TargetMatrix& A, const SourceMatrix& B )
{
   using Index = typename TargetMatrix::IndexType;
   using Device = typename TargetMatrix::DeviceType;
   using SourceIndexType = typename SourceMatrix::IndexType;
   using SourceRealType = typename SourceMatrix::RealType;
   using SourceDeviceType = typename SourceMatrix::DeviceType;

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
      TNL::Matrices::reduceAllRows(
         B,
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
               TNL_ASSERT_GE(
                  thisGlobalIdx,
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
      detail::copySparseMatrixElementsAcrossDevices( A, B, generalToSymmetric, false );
   }
}

template< typename TargetMatrix, typename SourceMatrix >
void
copySymmetricSparseToGeneralSparseMatrix( TargetMatrix& A, const SourceMatrix& B )
{
   static_assert( SourceMatrix::isSymmetric(), "SourceMatrix must be symmetric." );
   static_assert( ! TargetMatrix::isSymmetric(), "TargetMatrix must be general (not symmetric)." );

   using Index = typename TargetMatrix::IndexType;
   using Device = typename TargetMatrix::DeviceType;
   using SourceIndexType = typename SourceMatrix::IndexType;
   using SourceRealType = typename SourceMatrix::RealType;
   using SourceDeviceType = typename SourceMatrix::DeviceType;

   // Set row capacities for matrix A
   Containers::Vector< SourceIndexType, SourceDeviceType, SourceIndexType > rowCapacities( B.getRows(), 0 );
   auto rowCapacities_view = rowCapacities.getView();
   B.reduceAllRows(
      [ = ] __cuda_callable__(
         SourceIndexType rowIdx, SourceIndexType columnIdx, const SourceRealType& value ) mutable -> SourceIndexType
      {
         if( rowIdx != columnIdx )
            Algorithms::AtomicOperations< SourceDeviceType >::add(
               rowCapacities_view[ columnIdx ], static_cast< SourceIndexType >( 1 ) );
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
                  rowIdx,
                  Algorithms::AtomicOperations< Device >::add( aLocalIndexes_view[ rowIdx ], static_cast< Index >( 1 ) ) );
               TNL_ASSERT_GE(
                  thisGlobalIdx,
                  0,
                  "Global index must be non-negative. Negative values may appear due to Index type overflow." );
               columns_view[ thisGlobalIdx ] = columnIdx;
               if( ! TargetMatrix::isBinary() )
                  values_view[ thisGlobalIdx ] = value;
               if( rowIdx != columnIdx ) {  // Transpose element
                  thisGlobalIdx = segments_view.getGlobalIndex(
                     columnIdx,
                     Algorithms::AtomicOperations< Device >::add(
                        aLocalIndexes_view[ columnIdx ], static_cast< Index >( 1 ) ) );
                  TNL_ASSERT_GE(
                     thisGlobalIdx,
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
      detail::copySparseMatrixElementsAcrossDevices( A, B, false, true );
   }
}

template< typename Vector, typename Matrix >
__global__
void
SparseMatrixSetRowLengthsVectorKernel(
   Vector* rowLengths,
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
SparseMatrixCopyKernel(
   Matrix1* A,
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
   static_assert(
      std::is_same_v< typename Matrix1::RealType, typename Matrix2::RealType >, "The matrices must have the same RealType." );
   static_assert(
      std::is_same_v< typename Matrix1::DeviceType, typename Matrix2::DeviceType >,
      "The matrices must be allocated on the same device." );
   static_assert(
      std::is_same_v< typename Matrix1::IndexType, typename Matrix2::IndexType >,
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
      Backend::launchKernelSync(
         kernelRowLengths,
         launch_config,
         rowLengths.getData(),
         &Bpointer.template getData< TNL::Devices::Cuda >(),
         rows,
         cols );
      Apointer->setRowCapacities( rowLengths );

      // copy rows
      Pointers::synchronizeSmartPointersOnDevice< Devices::Cuda >();
      constexpr auto kernelCopy = SparseMatrixCopyKernel< Matrix1, Matrix2 >;
      Backend::launchKernelSync(
         kernelCopy,
         launch_config,
         &Apointer.template modifyData< TNL::Devices::Cuda >(),
         &Bpointer.template getData< TNL::Devices::Cuda >(),
         rowLengths.getData(),
         rows );
   }
}

// cross-device copy (host -> gpu)
template< typename Matrix1, typename Matrix2 >
std::enable_if_t<
   ! std::is_same_v< typename Matrix1::DeviceType, typename Matrix2::DeviceType >
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
std::enable_if_t<
   ! std::is_same_v< typename Matrix1::DeviceType, typename Matrix2::DeviceType >
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
   static_assert(
      std::is_same_v< typename Matrix::DeviceType, Devices::Host >,
      "The function is not implemented for CUDA matrices - it would require atomic insertions "
      "of elements into the sparse format." );
   static_assert(
      std::is_same_v< typename Matrix::DeviceType, typename AdjacencyMatrix::DeviceType >,
      "The matrices must be allocated on the same device." );
   static_assert(
      std::is_same_v< typename Matrix::IndexType, typename AdjacencyMatrix::IndexType >,
      "The matrices must have the same IndexType." );
   //static_assert( std::is_same_v< typename AdjacencyMatrix::RealType, bool >,
   //               "The RealType of the adjacency matrix must be bool." );

   using RealType = typename Matrix::RealType;
   using IndexType = typename Matrix::IndexType;

   if( A.getRows() != A.getColumns() ) {
      throw std::logic_error(
         "The matrix is not square: " + std::to_string( A.getRows() ) + " rows, " + std::to_string( A.getColumns() )
         + " columns." );
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
   static_assert(
      std::is_same_v< typename Matrix1::DeviceType, Devices::Host >
         || std::is_same_v< typename Matrix1::DeviceType, Devices::Sequential >,
      "matrix reordering is implemented only for host" );
   static_assert(
      std::is_same_v< typename Matrix2::DeviceType, Devices::Host >
         || std::is_same_v< typename Matrix2::DeviceType, Devices::Sequential >,
      "matrix reordering is implemented only for host" );
   static_assert(
      std::is_same_v< typename PermutationArray::DeviceType, Devices::Host >
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
      TNL::Plus{},
      [ = ] __cuda_callable__( Index rowIdx, Index value ) mutable
      {
         row_capacities_view[ rowIdx ] = value;
      } );
   Matrix aux_matrix( A.getRows(), A.getColumns() );
   aux_matrix.setRowCapacities( row_capacities );

   auto aux_matrix_view = aux_matrix.getView();
   A.forAllRows(
      [ = ] __cuda_callable__( RowView& row ) mutable
      {
         auto aux_matrix_row = aux_matrix_view.getRow( row.getRowIndex() );
         Index localIdx = 0;
         for( auto element : row )
            if( element.value() != 0.0 )
               aux_matrix_row.setElement( localIdx++, element.columnIndex(), element.value() );
      } );
   A = std::move( aux_matrix );
}

template< typename Array1, typename Array2, typename PermutationArray >
void
reorderArray( const Array1& src, Array2& dest, const PermutationArray& perm )
{
   static_assert(
      std::is_same_v< typename Array1::DeviceType, typename Array2::DeviceType >, "Arrays must reside on the same device." );
   static_assert(
      std::is_same_v< typename Array1::DeviceType, typename PermutationArray::DeviceType >,
      "Arrays must reside on the same device." );
   if( src.getSize() != perm.getSize() )
      throw std::invalid_argument( "reorderArray: source array and permutation must have the same size." );
   if( dest.getSize() != perm.getSize() )
      throw std::invalid_argument( "reorderArray: destination array and permutation must have the same size." );

   using DeviceType = typename Array1::DeviceType;
   using IndexType = typename Array1::IndexType;

   auto kernel = [] __cuda_callable__(
                    IndexType i,
                    const typename Array1::ValueType* src,
                    typename Array2::ValueType* dest,
                    const typename PermutationArray::ValueType* perm )
   {
      dest[ i ] = src[ perm[ i ] ];
   };

   Algorithms::parallelFor< DeviceType >( 0, src.getSize(), kernel, src.getData(), dest.getData(), perm.getData() );
}

}  // namespace TNL::Matrices
