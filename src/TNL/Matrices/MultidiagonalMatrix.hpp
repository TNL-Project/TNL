// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "MultidiagonalMatrix.h"

namespace TNL::Matrices {

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Vector >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::MultidiagonalMatrix(
   Index rows,
   Index columns,
   const Vector& diagonalOffsets )
{
   if( diagonalOffsets.getSize() == 0 )
      throw std::invalid_argument( "Cannot construct multidiagonal matrix with no diagonal offsets." );
   this->setDimensions( rows, columns, diagonalOffsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename ListIndex >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::MultidiagonalMatrix(
   Index rows,
   Index columns,
   const std::initializer_list< ListIndex > diagonalOffsets )
{
   if( std::empty( diagonalOffsets ) )
      throw std::invalid_argument( "Cannot construct multidiagonal matrix with no diagonal offsets." );
   DiagonalOffsetsType offsets( diagonalOffsets );
   this->setDimensions( rows, columns, offsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename ListIndex, typename ListReal >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::MultidiagonalMatrix(
   Index columns,
   const std::initializer_list< ListIndex > diagonalOffsets,
   const std::initializer_list< std::initializer_list< ListReal > >& data )
{
   if( std::empty( diagonalOffsets ) )
      throw std::invalid_argument( "Cannot construct multidiagonal matrix with no diagonal offsets." );
   DiagonalOffsetsType offsets( diagonalOffsets );
   this->setDimensions( data.size(), columns, offsets );
   this->setElements( data );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::getView() -> ViewType
{
   return { this->getValues().getView(), diagonalOffsets.getView(), hostDiagonalOffsets.getView(), this->getIndexer() };
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
auto
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::getConstView() const -> ConstViewType
{
   return {
      this->getValues().getConstView(), diagonalOffsets.getConstView(), hostDiagonalOffsets.getConstView(), this->getIndexer()
   };
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Vector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::setDimensions(
   Index rows,
   Index columns,
   const Vector& diagonalOffsets )
{
   this->diagonalOffsets = diagonalOffsets;
   this->hostDiagonalOffsets = diagonalOffsets;
   const Index minOffset = min( diagonalOffsets );
   Index nonemptyRows = min( rows, columns );
   if( rows > columns && minOffset < 0 )
      nonemptyRows = min( rows, nonemptyRows - minOffset );
   this->getIndexer().set( rows, columns, diagonalOffsets.getSize(), nonemptyRows );
   this->values.setSize( this->indexer.getStorageSize() );
   this->values = 0.0;
   // update the base
   Base::bind( values.getView(), this->diagonalOffsets.getView(), this->hostDiagonalOffsets.getView(), this->getIndexer() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Vector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::setDiagonalOffsets(
   const Vector& diagonalOffsets )
{
   if( diagonalOffsets.getSize() == 0 )
      throw std::invalid_argument( "Cannot construct multidiagonal matrix with no diagonal offsets." );
   this->setDimensions( this->getRows(), this->getColumns(), diagonalOffsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename ListIndex >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::setDiagonalOffsets(
   const std::initializer_list< ListIndex > diagonalOffsets )
{
   if( std::empty( diagonalOffsets ) )
      throw std::invalid_argument( "Cannot construct multidiagonal matrix with no diagonal offsets." );
   DiagonalOffsetsType offsets( diagonalOffsets );
   this->setDimensions( this->getRows(), this->getColumns(), offsets );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real_,
          typename Device_,
          typename Index_,
          ElementsOrganization Organization_,
          typename RealAllocator_,
          typename IndexAllocator_ >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::setLike(
   const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix )
{
   this->setDimensions( matrix.getRows(), matrix.getColumns(), matrix.getDiagonalOffsets() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename RowCapacitiesVector >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::setRowCapacities(
   const RowCapacitiesVector& rowCapacities )
{
   if( max( rowCapacities ) > 3 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( rowCapacities.getElement( 0 ) > 2 )
      throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   const Index diagonalLength = min( this->getRows(), this->getColumns() );
   if( this->getRows() > this->getColumns() )
      if( rowCapacities.getElement( this->getRows() - 1 ) > 1 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() == this->getColumns() )
      if( rowCapacities.getElement( this->getRows() - 1 ) > 2 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
   if( this->getRows() < this->getColumns() )
      if( rowCapacities.getElement( this->getRows() - 1 ) > 3 )
         throw std::logic_error( "Too many non-zero elements per row in a tri-diagonal matrix." );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename ListReal >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::setElements(
   const std::initializer_list< std::initializer_list< ListReal > >& data )
{
   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      this->getValues() = 0.0;
      auto row_it = data.begin();
      for( size_t rowIdx = 0; rowIdx < data.size(); rowIdx++ ) {
         auto data_it = row_it->begin();
         Index i = 0;
         while( data_it != row_it->end() )
            this->getRow( rowIdx ).setElement( i++, *data_it++ );
         row_it++;
      }
   }
   else {
      MultidiagonalMatrix< Real, Devices::Host, Index, Organization > hostMatrix(
         this->getRows(), this->getColumns(), this->getDiagonalOffsets() );
      hostMatrix.setElements( data );
      *this = hostMatrix;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::reset()
{
   this->setDimensions( 0, 0, DiagonalOffsetsType() );
}

template< typename InMatrixView, typename OutMatrixView, typename Real, typename Index >
__global__
void
MultidiagonalMatrixTranspositionCudaKernel( const InMatrixView inMatrix,
                                            OutMatrixView outMatrix,
                                            const Real matrixMultiplicator,
                                            const Index gridIdx )
{
#ifdef __CUDACC__
   const Index rowIdx = ( gridIdx * Cuda::getMaxGridXSize() + blockIdx.x ) * blockDim.x + threadIdx.x;
   if( rowIdx < inMatrix.getRows() ) {
      if( rowIdx > 0 )
         outMatrix.setElementFast( rowIdx - 1, rowIdx, matrixMultiplicator * inMatrix.getElementFast( rowIdx, rowIdx - 1 ) );
      outMatrix.setElementFast( rowIdx, rowIdx, matrixMultiplicator * inMatrix.getElementFast( rowIdx, rowIdx ) );
      if( rowIdx < inMatrix.getRows() - 1 )
         outMatrix.setElementFast( rowIdx + 1, rowIdx, matrixMultiplicator * inMatrix.getElementFast( rowIdx, rowIdx + 1 ) );
   }
#endif
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, typename Index2 >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::getTransposition(
   const MultidiagonalMatrix< Real2, Device, Index2 >& matrix,
   const Real& matrixMultiplicator )
{
   TNL_ASSERT_EQ( this->getRows(), matrix.getRows(), "The matrices must have the same number of rows." );

   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      const Index rows = matrix.getRows();
      for( Index i = 1; i < rows; i++ ) {
         Real aux = matrix.getElement( i, i - 1 );
         this->setElement( i, i - 1, matrix.getElement( i - 1, i ) );
         this->setElement( i, i, matrix.getElement( i, i ) );
         this->setElement( i - 1, i, aux );
      }
   }
   if constexpr( std::is_same_v< Device, Devices::Cuda > ) {
      Cuda::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      launch_config.gridSize.x = Cuda::getMaxGridXSize();
      const Index cudaBlocks = roundUpDivision( matrix.getRows(), launch_config.blockSize.x );
      const Index cudaGrids = roundUpDivision( cudaBlocks, launch_config.gridSize.x );
      for( Index gridIdx = 0; gridIdx < cudaGrids; gridIdx++ ) {
         if( gridIdx == cudaGrids - 1 )
            launch_config.gridSize.x = cudaBlocks % Cuda::getMaxGridXSize();
         constexpr auto kernel =
            MultidiagonalMatrixTranspositionCudaKernel< decltype( matrix.getConstView() ), ViewType, Real, Index >;
         Cuda::launchKernelAsync( kernel, launch_config, matrix.getConstView(), getView(), matrixMultiplicator, gridIdx );
      }
      cudaStreamSynchronize( launch_config.stream );
      TNL_CHECK_CUDA_DEVICE;
   }
}

// copy assignment
template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >&
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::operator=(
   const MultidiagonalMatrix& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   return *this;
}

// cross-device copy assignment
template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real_,
          typename Device_,
          typename Index_,
          ElementsOrganization Organization_,
          typename RealAllocator_,
          typename IndexAllocator_ >
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >&
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::operator=(
   const MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >& matrix )
{
   using RHSMatrix = MultidiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_, IndexAllocator_ >;
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;
   using RHSIndexAllocatorType = typename RHSMatrix::IndexAllocatorType;

   this->setLike( matrix );
   if( Organization == Organization_ )
      this->values = matrix.getValues();
   else {
      if( std::is_same_v< Device, Device_ > ) {
         const auto matrix_view = matrix.getConstView();
         auto f =
            [ = ] __cuda_callable__( const Index& rowIdx, const Index& localIdx, const Index& column, Real& value ) mutable
         {
            value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
         };
         this->forAllElements( f );
      }
      else {
         const Index maxRowLength = this->diagonalOffsets.getSize();
         const Index bufferRowsCount = 128;
         const size_t bufferSize = bufferRowsCount * maxRowLength;
         Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
         Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType, RHSIndexAllocatorType > matrixColumnsBuffer(
            bufferSize );
         Containers::Vector< Real, Device, Index, RealAllocatorType > thisValuesBuffer( bufferSize );
         Containers::Vector< Index, Device, Index, IndexAllocatorType > thisColumnsBuffer( bufferSize );
         auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
         auto thisValuesBuffer_view = thisValuesBuffer.getView();

         Index baseRow = 0;
         const Index rowsCount = this->getRows();
         while( baseRow < rowsCount ) {
            const Index lastRow = min( baseRow + bufferRowsCount, rowsCount );

            // Copy matrix elements into buffer
            auto f1 =
               [ = ] __cuda_callable__(
                  RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value ) mutable
            {
               const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
               matrixValuesBuffer_view[ bufferIdx ] = value;
            };
            matrix.forElements( baseRow, lastRow, f1 );

            // Copy the source matrix buffer to this matrix buffer
            thisValuesBuffer_view = matrixValuesBuffer_view;

            // Copy matrix elements from the buffer to the matrix
            auto f2 =
               [ = ] __cuda_callable__( const Index rowIdx, const Index localIdx, const Index columnIndex, Real& value ) mutable
            {
               const Index bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
               value = thisValuesBuffer_view[ bufferIdx ];
            };
            this->forElements( baseRow, lastRow, f2 );
            baseRow += bufferRowsCount;
         }
      }
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::save( File& file ) const
{
   file.save( &this->rows );
   file.save( &this->columns );
   file << values << diagonalOffsets;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::load( File& file )
{
   Index rows = 0;
   Index columns = 0;
   file.load( &rows );
   file.load( &columns );
   file >> values >> diagonalOffsets;

   hostDiagonalOffsets = diagonalOffsets;
   const Index minOffset = min( diagonalOffsets );
   Index nonemptyRows = min( rows, columns );
   if( rows > columns && minOffset < 0 )
      nonemptyRows = min( rows, nonemptyRows - minOffset );
   this->getIndexer().set( rows, columns, diagonalOffsets.getSize(), nonemptyRows );
   // update the base
   Base::bind( values.getView(), diagonalOffsets.getView(), hostDiagonalOffsets.getView(), this->getIndexer() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator,
          typename IndexAllocator >
void
MultidiagonalMatrix< Real, Device, Index, Organization, RealAllocator, IndexAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

}  // namespace TNL::Matrices
