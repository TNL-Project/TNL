// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "TridiagonalMatrix.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::TridiagonalMatrix( Index rows, Index columns )
{
   this->setDimensions( rows, columns );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::TridiagonalMatrix( const TridiagonalMatrix& matrix )
: values( matrix.values )
{
   // update the base
   Base::bind( values.getView(), matrix.getIndexer() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename ListReal >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::TridiagonalMatrix(
   Index columns,
   const std::initializer_list< std::initializer_list< ListReal > >& data )
{
   this->setDimensions( data.size(), columns );
   this->setElements( data );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
auto
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::getView() -> ViewType
{
   return { this->getValues().getView(), this->getIndexer() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
auto
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::getConstView() const -> ConstViewType
{
   return { this->getValues().getConstView(), this->getIndexer() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::setDimensions( Index rows, Index columns )
{
   this->getIndexer().setDimensions( rows, columns );
   this->values.setSize( this->indexer.getStorageSize() );
   this->values = 0.0;
   // update the base
   Base::bind( values.getView(), this->getIndexer() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RowCapacitiesVector >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::setRowCapacities(
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

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename ListReal >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::setElements(
   const std::initializer_list< std::initializer_list< ListReal > >& data )
{
   if( std::is_same_v< Device, Devices::Host > ) {
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
      TridiagonalMatrix< Real, Devices::Host, Index, Organization > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setElements( data );
      *this = hostMatrix;
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::setLike(
   const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& m )
{
   this->setDimensions( m.getRows(), m.getColumns() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::reset()
{
   this->setDimensions( 0, 0 );
}

template< typename InMatrixView, typename OutMatrixView, typename Real, typename Index >
__global__
void
TridiagonalMatrixTranspositionCudaKernel( const InMatrixView inMatrix,
                                          OutMatrixView outMatrix,
                                          Real matrixMultiplicator,
                                          Index gridIdx )
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

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Real2, typename Index2 >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::getTransposition(
   const TridiagonalMatrix< Real2, Device, Index2 >& matrix,
   const Real& matrixMultiplicator )
{
   TNL_ASSERT_EQ( this->getRows(), matrix.getRows(), "The matrices must have the same number of rows." );

   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      const Index& rows = matrix.getRows();
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
            TridiagonalMatrixTranspositionCudaKernel< decltype( matrix.getConstView() ), ViewType, Real, Index >;
         Cuda::launchKernelAsync( kernel, launch_config, matrix.getConstView(), getView(), matrixMultiplicator, gridIdx );
      }
      Backend::streamSynchronize( launch_config.stream );
   }
}

// copy assignment
template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >&
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::operator=( const TridiagonalMatrix& matrix )
{
   this->setLike( matrix );
   this->values = matrix.values;
   return *this;
}

// cross-device copy assignment
template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_, typename RealAllocator_ >
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >&
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const TridiagonalMatrix< Real_, Device_, Index_, Organization_, RealAllocator_ >& matrix )
{
   static_assert( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Cuda >, "unknown device" );
   static_assert( std::is_same_v< Device_, Devices::Host > || std::is_same_v< Device_, Devices::Cuda >, "unknown device" );

   this->setLike( matrix );
   if constexpr( Organization == Organization_ )
      this->values = matrix.getValues();
   else if constexpr( std::is_same_v< Device, Device_ > ) {
      const auto matrix_view = matrix.getConstView();
      auto f = [ = ] __cuda_callable__( const Index& rowIdx, const Index& localIdx, const Index& column, Real& value ) mutable
      {
         value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      this->forAllElements( f );
   }
   else {
      TridiagonalMatrix< Real, Device, Index, Organization_ > auxMatrix;
      auxMatrix = matrix;
      const auto matrix_view = auxMatrix.getView();
      auto f = [ = ] __cuda_callable__( const Index& rowIdx, const Index& localIdx, const Index& column, Real& value ) mutable
      {
         value = matrix_view.getValues()[ matrix_view.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      this->forAllElements( f );
   }
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::save( File& file ) const
{
   file.save( &this->rows );
   file.save( &this->columns );
   file << values;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::load( File& file )
{
   Index rows = 0;
   Index columns = 0;
   file.load( &rows );
   file.load( &columns );
   file >> values;
   typename Base::IndexerType indexer;
   indexer.setDimensions( rows, columns );
   // update the base
   Base::bind( values.getView(), indexer );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
TridiagonalMatrix< Real, Device, Index, Organization, RealAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

}  // namespace TNL::Matrices
