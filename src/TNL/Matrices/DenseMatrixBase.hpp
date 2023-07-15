// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <functional>

#include <TNL/Algorithms/reduce.h>
#include "DenseMatrixBase.h"

namespace TNL::Matrices {

// The following kernel is an attempt to map more CUDA threads to one matrix row.
template< int BlockSize, int ThreadsPerRow, typename Matrix, typename InVector, typename OutVector >
__global__
void
VectorColumnMajorDenseMatrixVectorMultiplicationKernel( const Matrix matrix,
                                                        const InVector inVector,
                                                        OutVector outVector,
                                                        int begin,
                                                        int end,
                                                        int gridIdx,
                                                        typename Matrix::RealType matrixMultiplicator,
                                                        typename Matrix::RealType outVectorMultiplicator )
{
#ifdef __CUDACC__
   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   constexpr int inVectorCacheSize = 20480 / sizeof( Real );
   __shared__ Real inVectorCache[ inVectorCacheSize ];
   __shared__ Real result_[ BlockSize ];

   constexpr Index rowsPerBlock = 256 / ThreadsPerRow;
   const Index rowIdx = ( ( gridIdx * Cuda::getMaxGridXSize() + blockIdx.x ) * 256 + threadIdx.x ) / ThreadsPerRow + begin;
   const Index localColIdx = threadIdx.x / rowsPerBlock;
   const Index localRowIdx = threadIdx.x % rowsPerBlock;

   Real result( 0.0 );
   Index columnIdx( 0 );
   const auto& values = matrix.getValues();
   const auto& rowsCount = matrix.getRows();
   Index valuesPtr = rowIdx + localColIdx * rowsCount;

   while( columnIdx < matrix.getColumns() ) {
      const Index lastIdx = min( matrix.getColumns(), columnIdx + inVectorCacheSize );
      Index matrixColIdx = columnIdx + threadIdx.x;
      Index cacheColIdx = threadIdx.x;
      while( matrixColIdx < lastIdx ) {
         inVectorCache[ cacheColIdx ] = inVector[ matrixColIdx ];
         cacheColIdx += 256;
         matrixColIdx += 256;
      }
      __syncthreads();

      matrixColIdx = columnIdx + localColIdx;
      cacheColIdx = localColIdx;
      if( rowIdx < end )
         while( matrixColIdx < lastIdx ) {
            result += values[ valuesPtr ] * inVectorCache[ cacheColIdx ];
            cacheColIdx += ThreadsPerRow;
            matrixColIdx += ThreadsPerRow;
            valuesPtr += ThreadsPerRow * rowsCount;
         }
      columnIdx = lastIdx;
   }
   const int idx = localRowIdx * ThreadsPerRow + localColIdx;
   result_[ idx ] = result;
   if( ThreadsPerRow > 8 && localColIdx < ThreadsPerRow - 8 )
      result_[ idx ] += result_[ idx + 8 ];
   __syncwarp();
   if( ThreadsPerRow > 4 && localColIdx < ThreadsPerRow - 4 )
      result_[ idx ] += result_[ idx + 4 ];
   __syncwarp();
   if( ThreadsPerRow > 2 && localColIdx < ThreadsPerRow - 2 )
      result_[ idx ] += result_[ idx + 2 ];
   __syncwarp();
   if( ThreadsPerRow > 1 && localColIdx < ThreadsPerRow - 1 )
      result_[ idx ] += result_[ idx + 1 ];
   __syncwarp();

   if( rowIdx < end && localColIdx == 0 ) {
      if( outVectorMultiplicator == 0 )
         outVector[ rowIdx ] = matrixMultiplicator * result_[ idx ];
      else
         outVector[ rowIdx ] = matrixMultiplicator * result_[ idx ] + outVectorMultiplicator * outVector[ rowIdx ];
   }
#endif
}

template< typename Matrix, typename InVector, typename OutVector >
__global__
void
ColumnMajorDenseMatrixVectorMultiplicationKernel( const Matrix matrix,
                                                  const InVector inVector,
                                                  OutVector outVector,
                                                  int begin,
                                                  int end,
                                                  int gridIdx,
                                                  typename Matrix::RealType matrixMultiplicator,
                                                  typename Matrix::RealType outVectorMultiplicator )
{
#ifdef __CUDACC__
   using Real = typename Matrix::RealType;
   using Index = typename Matrix::IndexType;
   constexpr int inVectorCacheSize = 20480 / sizeof( Real );
   __shared__ Real inVectorCache[ inVectorCacheSize ];

   const int rowIdx = ( gridIdx * Cuda::getMaxGridXSize() + blockIdx.x ) * 256 + threadIdx.x + begin;

   Real result = 0;
   Index columnIdx = 0;
   const auto& values = matrix.getValues();
   const auto& rowsCount = matrix.getRows();
   Index valuesPtr = rowIdx;

   while( columnIdx < matrix.getColumns() ) {
      const Index lastIdx = min( matrix.getColumns(), columnIdx + inVectorCacheSize );
      Index matrixColIdx = columnIdx + threadIdx.x;
      Index cacheColIdx = threadIdx.x;
      while( matrixColIdx < lastIdx ) {
         inVectorCache[ cacheColIdx ] = inVector[ matrixColIdx ];
         cacheColIdx += 256;
         matrixColIdx += 256;
      }
      __syncthreads();

      matrixColIdx = columnIdx;
      cacheColIdx = 0;
      if( rowIdx < end )
         while( matrixColIdx < lastIdx ) {
            result += values[ valuesPtr ] * inVectorCache[ cacheColIdx ];
            cacheColIdx++;
            matrixColIdx++;
            valuesPtr += rowsCount;
         }
      columnIdx = lastIdx;
   }

   if( rowIdx < end ) {
      if( outVectorMultiplicator == 0 )
         outVector[ rowIdx ] = matrixMultiplicator * result;
      else
         outVector[ rowIdx ] = matrixMultiplicator * result + outVectorMultiplicator * outVector[ rowIdx ];
   }
#endif
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixBase< Real, Device, Index, Organization >::bind( IndexType rows,
                                                            IndexType columns,
                                                            typename Base::ValuesViewType values,
                                                            SegmentsViewType segments )
{
   Base::bind( rows, columns, std::move( values ) );
   this->segments.bind( std::move( segments ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
DenseMatrixBase< Real, Device, Index, Organization >::DenseMatrixBase( IndexType rows,
                                                                       IndexType columns,
                                                                       typename Base::ValuesViewType values )
: Base( rows, columns, std::move( values ) ), segments( rows, columns )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::string
DenseMatrixBase< Real, Device, Index, Organization >::getSerializationType()
{
   return "Matrices::DenseMatrix< " + TNL::getSerializationType< RealType >() + ", [any_device], "
        + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization ) + " >";
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
DenseMatrixBase< Real, Device, Index, Organization >::getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return ( value != 0.0 );
   };
   auto keep = [ = ] __cuda_callable__( IndexType rowIdx, IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
DenseMatrixBase< Real, Device, Index, Organization >::getRowCapacities( Vector& rowCapacities ) const
{
   rowCapacities.setSize( this->getRows() );
   rowCapacities = this->getColumns();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
DenseMatrixBase< Real, Device, Index, Organization >::setValue( const Real& value )
{
   this->getValues() = value;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixBase< Real, Device, Index, Organization >::getRow( IndexType rowIdx ) const -> ConstRowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return { this->segments.getSegmentView( rowIdx ), this->getValues().getConstView() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixBase< Real, Device, Index, Organization >::getRow( IndexType rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return { this->segments.getSegmentView( rowIdx ), this->getValues().getView() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Real&
DenseMatrixBase< Real, Device, Index, Organization >::operator()( IndexType row, IndexType column )
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->getValues().operator[]( this->getElementIndex( row, column ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
const Real&
DenseMatrixBase< Real, Device, Index, Organization >::operator()( IndexType row, IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->getValues().operator[]( this->getElementIndex( row, column ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixBase< Real, Device, Index, Organization >::setElement( IndexType row, IndexType column, const RealType& value )
{
   this->getValues().setElement( this->getElementIndex( row, column ), value );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
DenseMatrixBase< Real, Device, Index, Organization >::addElement( IndexType row,
                                                                  IndexType column,
                                                                  const RealType& value,
                                                                  const RealType& thisElementMultiplicator )
{
   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->getValues().setElement( elementIndex, this->getValues().getElement( elementIndex ) + value );
   else
      this->getValues().setElement( elementIndex,
                                    thisElementMultiplicator * this->getValues().getElement( elementIndex ) + value );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Real
DenseMatrixBase< Real, Device, Index, Organization >::getElement( IndexType row, IndexType column ) const
{
   return this->getValues().getElement( this->getElementIndex( row, column ) );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
DenseMatrixBase< Real, Device, Index, Organization >::reduceRows( IndexType begin,
                                                                  IndexType end,
                                                                  Fetch& fetch,
                                                                  const Reduce& reduce,
                                                                  Keep& keep,
                                                                  const FetchValue& identity ) const
{
   const auto values = this->getValues().getConstView();
   auto fetch_ = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable
      -> decltype( fetch( IndexType(), IndexType(), RealType() ) )
   {
      return fetch( rowIdx, columnIdx, values[ globalIdx ] );
   };
   SegmentsReductionKernel::reduceSegments( this->segments, begin, end, fetch_, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
DenseMatrixBase< Real, Device, Index, Organization >::reduceAllRows( Fetch& fetch,
                                                                     const Reduce& reduce,
                                                                     Keep& keep,
                                                                     const FetchReal& identity ) const
{
   this->reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forElements( IndexType begin, IndexType end, Function&& function ) const
{
   const auto values = this->getValues().getConstView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx ) mutable
   {
      function( rowIdx, columnIdx, columnIdx, values[ globalIdx ] );
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forElements( IndexType begin, IndexType end, Function&& function )
{
   auto values = this->getValues().getView();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx ) mutable
   {
      function( rowIdx, columnIdx, globalIdx, values[ globalIdx ] );
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forAllElements( Function&& function ) const
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forAllElements( Function&& function )
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forRows( IndexType begin, IndexType end, Function&& function )
{
   auto values = this->getValues().getView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
   {
      auto rowView = RowView( segmentView, values );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forRows( IndexType begin, IndexType end, Function&& function ) const
{
   const auto values = this->getValues().getConstView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( const SegmentViewType& segmentView ) mutable
   {
      const auto rowView = ConstRowView( segmentView, values );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forAllRows( Function&& function )
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::forAllRows( Function&& function ) const
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::sequentialForRows( IndexType begin,
                                                                         IndexType end,
                                                                         Function&& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::sequentialForRows( IndexType begin, IndexType end, Function&& function )
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::sequentialForAllRows( Function&& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
DenseMatrixBase< Real, Device, Index, Organization >::sequentialForAllRows( Function&& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename InVector, typename OutVector >
void
DenseMatrixBase< Real, Device, Index, Organization >::vectorProduct( const InVector& inVector,
                                                                     OutVector& outVector,
                                                                     const RealType& matrixMultiplicator,
                                                                     const RealType& outVectorMultiplicator,
                                                                     IndexType begin,
                                                                     IndexType end ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns count differs with input vector size." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows count differs with output vector size." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->getValues().getConstView();
   if( end == 0 )
      end = this->getRows();

   // specialization for the case where we can use the CUDA shared memory
   if constexpr( std::is_same_v< DeviceType, Devices::Cuda > && Organization == Algorithms::Segments::ColumnMajorOrder ) {
      Cuda::LaunchConfiguration launch_config;
      launch_config.blockSize.x = 256;
      constexpr int ThreadsPerRow = 1;
      const std::size_t threadsCount = ( end - begin ) * ThreadsPerRow;
      const std::size_t blocksCount = roundUpDivision( threadsCount, launch_config.blockSize.x );
      const std::size_t gridsCount = roundUpDivision( blocksCount, Cuda::getMaxGridXSize() );
      for( std::size_t gridIdx = 0; gridIdx < gridsCount; gridIdx++ ) {
         launch_config.gridSize.x = Cuda::getMaxGridXSize();
         if( gridIdx == gridsCount - 1 )
            launch_config.gridSize.x = blocksCount % Cuda::getMaxGridXSize();
         constexpr auto kernel = ColumnMajorDenseMatrixVectorMultiplicationKernel< DenseMatrixBase,
                                                                                   decltype( inVectorView ),
                                                                                   decltype( outVectorView ) >;
         Cuda::launchKernelAsync( kernel,
                                  launch_config,
                                  *this,
                                  inVectorView,
                                  outVectorView,
                                  begin,
                                  end,
                                  gridIdx,
                                  matrixMultiplicator,
                                  outVectorMultiplicator );
      }
      cudaStreamSynchronize( launch_config.stream );
      TNL_CHECK_CUDA_DEVICE;
      return;
   }

   /***
    * The rest is general implementation based on segments
    */

   auto fetch = [ = ] __cuda_callable__( IndexType row, IndexType column, IndexType offset, bool& compute ) -> RealType
   {
      return valuesView[ offset ] * inVectorView[ column ];
   };
   auto keeperGeneral = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = matrixMultiplicator * value + outVectorMultiplicator * outVectorView[ row ];
   };
   auto keeperDirect = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = value;
   };
   auto keeperMatrixMult = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = matrixMultiplicator * value;
   };
   auto keeperVectorMult = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + value;
   };

   if( outVectorMultiplicator == 0.0 ) {
      if( matrixMultiplicator == 1.0 )
         SegmentsReductionKernel::reduceSegments(
            this->segments, begin, end, fetch, std::plus<>{}, keeperDirect, (RealType) 0.0 );
      else
         SegmentsReductionKernel::reduceSegments(
            this->segments, begin, end, fetch, std::plus<>{}, keeperMatrixMult, (RealType) 0.0 );
   }
   else {
      if( matrixMultiplicator == 1.0 )
         SegmentsReductionKernel::reduceSegments(
            this->segments, begin, end, fetch, std::plus<>{}, keeperVectorMult, (RealType) 0.0 );
      else
         SegmentsReductionKernel::reduceSegments(
            this->segments, begin, end, fetch, std::plus<>{}, keeperGeneral, (RealType) 0.0 );
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Matrix >
void
DenseMatrixBase< Real, Device, Index, Organization >::addMatrix( const Matrix& matrix,
                                                                 const RealType& matrixMultiplicator,
                                                                 const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT( this->getColumns() == matrix.getColumns() && this->getRows() == matrix.getRows(),
               std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                         << "This matrix rows: " << this->getRows() << std::endl
                         << "That matrix columns: " << matrix.getColumns() << std::endl
                         << "That matrix rows: " << matrix.getRows() << std::endl );

   if( thisMatrixMultiplicator == 1.0 )
      this->getValues() += matrixMultiplicator * matrix.getValues();
   else
      this->getValues() = thisMatrixMultiplicator * this->getValues() + matrixMultiplicator * matrix.getValues();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_ >
bool
DenseMatrixBase< Real, Device, Index, Organization >::operator==(
   const DenseMatrixBase< Real_, Device_, Index_, Organization >& matrix ) const
{
   return ( this->getRows() == matrix.getRows() && this->getColumns() == matrix.getColumns()
            && this->getValues() == matrix.getValues() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_ >
bool
DenseMatrixBase< Real, Device, Index, Organization >::operator!=(
   const DenseMatrixBase< Real_, Device_, Index_, Organization >& matrix ) const
{
   return ! ( *this == matrix );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
DenseMatrixBase< Real, Device, Index, Organization >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ ) {
      str << "Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ ) {
         std::stringstream str_;
         str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left
              << this->getElement( row, column );
         str << std::setw( 10 ) << str_.str();
      }
      if( row < this->getRows() - 1 )
         str << std::endl;
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Index
DenseMatrixBase< Real, Device, Index, Organization >::getElementIndex( IndexType row, IndexType column ) const
{
   return this->segments.getGlobalIndex( row, column );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::ostream&
operator<<( std::ostream& str, const DenseMatrixBase< Real, Device, Index, Organization >& matrix )
{
   matrix.print( str );
   return str;
}

}  // namespace TNL::Matrices
