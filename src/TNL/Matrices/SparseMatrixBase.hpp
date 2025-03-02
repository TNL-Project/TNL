// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>
#include <functional>

#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/AtomicOperations.h>
#include "SparseMatrixBase.h"
#include "details/SparseMatrix.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::bind( IndexType rows,
                                                                                      IndexType columns,
                                                                                      typename Base::ValuesViewType values,
                                                                                      ColumnIndexesViewType columnIndexes,
                                                                                      SegmentsViewType segments )
{
   Base::bind( rows, columns, std::move( values ) );
   this->columnIndexes.bind( std::move( columnIndexes ) );
   this->segments.bind( std::move( segments ) );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::SparseMatrixBase(
   IndexType rows,
   IndexType columns,
   typename Base::ValuesViewType values,
   ColumnIndexesViewType columnIndexes,
   SegmentsViewType segments )
: Base( rows, columns, values ),
  columnIndexes( std::move( columnIndexes ) ),
  segments( std::move( segments ) )
{}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
std::string
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getSerializationType()
{
   return "Matrices::SparseMatrix< " + TNL::getSerializationType< RealType >() + ", "
        + TNL::getSerializationType< SegmentsViewType >() + ", [any_device], " + TNL::getSerializationType< IndexType >() + ", "
        + MatrixType::getSerializationType() + ", [any_allocator], [any_allocator] >";
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Vector >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getCompressedRowLengths(
   Vector& rowLengths ) const
{
   details::set_size_if_resizable( rowLengths, this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return value != RealType{ 0 };
   };
   auto keep = [ = ] __cuda_callable__( IndexType rowIdx, IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, TNL::Plus{}, keep );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Vector >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRowCapacities( Vector& rowCapacities ) const
{
   details::set_size_if_resizable( rowCapacities, this->getRows() );
   rowCapacities = 0;
   auto rowCapacities_view = rowCapacities.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return 1;
   };
   auto keep = [ = ] __cuda_callable__( IndexType rowIdx, IndexType value ) mutable
   {
      rowCapacities_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, TNL::Plus{}, keep );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
Index
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRowCapacity( IndexType row ) const
{
   return this->segments.getSegmentSize( row );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
Index
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getNonzeroElementsCount() const
{
   if constexpr( ! Base::isSymmetric() )
      return sum( notEqualTo( this->getColumnIndexes(), paddingIndex< Index > )
                  && notEqualTo( this->getValues(), RealType{ 0 } ) );
   else {
      const auto rows = this->getRows();
      const auto columns = this->getColumns();
      Containers::Vector< IndexType, DeviceType, IndexType > row_sums( this->getRows(), 0 );
      auto row_sums_view = row_sums.getView();
      const auto columnIndexesView = this->columnIndexes.getConstView();
      auto fetch = [ = ] __cuda_callable__( IndexType row, IndexType localIdx, IndexType globalIdx ) -> IndexType
      {
         const IndexType column = columnIndexesView[ globalIdx ];
         if( column == paddingIndex< IndexType > )
            return 0;
         return 1 + ( column != row && column < rows && row < columns );  // the addition is for non-diagonal elements
      };
      auto keeper = [ = ] __cuda_callable__( IndexType row, const IndexType& value ) mutable
      {
         row_sums_view[ row ] = value;
      };
      DefaultSegmentsReductionKernel::reduceSegments(
         this->segments, 0, this->getRows(), fetch, std::plus<>{}, keeper, (IndexType) 0 );
      return sum( row_sums );
   }
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
auto
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRow( IndexType rowIdx ) const -> ConstRowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return ConstRowView( this->segments.getSegmentView( rowIdx ), this->values, this->columnIndexes );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
auto
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getRow( IndexType rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values, this->columnIndexes );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::setElement( IndexType row,
                                                                                            IndexType column,
                                                                                            const RealType& value )
{
   this->addElement( row, column, value, 0.0 );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::addElement(
   IndexType row,
   IndexType column,
   const RealType& value,
   const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "Sparse matrix row index cannot be negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Sparse matrix row index is larger than number of matrix rows." );
   TNL_ASSERT_GE( column, 0, "Sparse matrix column index cannot be negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Sparse matrix column index is larger than number of matrix columns." );

   if( Base::isSymmetric() && row < column ) {
      swap( row, column );
      TNL_ASSERT_LT( row, this->getRows(), "Column index is out of the symmetric part of the matrix after transposition." );
      TNL_ASSERT_LT( column, this->getColumns(), "Row index is out of the symmetric part of the matrix after transposition." );
   }

   const IndexType rowSize = this->segments.getSegmentSize( row );
   IndexType col = paddingIndex< IndexType >;
   IndexType i;
   IndexType globalIdx = 0;
   for( i = 0; i < rowSize; i++ ) {
      globalIdx = this->segments.getGlobalIndex( row, i );
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      col = this->columnIndexes.getElement( globalIdx );
      if( col == column ) {
         if( ! Base::isBinary() )
            this->values.setElement( globalIdx, thisElementMultiplicator * this->values.getElement( globalIdx ) + value );
         return;
      }
      if( col == paddingIndex< IndexType > || col > column )
         break;
   }
   if( i == rowSize ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      TNL_ASSERT_TRUE( false, "" );
      return;
#else
      std::stringstream msg;
      msg << "The capacity of the sparse matrix row number " << row << " was exceeded.";
      throw std::logic_error( msg.str() );
#endif
   }
   if( col == paddingIndex< IndexType > ) {
      this->columnIndexes.setElement( globalIdx, column );
      if( ! Base::isBinary() )
         this->values.setElement( globalIdx, value );
      return;
   }
   else {
      IndexType j = rowSize - 1;
      while( j > i ) {
         const IndexType globalIdx1 = this->segments.getGlobalIndex( row, j );
         const IndexType globalIdx2 = this->segments.getGlobalIndex( row, j - 1 );
         TNL_ASSERT_LT( globalIdx1, this->columnIndexes.getSize(), "" );
         TNL_ASSERT_LT( globalIdx2, this->columnIndexes.getSize(), "" );
         this->columnIndexes.setElement( globalIdx1, this->columnIndexes.getElement( globalIdx2 ) );
         if( ! Base::isBinary() )
            this->values.setElement( globalIdx1, this->values.getElement( globalIdx2 ) );
         j--;
      }

      this->columnIndexes.setElement( globalIdx, column );
      if( ! Base::isBinary() )
         this->values.setElement( globalIdx, value );
      return;
   }
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
auto
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getElement( IndexType row,
                                                                                            IndexType column ) const -> RealType
{
   TNL_ASSERT_GE( row, 0, "Sparse matrix row index cannot be negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Sparse matrix row index is larger than number of matrix rows." );
   TNL_ASSERT_GE( column, 0, "Sparse matrix column index cannot be negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Sparse matrix column index is larger than number of matrix columns." );

   if( Base::isSymmetric() && row < column ) {
      swap( row, column );
      if( row >= this->getRows() || column >= this->getColumns() )
         return 0;
   }

   const IndexType rowSize = this->segments.getSegmentSize( row );
   for( IndexType i = 0; i < rowSize; i++ ) {
      const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      const IndexType col = this->columnIndexes.getElement( globalIdx );
      if( col == column ) {
         if( Base::isBinary() )
            return 1;
         else
            return this->values.getElement( globalIdx );
      }
   }
   return 0;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename InVector, typename OutVector, typename SegmentsReductionKernel >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::vectorProduct(
   const InVector& inVector,
   OutVector& outVector,
   ComputeRealType matrixMultiplicator,
   ComputeRealType outVectorMultiplicator,
   IndexType begin,
   IndexType end,
   const SegmentsReductionKernel& kernel ) const
{
   if( this->getColumns() != inVector.getSize() )
      throw std::invalid_argument( "vectorProduct: size of the input vector does not match the number of matrix columns" );
   if( this->getRows() != outVector.getSize() )
      throw std::invalid_argument( "vectorProduct: size of the output vector does not match the number of matrix rows" );

   using OutVectorReal = typename OutVector::RealType;
   static_assert(
      ! MatrixType::isSymmetric() || ! std::is_same_v< Device, Devices::Cuda >
         || (std::is_same_v< OutVectorReal, float > || std::is_same_v< OutVectorReal, double >
             || std::is_same_v< OutVectorReal, int > || std::is_same_v< OutVectorReal, long long int >),
      "Given Real type is not supported by atomic operations on GPU which are necessary for symmetric operations." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   const auto columnIndexesView = this->columnIndexes.getConstView();

   if( end == 0 )
      end = this->getRows();

   if constexpr( Base::isSymmetric() ) {
      if( outVectorMultiplicator != ComputeRealType{ 1 } )
         outVector *= outVectorMultiplicator;
      auto fetch = [ valuesView, columnIndexesView, inVectorView, outVectorView, matrixMultiplicator ] __cuda_callable__(
                      IndexType row, IndexType localIdx, IndexType globalIdx ) mutable -> ComputeRealType
      {
         TNL_ASSERT_GE(
            globalIdx, 0, "Global index must be non-negative. Negative values may appear due to Index type overflow." );
         TNL_ASSERT_LT( globalIdx, columnIndexesView.getSize(), "Global index must be smaller than the number of elements." );
         const IndexType column = columnIndexesView[ globalIdx ];
         if( column == paddingIndex< IndexType > )
            return 0;
         if( column < row ) {
            if constexpr( Base::isBinary() )
               Algorithms::AtomicOperations< DeviceType >::add( outVectorView[ column ],
                                                                (OutVectorReal) matrixMultiplicator * inVectorView[ row ] );
            else
               Algorithms::AtomicOperations< DeviceType >::add( outVectorView[ column ],
                                                                (OutVectorReal) matrixMultiplicator * valuesView[ globalIdx ]
                                                                   * inVectorView[ row ] );
         }
         if constexpr( Base::isBinary() )
            return inVectorView[ column ];
         return valuesView[ globalIdx ] * inVectorView[ column ];
      };
      auto keep = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
      {
         typename OutVector::RealType aux = matrixMultiplicator * value;
         Algorithms::AtomicOperations< DeviceType >::add( outVectorView[ row ], aux );
      };
      kernel.reduceSegments( this->segments, begin, end, fetch, std::plus<>{}, keep, (ComputeRealType) 0.0 );
   }
   else {
      auto fetch =
         [ inVectorView, valuesView, columnIndexesView ] __cuda_callable__( IndexType globalIdx ) mutable -> ComputeRealType
      {
         TNL_ASSERT_GE(
            globalIdx, 0, "Global index must be non-negative. Negative values may appear due to Index type overflow." );
         TNL_ASSERT_LT( globalIdx, columnIndexesView.getSize(), "Global index must be smaller than the number of elements." );
         const IndexType column = columnIndexesView[ globalIdx ];
         TNL_ASSERT_TRUE( (column >= 0 || column == paddingIndex< Index >), "Wrong column index." );
         TNL_ASSERT_LT( column, inVectorView.getSize(), "Wrong column index." );
         if( column == paddingIndex< Index > )
            return 0;
         TNL_ASSERT_TRUE( column >= 0, "Wrong column index." );
         if constexpr( Base::isBinary() )
            return inVectorView[ column ];
         return valuesView[ globalIdx ] * inVectorView[ column ];
      };
      if( outVectorMultiplicator == ComputeRealType{ 0 } ) {
         if( matrixMultiplicator == ComputeRealType{ 1 } ) {
            auto keep = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
            {
               TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
               TNL_ASSERT_LT( row, outVectorView.getSize(), "Row index must be smaller than the number of elements." );
               outVectorView[ row ] = value;
            };
            kernel.reduceSegments( this->segments, begin, end, fetch, std::plus<>{}, keep, (ComputeRealType) 0.0 );
         }
         else {
            auto keep = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
            {
               TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
               outVectorView[ row ] = matrixMultiplicator * value;
            };
            kernel.reduceSegments( this->segments, begin, end, fetch, std::plus<>{}, keep, (ComputeRealType) 0.0 );
         }
      }
      else {
         if( matrixMultiplicator == ComputeRealType{ 1 } ) {
            auto keep = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
            {
               TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
               outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + value;
            };
            kernel.reduceSegments( this->segments, begin, end, fetch, std::plus<>{}, keep, (ComputeRealType) 0.0 );
         }
         else {
            auto keep = [ = ] __cuda_callable__( IndexType row, const ComputeRealType& value ) mutable
            {
               TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
               outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + matrixMultiplicator * value;
            };
            kernel.reduceSegments( this->segments, begin, end, fetch, std::plus<>{}, keep, (ComputeRealType) 0.0 );
         }
      }
   }
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename InVector,
          typename OutVector,
          typename SegmentsReductionKernel,
          typename...,
          std::enable_if_t< ! std::is_convertible_v< SegmentsReductionKernel, ComputeReal >, bool > >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::vectorProduct(
   const InVector& inVector,
   OutVector& outVector,
   const SegmentsReductionKernel& kernel ) const
{
   vectorProduct( inVector, outVector, 1.0, 0.0, 0, 0, kernel );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename InVector, typename OutVector >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::transposedVectorProduct(
   const InVector& inVector,
   OutVector& outVector,
   ComputeRealType matrixMultiplicator,
   ComputeRealType outVectorMultiplicator,
   IndexType begin,
   IndexType end ) const
{
   if( this->getRows() != inVector.getSize() )
      throw std::invalid_argument(
         "transposedVectorProduct: size of the input vector does not match the number of matrix rows" );
   if( this->getColumns() != outVector.getSize() )
      throw std::invalid_argument(
         "transposedVectorProduct: size of the output vector does not match the number of matrix columns" );

   if constexpr( MatrixType::isSymmetric() ) {
      this->vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, begin, end );
      return;
   }

   using OutVectorReal = typename OutVector::RealType;
   static_assert(
      ! std::is_same< Device, Devices::Cuda >::value
         || ( std::is_same< OutVectorReal, float >::value || std::is_same< OutVectorReal, double >::value
              || std::is_same< OutVectorReal, int >::value || std::is_same< OutVectorReal, long long int >::value
              || std::is_same< OutVectorReal, long >::value ),
      "Given Real type is not supported by atomic operations on GPU which are necessary for symmetric operations." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();

   if( end == 0 )
      end = this->getColumns();

   if( outVectorMultiplicator != ComputeRealType{ 1 } )
      outVector *= outVectorMultiplicator;
   auto compute = [ inVectorView, outVectorView, matrixMultiplicator, begin, end ] __cuda_callable__(
                     IndexType row, IndexType localIdx, IndexType column, const RealType& value ) mutable
   {
      if( column >= begin && column < end ) {
         if( column != paddingIndex< IndexType > )
            Algorithms::AtomicOperations< DeviceType >::add(
               outVectorView[ column ], (OutVectorReal) matrixMultiplicator * inVectorView[ row ] * value );
      }
   };
   this->forAllElements( compute );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue, typename SegmentsReductionKernel >
std::enable_if_t< Algorithms::SegmentsReductionKernels::isSegmentReductionKernel< SegmentsReductionKernel >::value >
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceRows(
   IndexType begin,
   IndexType end,
   Fetch&& fetch,
   const Reduce& reduce,
   Keep&& keep,
   const FetchValue& identity,
   const SegmentsReductionKernel& kernel ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   auto fetch_ = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable
      -> decltype( fetch( IndexType(), IndexType(), RealType() ) )
   {
      TNL_ASSERT_LT( globalIdx, (IndexType) columns_view.getSize(), "" );
      IndexType columnIdx = columns_view[ globalIdx ];
      if( columnIdx != paddingIndex< IndexType > ) {
         if( Base::isBinary() )
            return fetch( rowIdx, columnIdx, 1 );
         else
            return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      }
      return identity;
   };
   kernel.reduceSegments( this->segments, begin, end, fetch_, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename SegmentsReductionKernel >
std::enable_if_t< Algorithms::SegmentsReductionKernels::isSegmentReductionKernel< SegmentsReductionKernel >::value >
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceRows(
   IndexType begin,
   IndexType end,
   Fetch&& fetch,
   const Reduce& reduce,
   Keep&& keep,
   const SegmentsReductionKernel& kernel ) const
{
   using FetchValue = decltype( fetch( IndexType(), IndexType(), RealType() ) );
   this->reduceRows( begin,
                     end,
                     std::forward< Fetch >( fetch ),
                     reduce,
                     std::forward< Keep >( keep ),
                     reduce.template getIdentity< FetchValue >(),
                     kernel );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue, typename SegmentsReductionKernel >
std::enable_if_t< Algorithms::SegmentsReductionKernels::isSegmentReductionKernel< SegmentsReductionKernel >::value >
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceAllRows(
   Fetch&& fetch,
   const Reduce& reduce,
   Keep&& keep,
   const FetchValue& identity,
   const SegmentsReductionKernel& kernel ) const
{
   this->reduceRows( (IndexType) 0, this->getRows(), fetch, reduce, keep, identity, kernel );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Fetch, typename Reduce, typename Keep, typename SegmentsReductionKernel >
std::enable_if_t< Algorithms::SegmentsReductionKernels::isSegmentReductionKernel< SegmentsReductionKernel >::value >
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::reduceAllRows(
   Fetch&& fetch,
   const Reduce& reduce,
   Keep&& keep,
   const SegmentsReductionKernel& kernel ) const
{
   using FetchValue = decltype( fetch( IndexType(), IndexType(), RealType() ) );
   this->reduceAllRows( fetch, reduce, keep, reduce.template getIdentity< FetchValue >(), kernel );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forElements( IndexType begin,
                                                                                             IndexType end,
                                                                                             Function&& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   auto columns = this->getColumns();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable
   {
      if( localIdx < columns ) {
         if( Base::isBinary() )
            function( rowIdx, localIdx, columns_view[ globalIdx ], (RealType) 1.0 );
         else
            function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
      }
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forElements( IndexType begin,
                                                                                             IndexType end,
                                                                                             Function&& function )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   auto columns = this->getColumns();
   auto f = [ = ] __cuda_callable__( IndexType rowIdx, IndexType localIdx, IndexType globalIdx ) mutable
   {
      if( localIdx < columns ) {
         if( Base::isBinary() ) {
            RealType one = columns_view[ globalIdx ] != paddingIndex< IndexType >;
            function( rowIdx, localIdx, columns_view[ globalIdx ], one );
         }
         else
            function( rowIdx, localIdx, columns_view[ globalIdx ], values_view[ globalIdx ] );
      }
   };
   this->segments.forElements( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllElements( Function&& function ) const
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllElements( Function&& function )
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forRows( IndexType begin,
                                                                                         IndexType end,
                                                                                         Function&& function )
{
   auto columns_view = this->columnIndexes.getView();
   auto values_view = this->values.getView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( SegmentViewType & segmentView ) mutable
   {
      auto rowView = RowView( segmentView, values_view, columns_view );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forRows( IndexType begin,
                                                                                         IndexType end,
                                                                                         Function&& function ) const
{
   const auto columns_view = this->columnIndexes.getConstView();
   const auto values_view = this->values.getConstView();
   using SegmentViewType = typename SegmentsViewType::SegmentViewType;
   auto f = [ = ] __cuda_callable__( const SegmentViewType& segmentView ) mutable
   {
      const auto rowView = ConstRowView( segmentView, values_view, columns_view );
      function( rowView );
   };
   this->segments.forSegments( begin, end, f );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllRows( Function&& function )
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::forAllRows( Function&& function ) const
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForRows( IndexType begin,
                                                                                                   IndexType end,
                                                                                                   Function&& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForRows( IndexType begin,
                                                                                                   IndexType end,
                                                                                                   Function&& function )
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForAllRows(
   Function&& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Function >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sequentialForAllRows( Function&& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Matrix >
bool
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::operator==( const Matrix& m ) const
{
   const auto& view1 = *this;
   const auto view2 = m.getConstView();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> bool
   {
      return view1.getRow( i ) == view2.getRow( i );
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->getRows(), fetch, std::logical_and<>{}, true );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
template< typename Matrix >
bool
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::operator!=( const Matrix& m ) const
{
   return ! operator==( m );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::sortColumnIndexes()
{
   this->forAllRows(
      [ = ] __cuda_callable__( RowView & row )
      {
         row.sortColumnIndexes();
      } );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
__cuda_callable__
Index
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::findElement( IndexType row,
                                                                                             IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "Sparse matrix row index cannot be negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Sparse matrix row index is larger than number of matrix rows." );
   TNL_ASSERT_GE( column, 0, "Sparse matrix column index cannot be negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Sparse matrix column index is larger than number of matrix columns." );

   if( Base::isSymmetric() && row < column ) {
      swap( row, column );
      if( row >= this->getRows() || column >= this->getColumns() )
         return paddingIndex< IndexType >;
   }

   const IndexType rowSize = this->segments.getSegmentSize( row );
   for( IndexType i = 0; i < rowSize; i++ ) {
      const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
      TNL_ASSERT_LT( globalIdx, this->columnIndexes.getSize(), "" );
      const IndexType col = this->columnIndexes.getElement( globalIdx );
      if( col == column )
         return globalIdx;
   }
   return paddingIndex< IndexType >;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
void
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::print( std::ostream& str ) const
{
   if constexpr( Base::isSymmetric() ) {
      for( IndexType row = 0; row < this->getRows(); row++ ) {
         str << "Row: " << row << " -> ";
         for( IndexType column = 0; column < this->getColumns(); column++ ) {
            auto value = this->getElement( row, column );
            if( value != (RealType) 0 )
               str << column << ":" << value << "\t";
         }
         str << std::endl;
      }
   }
   else {
      for( IndexType row = 0; row < this->getRows(); row++ ) {
         str << "Row: " << row << " -> ";
         const auto rowLength = this->segments.getSegmentSize( row );
         for( IndexType i = 0; i < rowLength; i++ ) {
            const IndexType globalIdx = this->segments.getGlobalIndex( row, i );
            const IndexType column = this->columnIndexes.getElement( globalIdx );
            if( column == paddingIndex< IndexType > )
               break;
            RealType value;
            if( Base::isBinary() )
               value = (RealType) 1.0;
            else
               value = this->values.getElement( globalIdx );
            if( value != (RealType) 0 ) {
               std::stringstream str_;
               str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << value;
               str << std::setw( 10 ) << str_.str();
            }
         }
         str << std::endl;
      }
   }
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
auto
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getSegments() const -> const SegmentsViewType&
{
   return this->segments;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
auto
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getSegments() -> SegmentsViewType&
{
   return this->segments;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
auto
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getColumnIndexes() const
   -> const ColumnIndexesViewType&
{
   return this->columnIndexes;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
auto
SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >::getColumnIndexes() -> ColumnIndexesViewType&
{
   return this->columnIndexes;
}

}  // namespace TNL::Matrices
