// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <iomanip>

#include <TNL/Assert.h>
#include "MultidiagonalMatrixBase.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::bind( typename Base::ValuesViewType values,
                                                                    DiagonalOffsetsView diagonalOffsets,
                                                                    HostDiagonalOffsetsView hostDiagonalOffsets,
                                                                    IndexerType indexer )
{
   Base::bind( indexer.getRows(), indexer.getColumns(), std::move( values ) );
   this->diagonalOffsets.bind( std::move( diagonalOffsets ) );
   this->hostDiagonalOffsets.bind( std::move( hostDiagonalOffsets ) );
   this->indexer = std::move( indexer );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
MultidiagonalMatrixBase< Real, Device, Index, Organization >::MultidiagonalMatrixBase(
   typename Base::ValuesViewType values,
   DiagonalOffsetsView diagonalOffsets,
   HostDiagonalOffsetsView hostDiagonalOffsets,
   IndexerType indexer )
: Base( indexer.getRows(), indexer.getColumns(), std::move( values ) ),
  diagonalOffsets( std::move( diagonalOffsets ) ),
  hostDiagonalOffsets( std::move( hostDiagonalOffsets ) ),
  indexer( std::move( indexer ) )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
std::string
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getSerializationType()
{
   return "Matrices::MultidiagonalMatrix< " + TNL::getSerializationType< RealType >() + ", [any_device], "
        + TNL::getSerializationType< IndexType >() + ", " + TNL::getSerializationType( Organization )
        + ", [any_allocator], [any_allocator] >";
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
Index
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getDiagonalsCount() const
{
   return this->diagonalOffsets.getSize();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getRowCapacities( Vector& rowCapacities ) const
{
   rowCapacities.setSize( this->getRows() );
   rowCapacities = this->getDiagonalsCount();
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Vector >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__( IndexType row, IndexType column, const RealType& value ) -> IndexType
   {
      return value != 0.0;
   };
   auto reduce = [] __cuda_callable__( IndexType & aux, IndexType a )
   {
      aux += a;
   };
   auto keep = [ = ] __cuda_callable__( IndexType rowIdx, IndexType value ) mutable
   {
      rowLengths_view[ rowIdx ] = value;
   };
   this->reduceAllRows( fetch, reduce, keep, 0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
Index
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [ = ] __cuda_callable__( IndexType i ) -> IndexType
   {
      return values_view[ i ] != 0.0;
   };
   return Algorithms::reduce< DeviceType >( (IndexType) 0, this->values.getSize(), fetch, std::plus<>{}, 0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
bool
MultidiagonalMatrixBase< Real, Device, Index, Organization >::operator==(
   const MultidiagonalMatrixBase< Real_, Device_, Index_, Organization_ >& matrix ) const
{
   static_assert( Organization == Organization_,
                  "comparison of multidiagonal matrices with different organizations is not implemented" );
   return this->values == matrix.values;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
bool
MultidiagonalMatrixBase< Real, Device, Index, Organization >::operator!=(
   const MultidiagonalMatrixBase< Real_, Device_, Index_, Organization_ >& matrix ) const
{
   return ! this->operator==( matrix );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::setValue( const RealType& v )
{
   // we dont do this->values = v here because it would set even elements 'outside' the matrix
   // method getNumberOfNonzeroElements would not work well then
   const RealType newValue = v;
   auto f = [ = ] __cuda_callable__(
               const IndexType& rowIdx, const IndexType& localIdx, const IndexType columnIdx, RealType& value ) mutable
   {
      value = newValue;
   };
   this->forAllElements( f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getRow( IndexType rowIdx ) const -> ConstRowView
{
   return { rowIdx, this->diagonalOffsets.getConstView(), this->values.getConstView(), this->indexer };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getRow( IndexType rowIdx ) -> RowView
{
   return { rowIdx, this->diagonalOffsets.getView(), this->values.getView(), this->indexer };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::setElement( IndexType row,
                                                                          IndexType column,
                                                                          const RealType& value )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   for( IndexType i = 0; i < diagonalOffsets.getSize(); i++ )
      if( row + diagonalOffsets.getElement( i ) == column ) {
         this->values.setElement( this->indexer.getGlobalIndex( row, i ), value );
         return;
      }
   if( value != 0.0 ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      TNL_ASSERT_TRUE( false, "" );
#else
      throw std::logic_error( "Wrong matrix element coordinates ( " + std::to_string( row ) + ", " + std::to_string( column )
                              + " ) in multidiagonal matrix." );
#endif
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::addElement( IndexType row,
                                                                          IndexType column,
                                                                          const RealType& value,
                                                                          const RealType& thisElementMultiplicator )
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   for( IndexType i = 0; i < diagonalOffsets.getSize(); i++ )
      if( row + diagonalOffsets.getElement( i ) == column ) {
         const Index idx = this->indexer.getGlobalIndex( row, i );
         this->values.setElement( idx, thisElementMultiplicator * this->values.getElement( idx ) + value );
         return;
      }
   if( value != 0.0 ) {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      TNL_ASSERT_TRUE( false, "" );
#else
      throw std::logic_error( "Wrong matrix element coordinates ( " + std::to_string( row ) + ", " + std::to_string( column )
                              + " ) in multidiagonal matrix." );
#endif
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getElement( IndexType row, IndexType column ) const -> RealType
{
   TNL_ASSERT_GE( row, 0, "" );
   TNL_ASSERT_LT( row, this->getRows(), "" );
   TNL_ASSERT_GE( column, 0, "" );
   TNL_ASSERT_LT( column, this->getColumns(), "" );

   for( IndexType localIdx = 0; localIdx < diagonalOffsets.getSize(); localIdx++ )
      if( row + diagonalOffsets.getElement( localIdx ) == column )
         return this->values.getElement( this->indexer.getGlobalIndex( row, localIdx ) );
   return 0;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::reduceRows( IndexType begin,
                                                                          IndexType end,
                                                                          Fetch&& fetch,
                                                                          const Reduce& reduce,
                                                                          Keep&& keep,
                                                                          const FetchReal& identity ) const
{
   using Real_ = decltype( fetch( IndexType(), IndexType(), RealType() ) );
   const auto values_view = this->values.getConstView();
   const auto diagonalOffsets_view = this->diagonalOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      Real_ sum = identity;
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
         const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            reduce( sum, fetch( rowIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] ) );
      }
      keep( rowIdx, sum );
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::reduceAllRows( Fetch&& fetch,
                                                                             const Reduce& reduce,
                                                                             Keep&& keep,
                                                                             const FetchReal& identity ) const
{
   this->reduceRows( (IndexType) 0, this->indexer.getNonemptyRowsCount(), fetch, reduce, keep, identity );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElements( IndexType begin,
                                                                           IndexType end,
                                                                           Function&& function ) const
{
   const auto values_view = this->values.getConstView();
   const auto diagonalOffsets_view = this->diagonalOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
         const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
      }
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElements( IndexType begin, IndexType end, Function&& function )
{
   auto values_view = this->values.getView();
   const auto diagonalOffsets_view = this->diagonalOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
         const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
      }
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forAllElements( Function&& function ) const
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forAllElements( Function&& function )
{
   this->forElements( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Array, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElements( const Array& rowIndexes,
                                                                           IndexType begin,
                                                                           IndexType end,
                                                                           Function&& function ) const
{
   const auto values_view = this->values.getConstView();
   const auto diagonalOffsets_view = this->diagonalOffsets.getConstView();
   const auto rowIndexes_view = rowIndexes.getConstView();
   const IndexType diagonalsCount = this->diagonalOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
   {
      IndexType rowIdx = rowIndexes_view[ idx ];
      TNL_ASSERT_GE( rowIdx, 0, "" );
      TNL_ASSERT_LT( rowIdx, this->getRows(), "" );
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
         const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
      }
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Array, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElements( const Array& rowIndexes,
                                                                           IndexType begin,
                                                                           IndexType end,
                                                                           Function&& function )
{
   auto values_view = this->values.getView();
   const auto diagonalOffsets_view = this->diagonalOffsets.getConstView();
   auto rowIndexes_view = rowIndexes.getConstView();
   const IndexType diagonalsCount = this->diagonalOffsets.getSize();
   const IndexType columns = this->getColumns();
#ifndef NDEBUG  // we need rows for assertions in the lambda function f
   const IndexType rows = this->getRows();
#endif
   const auto indexer = this->indexer;
   std::cout << rowIndexes_view << '\n';
   auto f = [ = ] __cuda_callable__( IndexType idx ) mutable
   {
      TNL_ASSERT_LT( idx, rowIndexes_view.getSize(), "Index out of bounds." );
      TNL_ASSERT_GE( idx, 0, "Index out of bounds." );
      IndexType rowIdx = rowIndexes_view[ idx ];
      TNL_ASSERT_GE( rowIdx, 0, "" );
      TNL_ASSERT_LT( rowIdx, rows, "" );
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
         const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns ) {
            TNL_ASSERT_LT( indexer.getGlobalIndex( rowIdx, localIdx ),
                           values_view.getSize(),
                           "Global index is larger than number of matrix elements." );
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
         }
      }
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Array, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElements( const Array& rowIndexes, Function&& function ) const
{
   this->forElements( rowIndexes, (Index) 0, rowIndexes.getSize(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Array, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElements( const Array& rowIndexes, Function&& function )
{
   this->forElements( rowIndexes, (Index) 0, rowIndexes.getSize(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Condition, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElementsIf( IndexType begin,
                                                                             IndexType end,
                                                                             Condition&& condition,
                                                                             Function&& function ) const
{
   const auto values_view = this->values.getConstView();
   const auto diagonalOffsets_view = this->diagonalOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      if( ! condition( rowIdx ) )
         return;
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
         const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
      }
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Condition, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forElementsIf( IndexType begin,
                                                                             IndexType end,
                                                                             Condition&& condition,
                                                                             Function&& function )
{
   auto values_view = this->values.getView();
   const auto diagonalOffsets_view = this->diagonalOffsets.getConstView();
   const IndexType diagonalsCount = this->diagonalOffsets.getSize();
   const IndexType columns = this->getColumns();
   const auto indexer = this->indexer;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      if( ! condition( rowIdx ) )
         return;
      for( IndexType localIdx = 0; localIdx < diagonalsCount; localIdx++ ) {
         const IndexType columnIdx = rowIdx + diagonalOffsets_view[ localIdx ];
         if( columnIdx >= 0 && columnIdx < columns )
            function( rowIdx, localIdx, columnIdx, values_view[ indexer.getGlobalIndex( rowIdx, localIdx ) ] );
      }
   };
   Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Condition, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forAllElementsIf( Condition&& condition,
                                                                                Function&& function ) const
{
   this->forElementsIf( (IndexType) 0, this->getRows(), condition, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Condition, typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forAllElementsIf( Condition&& condition, Function&& function )
{
   this->forElementsIf( (IndexType) 0, this->getRows(), condition, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forRows( IndexType begin, IndexType end, Function&& function )
{
   auto view = *this;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      auto rowView = view.getRow( rowIdx );
      function( rowView );
   };
   TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forRows( IndexType begin,
                                                                       IndexType end,
                                                                       Function&& function ) const
{
   auto view = *this;
   auto f = [ = ] __cuda_callable__( IndexType rowIdx ) mutable
   {
      auto rowView = view.getRow( rowIdx );
      function( rowView );
   };
   TNL::Algorithms::parallelFor< DeviceType >( begin, end, f );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forAllRows( Function&& function )
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::forAllRows( Function&& function ) const
{
   this->forRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::sequentialForRows( IndexType begin,
                                                                                 IndexType end,
                                                                                 Function& function ) const
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::sequentialForRows( IndexType begin,
                                                                                 IndexType end,
                                                                                 Function& function )
{
   for( IndexType row = begin; row < end; row++ )
      this->forRows( row, row + 1, function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::sequentialForAllRows( Function& function ) const
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Function >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::sequentialForAllRows( Function& function )
{
   this->sequentialForRows( (IndexType) 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename InVector, typename OutVector >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::vectorProduct( const InVector& inVector,
                                                                             OutVector& outVector,
                                                                             RealType matrixMultiplicator,
                                                                             RealType outVectorMultiplicator,
                                                                             IndexType begin,
                                                                             IndexType end ) const
{
   if( this->getColumns() != inVector.getSize() )
      throw std::invalid_argument( "vectorProduct: size of the input vector does not match the number of matrix columns" );
   if( this->getRows() != outVector.getSize() )
      throw std::invalid_argument( "vectorProduct: size of the output vector does not match the number of matrix rows" );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   auto fetch = [ = ] __cuda_callable__( const IndexType& row, const IndexType& column, const RealType& value ) -> RealType
   {
      return value * inVectorView[ column ];
   };
   auto reduction = [] __cuda_callable__( RealType & sum, const RealType& value )
   {
      sum += value;
   };
   auto keeper1 = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = matrixMultiplicator * value;
   };
   auto keeper2 = [ = ] __cuda_callable__( IndexType row, const RealType& value ) mutable
   {
      outVectorView[ row ] = outVectorMultiplicator * outVectorView[ row ] + matrixMultiplicator * value;
   };

   if( end == 0 )
      end = this->getRows();
   if( outVectorMultiplicator == (RealType) 0.0 )
      this->reduceRows( begin, end, fetch, reduction, keeper1, (RealType) 0.0 );
   else
      this->reduceRows( begin, end, fetch, reduction, keeper2, (RealType) 0.0 );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization_ >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::addMatrix(
   const MultidiagonalMatrixBase< Real_, Device_, Index_, Organization_ >& matrix,
   const RealType& matrixMultiplicator,
   const RealType& thisMatrixMultiplicator )
{
   if( this->getRows() != matrix.getRows() )
      throw std::invalid_argument( "addMatrix: numbers of matrix rows are not equal" );
   if( this->getColumns() != matrix.getColumns() )
      throw std::invalid_argument( "addMatrix: numbers of matrix columns are not equal" );

   /*if( Organization == Organization_ )
   {
      if( thisMatrixMultiplicator == 1 )
         this->values += matrixMultiplicator * matrix.getValues();
      else
         this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.getValues();
   }
   else
   {
      const auto matrix_view = matrix;
      const auto matrixMult = matrixMultiplicator;
      const auto thisMult = thisMatrixMultiplicator;
      auto add0 = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real&
   value ) mutable { value = matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      auto add1 = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real&
   value ) mutable { value += matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx, localIdx ) ];
      };
      auto addGen = [=] __cuda_callable__ ( const IndexType& rowIdx, const IndexType& localIdx, const IndexType& column, Real&
   value ) mutable { value = thisMult * value + matrixMult * matrix.getValues()[ matrix.getIndexer().getGlobalIndex( rowIdx,
   localIdx ) ];
      };
      if( thisMult == 0 )
         this->forAllElements( add0 );
      else if( thisMult == 1 )
         this->forAllElements( add1 );
      else
         this->forAllElements( addGen );
   }*/
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
void
MultidiagonalMatrixBase< Real, Device, Index, Organization >::print( std::ostream& str ) const
{
   for( IndexType rowIdx = 0; rowIdx < this->getRows(); rowIdx++ ) {
      str << "Row: " << rowIdx << " -> ";
      for( IndexType localIdx = 0; localIdx < this->hostDiagonalOffsets.getSize(); localIdx++ ) {
         const IndexType columnIdx = rowIdx + this->hostDiagonalOffsets[ localIdx ];
         if( columnIdx >= 0 && columnIdx < this->columns ) {
            auto value = this->values.getElement( this->indexer.getGlobalIndex( rowIdx, localIdx ) );
            if( value ) {
               std::stringstream str_;
               str_ << std::setw( 4 ) << std::right << columnIdx << ":" << std::setw( 4 ) << std::left << value;
               str << std::setw( 10 ) << str_.str();
            }
         }
      }
      str << '\n';
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getIndexer() const -> const IndexerType&
{
   return this->indexer;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getIndexer() -> IndexerType&
{
   return this->indexer;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getDiagonalOffsets() const ->
   typename DiagonalOffsetsView::ConstViewType
{
   return this->diagonalOffsets;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
__cuda_callable__
auto
MultidiagonalMatrixBase< Real, Device, Index, Organization >::getDiagonalOffsets() -> DiagonalOffsetsView
{
   return this->diagonalOffsets;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator<<( File& file, const MultidiagonalMatrixBase< Real, Device, Index, Organization >& matrix )
{
   saveObjectType( file, matrix.getSerializationType() );
   const std::size_t rows = matrix.getRows();
   const std::size_t columns = matrix.getColumns();
   file.save( &rows );
   file.save( &columns );
   file << matrix.getDiagonalOffsets() << matrix.getValues();
   return file;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
File&
operator<<( File&& file, const MultidiagonalMatrixBase< Real, Device, Index, Organization >& matrix )
{
   // named r-value is an l-value reference, so this is not recursion
   return file << matrix;
}

}  // namespace TNL::Matrices
