// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include <functional>
#include <sstream>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Matrices/Sandbox/SparseSandboxMatrix.h>
#include <TNL/Matrices/SparseOperations.h>

namespace TNL::Matrices::Sandbox {

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ), columnIndexes( indexAllocator ), rowPointers( (IndexType) 1, (IndexType) 0, indexAllocator )
{
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Index_t, std::enable_if_t< std::is_integral< Index_t >::value, int > >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const Index_t rows,
   const Index_t columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: rows( rows ), columns( columns ), values( realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rows + 1, (IndexType) 0, indexAllocator )
{
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename ListIndex >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const std::initializer_list< ListIndex >& rowCapacities,
   const IndexType columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: rows( rowCapacities.size() ), columns( columns ), values( realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rowCapacities.size() + 1, (IndexType) 0, indexAllocator )
{
   this->setRowCapacities( RowsCapacitiesType( rowCapacities ) );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename RowCapacitiesVector, std::enable_if_t< TNL::IsArrayType< RowCapacitiesVector >::value, int > >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const RowCapacitiesVector& rowCapacities,
   const IndexType columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: rows( rowCapacities.size() ), columns( columns ), values( realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rowCapacities.getSize() + 1, (IndexType) 0, indexAllocator )
{
   this->setRowCapacities( rowCapacities );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const IndexType rows,
   const IndexType columns,
   const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: rows( rows ), columns( columns ), values( realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rows + 1, (IndexType) 0, indexAllocator )
{
   this->setElements( data );
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename MapIndex, typename MapValue >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::SparseSandboxMatrix(
   const IndexType rows,
   const IndexType columns,
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: rows( rows ), columns( columns ), values( realAllocator ), columnIndexes( indexAllocator ),
  rowPointers( rows + 1, (IndexType) 0, indexAllocator )
{
   this->setDimensions( rows, columns );
   this->setElements( map );
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
Index
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRows() const
{
   return this->rows;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
Index
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getColumns() const
{
   return this->columns;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getValues() const
   -> const ValuesVectorType&
{
   return this->values;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getValues() -> ValuesVectorType&
{
   return this->values;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getView() -> ViewType
{
   return { this->getRows(), this->getColumns(), this->getValues().getView(), columnIndexes.getView(), rowPointers.getView() };
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getConstView() const -> ConstViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getConstView(),
            columnIndexes.getConstView(),
            rowPointers.getConstView() };
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
std::string
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getSerializationType()
{
   return ViewType::getSerializationType();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
std::string
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setDimensions( const IndexType rows,
                                                                                                      const IndexType columns )
{
   TNL_ASSERT( rows >= 0 && columns >= 0, std::cerr << " rows = " << rows << " columns = " << columns );
   this->rows = rows;
   this->columns = columns;
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Matrix_ >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setLike( const Matrix_& matrix )
{
   setDimensions( matrix.getRows(), matrix.getColumns() );
   // SANDBOX_TODO: Replace the following line with assignment of metadata required by your format.
   //               Do not assign matrix elements here.
   this->rowPointers = matrix.rowPointers;
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename RowsCapacitiesVector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setRowCapacities(
   const RowsCapacitiesVector& rowsCapacities )
{
   TNL_ASSERT_EQ(
      rowsCapacities.getSize(), this->getRows(), "Number of matrix rows does not fit with rowCapacities vector size." );
   using RowsCapacitiesVectorDevice = typename RowsCapacitiesVector::DeviceType;

   // SANDBOX_TODO: Replace the following lines with the setup of your sparse matrix format based on
   //               `rowsCapacities`. This container has the same number of elements as is the number of
   //               rows of this matrix. Each element says how many nonzero elements the user needs to have
   //               in each row. This number can be increased if the sparse matrix format uses padding zeros.
   this->rowPointers.setSize( this->getRows() + 1 );
   if( std::is_same< DeviceType, RowsCapacitiesVectorDevice >::value ) {
      // GOTCHA: when this->getRows() == 0, getView returns a full view with size == 1
      if( this->getRows() > 0 ) {
         auto view = this->rowPointers.getView( 0, this->getRows() );
         view = rowsCapacities;
      }
   }
   else {
      RowsCapacitiesType thisRowsCapacities;
      thisRowsCapacities = rowsCapacities;
      if( this->getRows() > 0 ) {
         auto view = this->rowPointers.getView( 0, this->getRows() );
         view = thisRowsCapacities;
      }
   }
   this->rowPointers.setElement( this->getRows(), 0 );
   Algorithms::inplaceExclusiveScan( this->rowPointers );
   // this->rowPointers.template scan< Algorithms::ScanType::Exclusive >();
   //  End of sparse matrix format initiation.

   // SANDBOX_TODO: Compute number of all elements that need to be allocated by your format.
   const auto storageSize = rowPointers.getElement( this->getRows() );

   // The rest of this methods needs no changes.
   if( ! isBinary() ) {
      this->values.setSize( storageSize );
      this->values = (RealType) 0;
   }
   this->columnIndexes.setSize( storageSize );
   this->columnIndexes = paddingIndex< Index >;
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Vector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRowCapacities(
   Vector& rowCapacities ) const
{
   this->view.getRowCapacities( rowCapacities );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setElements(
   const std::initializer_list< std::tuple< IndexType, IndexType, RealType > >& data )
{
   const auto& rows = this->getRows();
   const auto& columns = this->getColumns();
   Containers::Vector< IndexType, Devices::Host, IndexType > rowCapacities( rows, 0 );
   for( const auto& i : data ) {
      if( std::get< 0 >( i ) >= rows ) {
         std::stringstream s;
         s << "Wrong row index " << std::get< 0 >( i ) << " in an initializer list";
         throw std::logic_error( s.str() );
      }
      rowCapacities[ std::get< 0 >( i ) ]++;
   }
   SparseSandboxMatrix< Real, Devices::Host, Index, MatrixType > hostMatrix( rows, columns );
   hostMatrix.setRowCapacities( rowCapacities );
   for( const auto& i : data ) {
      if( std::get< 1 >( i ) >= columns ) {
         std::stringstream s;
         s << "Wrong column index " << std::get< 1 >( i ) << " in an initializer list";
         throw std::logic_error( s.str() );
      }
      hostMatrix.setElement( std::get< 0 >( i ), std::get< 1 >( i ), std::get< 2 >( i ) );
   }
   ( *this ) = hostMatrix;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename MapIndex, typename MapValue >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setElements(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   Containers::Vector< IndexType, Devices::Host, IndexType > rowsCapacities( this->getRows(), 0 );
   for( auto element : map )
      rowsCapacities[ element.first.first ]++;
   if( ! std::is_same< DeviceType, Devices::Host >::value ) {
      SparseSandboxMatrix< Real, Devices::Host, Index, MatrixType > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setRowCapacities( rowsCapacities );
      for( auto element : map )
         hostMatrix.setElement( element.first.first, element.first.second, element.second );
      *this = hostMatrix;
   }
   else {
      this->setRowCapacities( rowsCapacities );
      for( auto element : map )
         this->setElement( element.first.first, element.first.second, element.second );
   }
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Vector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getCompressedRowLengths(
   Vector& rowLengths ) const
{
   this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
Index
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRowCapacity(
   const IndexType row ) const
{
   return this->view.getRowCapacity( row );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
Index
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getNonzeroElementsCount() const
{
   return this->view.getNonzeroElementsCount();
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reset()
{
   this->rows = 0;
   this->columns = 0;
   this->values.reset();
   this->columnIndexes.reset();
   // SANDBOX_TODO: Reset the metadata required by your format here.
   this->rowPointers.reset();
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRow( const IndexType& rowIdx ) const
   -> ConstRowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getRow( const IndexType& rowIdx )
   -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::setElement( const IndexType row,
                                                                                                   const IndexType column,
                                                                                                   const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::addElement(
   const IndexType row,
   const IndexType column,
   const RealType& value,
   const RealType& thisElementMultiplicator )
{
   this->view.addElement( row, column, value, thisElementMultiplicator );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
__cuda_callable__
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getElement(
   const IndexType row,
   const IndexType column ) const -> RealType
{
   return this->view.getElement( row, column );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename InVector, typename OutVector >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::vectorProduct(
   const InVector& inVector,
   OutVector& outVector,
   RealType matrixMultiplicator,
   RealType outVectorMultiplicator,
   IndexType firstRow,
   IndexType lastRow ) const
{
   this->view.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, firstRow, lastRow );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceRows( IndexType begin,
                                                                                                   IndexType end,
                                                                                                   Fetch& fetch,
                                                                                                   const Reduce& reduce,
                                                                                                   Keep& keep,
                                                                                                   const FetchValue& zero )
{
   this->view.reduceRows( begin, end, fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceRows(
   IndexType begin,
   IndexType end,
   Fetch& fetch,
   const Reduce& reduce,
   Keep& keep,
   const FetchValue& zero ) const
{
   this->view.reduceRows( begin, end, fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceAllRows( Fetch&& fetch,
                                                                                                      const Reduce&& reduce,
                                                                                                      Keep&& keep,
                                                                                                      const FetchReal& zero )
{
   this->reduceRows( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::reduceAllRows(
   Fetch& fetch,
   const Reduce& reduce,
   Keep& keep,
   const FetchReal& zero ) const
{
   this->reduceRows( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forElements( IndexType begin,
                                                                                                    IndexType end,
                                                                                                    Function&& function ) const
{
   this->view.forElements( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forElements( IndexType begin,
                                                                                                    IndexType end,
                                                                                                    Function&& function )
{
   this->view.forElements( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllElements(
   Function&& function ) const
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllElements( Function&& function )
{
   this->forElements( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forRows( IndexType begin,
                                                                                                IndexType end,
                                                                                                Function&& function )
{
   this->getView().forRows( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forRows( IndexType begin,
                                                                                                IndexType end,
                                                                                                Function&& function ) const
{
   this->getConstView().forRows( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllRows( Function&& function )
{
   this->getView().forAllRows( function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::forAllRows( Function&& function ) const
{
   this->getConstView().forAllRows( function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForRows(
   IndexType begin,
   IndexType end,
   Function& function ) const
{
   this->view.sequentialForRows( begin, end, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForRows( IndexType first,
                                                                                                          IndexType last,
                                                                                                          Function& function )
{
   this->view.sequentialForRows( first, last, function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForAllRows(
   Function& function ) const
{
   this->sequentialForRows( 0, this->getRows(), function );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Function >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::sequentialForAllRows(
   Function& function )
{
   this->sequentialForRows( 0, this->getRows(), function );
}

/*
template< typename Real,
          template< typename, typename, typename > class Segments,
          typename Device,
          typename Index,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, template< typename, typename > class Segments2, typename Index2, typename RealAllocator2, typename
IndexAllocator2 > void SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >:: addMatrix( const
SparseSandboxMatrix< Real2, Segments2, Device, Index2, RealAllocator2, IndexAllocator2 >& matrix, const RealType&
matrixMultiplicator, const RealType& thisMatrixMultiplicator )
{
}
*/

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Real2, typename Index2 >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getTransposition(
   const SparseSandboxMatrix< Real2, Device, Index2, MatrixType >& matrix,
   const RealType& matrixMultiplicator )
{
   // set transposed dimensions
   setDimensions( matrix.getColumns(), matrix.getRows() );

   // stage 1: compute row capacities for the transposition
   RowsCapacitiesType capacities;
   capacities.resize( this->getRows(), 0 );
   auto capacities_view = capacities.getView();
   using MatrixRowView = typename SparseSandboxMatrix< Real2, Device, Index2, MatrixType >::ConstRowView;
   matrix.forAllRows(
      [ = ] __cuda_callable__( const MatrixRowView& row ) mutable
      {
         for( IndexType c = 0; c < row.getSize(); c++ ) {
            // row index of the transpose = column index of the input
            const IndexType& transRowIdx = row.getColumnIndex( c );
            if( transRowIdx < 0 )
               continue;
            // increment the capacity for the row in the transpose
            Algorithms::AtomicOperations< DeviceType >::add( capacities_view[ row.getColumnIndex( c ) ], IndexType( 1 ) );
         }
      } );

   // set the row capacities
   setRowCapacities( capacities );
   capacities.reset();

   // index of the first unwritten element per row
   RowsCapacitiesType offsets;
   offsets.resize( this->getRows(), 0 );
   auto offsets_view = offsets.getView();

   // stage 2: copy and transpose the data
   auto trans_view = getView();
   matrix.forAllRows(
      [ = ] __cuda_callable__( const MatrixRowView& row ) mutable
      {
         // row index of the input = column index of the transpose
         const IndexType& rowIdx = row.getRowIndex();
         for( IndexType c = 0; c < row.getSize(); c++ ) {
            // row index of the transpose = column index of the input
            const IndexType& transRowIdx = row.getColumnIndex( c );
            if( transRowIdx < 0 )
               continue;
            // local index in the row of the transpose
            const IndexType transLocalIdx =
               Algorithms::AtomicOperations< DeviceType >::add( offsets_view[ transRowIdx ], IndexType( 1 ) );
            // get the row in the transposed matrix and set the value
            auto transRow = trans_view.getRow( transRowIdx );
            transRow.setElement( transLocalIdx, rowIdx, row.getValue( c ) * matrixMultiplicator );
         }
      } );
}

// copy assignment
template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=(
   const SparseSandboxMatrix& matrix )
{
   this->rows = matrix.rows;
   this->columns = matrix.columns;
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   // SANDBOX_TODO: Replace the following line with an assignment of metadata required by you sparse matrix format.
   this->rowPointers = matrix.rowPointers;
   this->view.bind( this->getView() );
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Device_ >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=(
   const SparseSandboxMatrix< RealType, Device_, IndexType, MatrixType, RealAllocator, IndexAllocator >& matrix )
{
   this->rows = matrix.rows;
   this->columns = matrix.columns;
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   // SANDBOX_TODO: Replace the following line with an assignment of metadata required by you sparse matrix format.
   this->rowPointers = matrix.rowPointers;
   this->view.bind( this->getView() );
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization, typename RealAllocator_ >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=(
   const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix )
{
   copyDenseToSparseMatrix( *this, matrix );
   this->view.bind( this->getView() );
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename RHSMatrix >
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >&
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator=( const RHSMatrix& matrix )
{
   copySparseToSparseMatrix( *this, matrix );
   this->view.bind( this->getView() );
   return *this;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Matrix >
bool
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator==( const Matrix& m ) const
{
   return view == m;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
template< typename Matrix >
bool
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::operator!=( const Matrix& m ) const
{
   return view != m;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::save( File& file ) const
{
   Object::save( file );
   file.save( &this->rows );
   file.save( &this->columns );
   file << this->values;
   file << this->columnIndexes;
   // SANDBOX_TODO: Replace this with medata required by your format
   file << this->rowPointers;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::load( File& file )
{
   Object::load( file );
   file.load( &this->rows );
   file.load( &this->columns );
   file >> this->values;
   file >> this->columnIndexes;
   // SANDBOX_TODO: Replace the following line with loading of metadata required by your sparse matrix format.
   file >> rowPointers;
   this->view.bind( this->getView() );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
void
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getColumnIndexes() const
   -> const ColumnsIndexesVectorType&
{
   return this->columnIndexes;
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename RealAllocator, typename IndexAllocator >
auto
SparseSandboxMatrix< Real, Device, Index, MatrixType, RealAllocator, IndexAllocator >::getColumnIndexes()
   -> ColumnsIndexesVectorType&
{
   return this->columnIndexes;
}

}  // namespace TNL::Matrices::Sandbox
