// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
// SPDX-License-Identifier: MIT

#pragma once

#include "SparseMatrix.h"
#include "SparseOperations.h"

namespace TNL::Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ), columnIndexes( indexAllocator )
{}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const SparseMatrix& matrix )
: values( matrix.values ), columnIndexes( matrix.columnIndexes ), segments( matrix.segments )
{
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Index_t, std::enable_if_t< std::is_integral< Index_t >::value, int > >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   Index_t rows,
   Index_t columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ), columnIndexes( indexAllocator ), segments( Containers::Vector< Index, Device, Index >( rows, 0 ) )
{
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename ListIndex >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const std::initializer_list< ListIndex >& rowCapacities,
   Index columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ), columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rowCapacities.size(), columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setRowCapacities( RowCapacitiesVectorType( rowCapacities ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename RowCapacitiesVector, std::enable_if_t< TNL::IsArrayType< RowCapacitiesVector >::value, int > >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const RowCapacitiesVector& rowCapacities,
   Index columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ), columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rowCapacities.getSize(), columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setRowCapacities( rowCapacities );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   Index rows,
   Index columns,
   const std::initializer_list< std::tuple< Index, Index, Real > >& data,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ), columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setElements( data );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename MapIndex, typename MapValue >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   Index rows,
   Index columns,
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ), columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setElements( map );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::getView() -> ViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getView(),
            this->getColumnIndexes().getView(),
            this->getSegments().getView() };
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
auto
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::getConstView() const
   -> ConstViewType
{
   return { this->getRows(),
            this->getColumns(),
            this->getValues().getConstView(),
            this->getColumnIndexes().getConstView(),
            this->getSegments().getConstView() };
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setDimensions(
   Index rows,
   Index columns )
{
   this->values.reset();
   this->columnIndexes.reset();
   this->segments.setSegmentsSizes( Containers::Vector< Index, Device, Index >( rows, 0 ) );
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setColumnsWithoutReset(
   Index columns )
{
   // update the base
   Base::bind( this->getRows(), columns, values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Matrix_ >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setLike(
   const Matrix_& matrix )
{
   this->segments.setSegmentsSizes( Containers::Vector< Index, Device, Index >( matrix.getRows(), 0 ) );
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), columnIndexes.getView(), segments.getView() );
   TNL_ASSERT_EQ( this->getRows(), segments.getSegmentsCount(), "mismatched segments count" );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename RowsCapacitiesVector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setRowCapacities(
   const RowsCapacitiesVector& rowCapacities )
{
   TNL_ASSERT_EQ(
      rowCapacities.getSize(), this->getRows(), "Number of matrix rows does not fit with rowCapacities vector size." );
   using RowsCapacitiesVectorDevice = typename RowsCapacitiesVector::DeviceType;
   if constexpr( std::is_same_v< Device, RowsCapacitiesVectorDevice > )
      this->segments.setSegmentsSizes( rowCapacities );
   else {
      RowCapacitiesVectorType thisRowCapacities;
      thisRowCapacities = rowCapacities;
      this->segments.setSegmentsSizes( thisRowCapacities );
   }
   if constexpr( ! Base::isBinary() ) {
      this->values.setSize( this->segments.getStorageSize() );
      this->values = 0;
   }
   this->values.setSize( this->segments.getStorageSize() );
   this->columnIndexes.setSize( this->segments.getStorageSize() );
   this->columnIndexes = paddingIndex< Index >;

   // update the base
   Base::bind( this->getRows(), this->getColumns(), values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setElements(
   const std::initializer_list< std::tuple< Index, Index, Real > >& data )
{
   const auto& rows = this->getRows();
   const auto& columns = this->getColumns();
   Containers::Vector< Index, Devices::Host, Index > rowCapacities( rows, 0 );
   for( const auto& i : data ) {
      if( std::get< 0 >( i ) >= rows ) {
         throw std::logic_error( "Wrong row index " + std::to_string( std::get< 0 >( i ) ) + " in an initializer list" );
      }
      rowCapacities[ std::get< 0 >( i ) ]++;
   }
   SparseMatrix< Real, Devices::Host, Index, MatrixType, Segments > hostMatrix( rows, columns );
   hostMatrix.setRowCapacities( rowCapacities );
   for( const auto& i : data ) {
      if( std::get< 1 >( i ) >= columns ) {
         throw std::logic_error( "Wrong column index " + std::to_string( std::get< 1 >( i ) ) + " in an initializer list" );
      }
      hostMatrix.setElement( std::get< 0 >( i ), std::get< 1 >( i ), std::get< 2 >( i ) );
   }
   if constexpr( std::is_same_v< Device, Devices::Host > )
      ( *this ) = std::move( hostMatrix );
   else
      ( *this ) = hostMatrix;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename MapIndex, typename MapValue >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setElements(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   Containers::Vector< Index, Devices::Host, Index > rowsCapacities( this->getRows(), 0 );
   for( auto element : map )
      rowsCapacities[ element.first.first ]++;
   if constexpr( ! std::is_same_v< Device, Devices::Host > ) {
      SparseMatrix< Real, Devices::Host, Index, MatrixType, Segments > hostMatrix( this->getRows(), this->getColumns() );
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

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::reset()
{
   this->values.reset();
   this->columnIndexes.reset();
   this->segments.reset();
   // update the base
   Base::bind( 0, 0, values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real2, typename Index2, template< typename, typename, typename > class Segments2 >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::getTransposition(
   const SparseMatrix< Real2, Device, Index2, MatrixType, Segments2 >& matrix,
   const ComputeReal& matrixMultiplicator )
{
   // set transposed dimensions
   setDimensions( matrix.getColumns(), matrix.getRows() );

   // stage 1: compute row capacities for the transposition
   RowCapacitiesVectorType capacities;
   capacities.resize( this->getRows(), 0 );
   auto capacities_view = capacities.getView();
   using MatrixRowView = typename SparseMatrix< Real2, Device, Index2, MatrixType, Segments2 >::ConstRowView;
   matrix.forAllRows(
      [ = ] __cuda_callable__( const MatrixRowView& row ) mutable
      {
         for( Index c = 0; c < row.getSize(); c++ ) {
            // row index of the transpose = column index of the input
            const Index& transRowIdx = row.getColumnIndex( c );
            if( transRowIdx == paddingIndex< Index > )
               continue;
            // increment the capacity for the row in the transpose
            Algorithms::AtomicOperations< Device >::add( capacities_view[ transRowIdx ], Index( 1 ) );
         }
      } );

   // set the row capacities
   setRowCapacities( capacities );
   capacities.reset();

   // index of the first unwritten element per row
   RowCapacitiesVectorType offsets;
   offsets.resize( this->getRows(), 0 );
   auto offsets_view = offsets.getView();

   // stage 2: copy and transpose the data
   auto trans_view = getView();
   matrix.forAllRows(
      [ = ] __cuda_callable__( const MatrixRowView& row ) mutable
      {
         // row index of the input = column index of the transpose
         const Index& rowIdx = row.getRowIndex();
         for( Index c = 0; c < row.getSize(); c++ ) {
            // row index of the transpose = column index of the input
            const Index& transRowIdx = row.getColumnIndex( c );
            if( transRowIdx == paddingIndex< Index > )
               continue;
            // local index in the row of the transpose
            const Index transLocalIdx = Algorithms::AtomicOperations< Device >::add( offsets_view[ transRowIdx ], Index( 1 ) );
            // get the row in the transposed matrix and set the value
            auto transRow = trans_view.getRow( transRowIdx );
            transRow.setElement( transLocalIdx, rowIdx, row.getValue( c ) * matrixMultiplicator );
         }
      } );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::operator=(
   const SparseMatrix& matrix )
{
   this->values = matrix.values;
   this->columnIndexes = matrix.columnIndexes;
   this->segments = matrix.segments;
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), columnIndexes.getView(), segments.getView() );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::operator=(
   SparseMatrix&& matrix ) noexcept( false )
{
   this->values = std::move( matrix.values );
   this->columnIndexes = std::move( matrix.columnIndexes );
   this->segments = std::move( matrix.segments );
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), columnIndexes.getView(), segments.getView() );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Real_, typename Device_, typename Index_, ElementsOrganization Organization, typename RealAllocator_ >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::operator=(
   const DenseMatrix< Real_, Device_, Index_, Organization, RealAllocator_ >& matrix )
{
   copyDenseToSparseMatrix( *this, matrix );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename RHSMatrix >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >&
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::operator=(
   const RHSMatrix& matrix )
{
   copySparseToSparseMatrix( *this, matrix );
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::save( File& file ) const
{
   file.save( &this->rows );
   file.save( &this->columns );
   file << values << columnIndexes;
   segments.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::load( File& file )
{
   Index rows = 0;
   Index columns = 0;
   file.load( &rows );
   file.load( &columns );
   file >> values >> columnIndexes;
   segments.load( file );
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::save(
   const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename >
          class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::load(
   const String& fileName )
{
   Object::load( fileName );
}

}  // namespace TNL::Matrices
