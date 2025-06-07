// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "SparseMatrix.h"
#include "SparseOperations.h"

namespace TNL::Matrices {

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ),
  columnIndexes( indexAllocator )
{}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const SparseMatrix& matrix )
: values( matrix.values ),
  columnIndexes( matrix.columnIndexes ),
  segments( matrix.segments )
{
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename Index_t, std::enable_if_t< std::is_integral_v< Index_t >, int > >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   Index_t rows,
   Index_t columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ),
  columnIndexes( indexAllocator ),
  segments( Containers::Vector< Index, Device, Index >( rows, 0 ) )
{
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename ListIndex >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const std::initializer_list< ListIndex >& rowCapacities,
   Index columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ),
  columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rowCapacities.size(), columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setRowCapacities( RowCapacitiesVectorType( rowCapacities ) );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename RowCapacitiesVector, std::enable_if_t< TNL::IsArrayType< RowCapacitiesVector >::value, int > >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   const RowCapacitiesVector& rowCapacities,
   Index columns,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ),
  columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rowCapacities.getSize(), columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setRowCapacities( rowCapacities );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   Index rows,
   Index columns,
   const std::initializer_list< std::tuple< Index, Index, Real > >& data,
   SymmetricMatrixEncoding encoding,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ),
  columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setElements( data, encoding );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename MapIndex, typename MapValue >
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::SparseMatrix(
   Index rows,
   Index columns,
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   SymmetricMatrixEncoding encoding,
   const RealAllocatorType& realAllocator,
   const IndexAllocatorType& indexAllocator )
: values( realAllocator ),
  columnIndexes( indexAllocator )
{
   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
   this->setElements( map, encoding );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setDimensions(
   Index rows,
   Index columns,
   SegmentsType segments )
{
   if( segments.getSegmentsCount() != rows )
      throw std::invalid_argument( "the number of segments not match the number of matrix rows" );

   this->segments = segments;

   if constexpr( ! Base::isBinary() ) {
      this->values.setSize( segments.getStorageSize() );
      this->values = 0;
   }
   this->columnIndexes.setSize( segments.getStorageSize() );
   this->columnIndexes = paddingIndex< Index >;

   // update the base
   Base::bind( rows, columns, values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename RowCapacitiesVector >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setRowCapacities(
   const RowCapacitiesVector& rowCapacities )
{
   if( (Index) rowCapacities.getSize() != this->getRows() )
      throw std::invalid_argument( "setRowCapacities: size of the input vector does not match the number of matrix rows" );

   using RowCapacitiesVectorDevice = typename RowCapacitiesVector::DeviceType;
   if constexpr( std::is_same_v< Device, RowCapacitiesVectorDevice > )
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
   this->columnIndexes.setSize( this->segments.getStorageSize() );
   this->columnIndexes = paddingIndex< Index >;

   // update the base
   Base::bind( this->getRows(), this->getColumns(), values.getView(), columnIndexes.getView(), segments.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setElements(
   const std::initializer_list< std::tuple< Index, Index, Real > >& data,
   SymmetricMatrixEncoding encoding )
{
   std::map< std::pair< Index, Index >, Real > map;
   for( const auto& [ row, column, value ] : data )
      map[ { row, column } ] = value;
   this->setElements( map, encoding );
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
template< typename MapIndex, typename MapValue >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::setElements(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map,
   SymmetricMatrixEncoding encoding )
{
   if constexpr( ! std::is_same_v< Device, Devices::Host > && ! std::is_same_v< Device, Devices::Sequential > ) {
      SparseMatrix< Real, Devices::Host, Index, MatrixType, Segments > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setElements( map, encoding );
      *this = hostMatrix;
   }
   else {
      RowCapacitiesVectorType capacities( this->getRows(), 0 );
      for( const auto& [ coordinates, value ] : map ) {
         auto [ rowIdx, columnIdx ] = coordinates;
         if( Base::isSymmetric() ) {
            if( encoding == SymmetricMatrixEncoding::Complete ) {
               auto query = map.find( { columnIdx, rowIdx } );
               if( query == map.end() || query->second != value )
                  throw std::logic_error( "SparseMatrix is configured as symmetric, but the input data is not symmetric." );
               if( rowIdx < columnIdx )
                  continue;
            }
            if( encoding == SymmetricMatrixEncoding::LowerPart && rowIdx < columnIdx )
               throw std::logic_error( "Only lower part of the symmetric matrix is expected." );
            if( encoding == SymmetricMatrixEncoding::UpperPart ) {
               if( rowIdx > columnIdx )
                  throw std::logic_error( "Only upper part of the symmetric matrix is expected." );
               swap( rowIdx, columnIdx );
            }
            if( encoding == SymmetricMatrixEncoding::SparseMixed ) {
               if( rowIdx < columnIdx )
                  swap( rowIdx, columnIdx );
            }
         }
         if( rowIdx >= this->getRows() )
            throw std::logic_error( "Wrong row index " + std::to_string( rowIdx ) + " in the input data structure." );
         if( columnIdx >= this->getColumns() )
            throw std::logic_error( "Wrong column index " + std::to_string( columnIdx ) + " in the input data structure." );
         capacities[ rowIdx ]++;
      }
      this->setRowCapacities( capacities );

      if( ! Base::isSymmetric() || encoding == SymmetricMatrixEncoding::LowerPart ) {
         // The following algorithm is based on the fact that the input std::map
         // is sorted in a row-major order and that row capacities were already
         // set. It is much more efficient than calling setElement over and over,
         // since it avoids the sequential lookups of column indexes in each row.

         Index lastRowIdx = 0;
         Index localIdx = 0;
         for( const auto& [ coordinates, value ] : map ) {
            const auto& [ rowIdx, columnIdx ] = coordinates;
            if( Base::isSymmetric() && rowIdx < columnIdx )
               continue;
            auto row = this->getRow( rowIdx );
            if( rowIdx != lastRowIdx )
               localIdx = 0;
            row.setElement( localIdx++, columnIdx, value );
            lastRowIdx = rowIdx;
         }
      }
      else {  // symmetric matrix with other coding than lower part
         for( const auto& [ coordinates, value ] : map ) {
            auto [ rowIdx, columnIdx ] = coordinates;
            if( encoding == SymmetricMatrixEncoding::Complete && rowIdx < columnIdx )
               continue;
            if( ( encoding == SymmetricMatrixEncoding::UpperPart || encoding == SymmetricMatrixEncoding::SparseMixed )
                && rowIdx < columnIdx )
               swap( rowIdx, columnIdx );
            this->setElement( rowIdx, columnIdx, value );
         }
      }
   }
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
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
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::save(
   const String& fileName ) const
{
   File( fileName, std::ios_base::out ) << *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
void
SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >::load(
   const String& fileName )
{
   File( fileName, std::ios_base::in ) >> *this;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
File&
operator>>( File& file,
            SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >& matrix )
{
   const std::string type = getObjectType( file );
   if( type != matrix.getSerializationType() )
      throw Exceptions::FileDeserializationError( file.getFileName(),
                                                  "object type does not match (expected " + matrix.getSerializationType()
                                                     + ", found " + type + ")." );
   std::size_t rows = 0;
   std::size_t columns = 0;
   file.load( &rows );
   file.load( &columns );
   Segments< Device, Index, IndexAllocator > segments;
   segments.load( file );
   // setDimensions initializes the internal segments attribute
   matrix.setDimensions( rows, columns, segments );
   file >> matrix.getValues() >> matrix.getColumnIndexes();
   // update views in the base class
   matrix.setColumnsWithoutReset( matrix.getColumns() );
   return file;
}

template< typename Real,
          typename Device,
          typename Index,
          typename MatrixType,
          template< typename, typename, typename > class Segments,
          typename ComputeReal,
          typename RealAllocator,
          typename IndexAllocator >
File&
operator>>( File&& file,
            SparseMatrix< Real, Device, Index, MatrixType, Segments, ComputeReal, RealAllocator, IndexAllocator >& matrix )
{
   // named r-value is an l-value reference, so this is not recursion
   return file >> matrix;
}

}  // namespace TNL::Matrices
