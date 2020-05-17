/***************************************************************************
                          Dense.hpp  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <TNL/Assert.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
DenseMatrix( const IndexType rows, const IndexType columns )
{
   this->setDimensions( rows, columns );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Value >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
DenseMatrix( std::initializer_list< std::initializer_list< Value > > data )
{
   this->setElements( data );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Value >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
setElements( std::initializer_list< std::initializer_list< Value > > data )
{
   IndexType rows = data.size();
   IndexType columns = 0;
   for( auto row : data )
      columns = max( columns, row.size() );
   this->setDimensions( rows, columns );
   if( ! std::is_same< DeviceType, Devices::Host >::value )
   {
      DenseMatrix< RealType, Devices::Host, IndexType > hostDense( rows, columns );
      IndexType rowIdx( 0 );
      for( auto row : data )
      {
         IndexType columnIdx( 0 );
         for( auto element : row )
            hostDense.setElement( rowIdx, columnIdx++, element );
         rowIdx++;
      }
      *this = hostDense;
   }
   else
   {
      IndexType rowIdx( 0 );
      for( auto row : data )
      {
         IndexType columnIdx( 0 );
         for( auto element : row )
            this->setElement( rowIdx, columnIdx++, element );
         rowIdx++;
      }
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getView() -> ViewType
{
   return ViewType( this->getRows(),
                    this->getColumns(),
                    this->getValues().getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->getRows(),
                         this->getColumns(),
                         this->getValues().getConstView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
String
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getSerializationType()
{
   return ViewType::getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
String
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
setDimensions( const IndexType rows,
               const IndexType columns )
{
   Matrix< Real, Device, Index >::setDimensions( rows, columns );
   this->segments.setSegmentsSizes( rows, columns );
   this->values.setSize( rows * columns );
   this->values = 0.0;
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Matrix_ >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
setLike( const Matrix_& matrix )
{
   this->setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename RowCapacitiesVector >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
setRowCapacities( const RowCapacitiesVector& rowCapacities )
{
   TNL_ASSERT_EQ( rowCapacities.getSize(), this->getRows(), "" );
   TNL_ASSERT_LE( max( rowCapacities ), this->getColumns(), "" );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename RowLengthsVector >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getCompressedRowLengths( RowLengthsVector& rowLengths ) const
{
   this->view.getCompressedRowLengths( rowLengths );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
Index
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getElementsCount() const
{
   return this->getRows() * this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
Index
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getNonzeroElementsCount() const
{
   return this->view.getNonzeroElementsCount();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
reset()
{
   Matrix< Real, Device, Index >::reset();
   this->segments.reset();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
setValue( const Real& value )
{
   this->view.setValue( value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__ auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__ auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getRow( const IndexType& rowIdx ) -> RowView
{
   return this->view.getRow( rowIdx );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
Real& DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator()( const IndexType row,
                                                const IndexType column )
{
   return this->view.operator()( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
const Real& DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator()( const IndexType row,
                                                      const IndexType column ) const
{
   return this->view.operator()( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__ void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
setElement( const IndexType row,
            const IndexType column,
            const RealType& value )
{
   this->view.setElement( row, column, value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__ void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   this->view.addElement( row, column, value, thisElementMultiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__ Real
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getElement( const IndexType row,
            const IndexType column ) const
{
   return this->view.getElement( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& zero ) const
{
   this->view.rowsReduction( first, last, fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   this->view.forRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
forRows( IndexType first, IndexType last, Function& function )
{
   this->view.forRows( first, last, function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Function >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
forAllRows( Function& function )
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename InVector,
             typename OutVector >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType& matrixMultiplicator,
               const RealType& outVectorMultiplicator,
               const IndexType begin,
               const IndexType end ) const
{
   this->view.vectorProduct( inVector, outVector, matrixMultiplicator, outVectorMultiplicator, begin, end );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Matrix >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
addMatrix( const Matrix& matrix,
           const RealType& matrixMultiplicator,
           const RealType& thisMatrixMultiplicator )
{
   TNL_ASSERT( this->getColumns() == matrix.getColumns() &&
              this->getRows() == matrix.getRows(),
            std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                 << "This matrix rows: " << this->getRows() << std::endl
                 << "That matrix columns: " << matrix.getColumns() << std::endl
                 << "That matrix rows: " << matrix.getRows() << std::endl );

   if( thisMatrixMultiplicator == 1.0 )
      this->values += matrixMultiplicator * matrix.values;
   else
      this->values = thisMatrixMultiplicator * this->values + matrixMultiplicator * matrix.values;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Matrix1, typename Matrix2, int tileDim >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getMatrixProduct( const Matrix1& matrix1,
                                                              const Matrix2& matrix2,
                                                              const RealType& matrix1Multiplicator,
                                                              const RealType& matrix2Multiplicator )
{
   TNL_ASSERT_EQ( matrix1.getColumns(), matrix2.getRows(), "" );
   this->setDimensions( matrix1.getRows(), matrix2.getColumns() );

   this->getView().getMatrixProduct( matrix1.getConstView(),
                                     matrix2.getConstView(),
                                     matrix1Multiplicator,
                                     matrix2Multiplicator );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Matrix, int tileDim >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getTransposition( const Matrix& matrix,
                                                              const RealType& matrixMultiplicator )
{
   TNL_ASSERT( this->getColumns() == matrix.getRows() &&
              this->getRows() == matrix.getColumns(),
               std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                    << "This matrix rows: " << this->getRows() << std::endl
                    << "That matrix columns: " << matrix.getColumns() << std::endl
                    << "That matrix rows: " << matrix.getRows() << std::endl );

   this->getView().getTransposition( matrix.getConstView(), matrixMultiplicator );

}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Vector1, typename Vector2 >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::performSORIteration( const Vector1& b,
                                                        const IndexType row,
                                                        Vector2& x,
                                                        const RealType& omega ) const
{
   RealType sum( 0.0 ), diagonalValue;
   for( IndexType i = 0; i < this->getColumns(); i++ )
   {
      if( i == row )
         diagonalValue = this->getElement( row, row );
      else
         sum += this->getElement( row, i ) * x[ i ];
   }
   x[ row ] = ( 1.0 - omega ) * x[ row ] + omega / diagonalValue * ( b[ row ] - sum );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
operator=( const DenseMatrix< Real, Device, Index, Organization, RealAllocator >& matrix )
{
   setLike( matrix );
   this->values = matrix.values;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename RHSReal, typename RHSDevice, typename RHSIndex,
             ElementsOrganization RHSOrganization, typename RHSRealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
operator=( const DenseMatrix< RHSReal, RHSDevice, RHSIndex, RHSOrganization, RHSRealAllocator >& matrix )
{
   using RHSMatrix = DenseMatrix< RHSReal, RHSDevice, RHSIndex, RHSOrganization, RHSRealAllocator >;
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;

   this->setLike( matrix );
   if( Organization == RHSOrganization )
   {
      this->values = matrix.getValues();
      return *this;
   }

   auto this_view = this->view;
   if( std::is_same< DeviceType, RHSDeviceType >::value )
   {
      auto f = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIdx, const RHSRealType& value, bool& compute ) mutable {
         this_view( rowIdx, columnIdx ) = value;
      };
      matrix.forAllRows( f );
   }
   else
   {
      const IndexType maxRowLength = matrix.getColumns();
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType > thisValuesBuffer( bufferSize );
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();

      IndexType baseRow( 0 );
      const IndexType rowsCount = this->getRows();
      while( baseRow < rowsCount )
      {
         const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );

         ////
         // Copy matrix elements into buffer
         auto f1 = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIdx, const RHSRealType& value, bool& compute ) mutable {
            const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + columnIdx;
            matrixValuesBuffer_view[ bufferIdx ] = value;
         };
         matrix.forRows( baseRow, lastRow, f1 );

         ////
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;

         ////
         // Copy matrix elements from the buffer to the matrix.
         auto this_view = this->view;
         auto f2 = [=] __cuda_callable__ ( IndexType columnIdx, IndexType bufferRowIdx ) mutable {
            IndexType bufferIdx = bufferRowIdx * maxRowLength + columnIdx;
            this_view( baseRow + bufferRowIdx, columnIdx ) = thisValuesBuffer_view[ bufferIdx ];
         };
         Algorithms::ParallelFor2D< DeviceType >::exec( ( IndexType ) 0, ( IndexType ) 0, ( IndexType ) maxRowLength, ( IndexType ) min( bufferRowsCount, this->getRows() - baseRow ), f2 );
         baseRow += bufferRowsCount;
      }
   }
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename RHSMatrix >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
operator=( const RHSMatrix& matrix )
{
   using RHSIndexType = typename RHSMatrix::IndexType;
   using RHSRealType = typename RHSMatrix::RealType;
   using RHSDeviceType = typename RHSMatrix::DeviceType;
   using RHSRealAllocatorType = typename RHSMatrix::RealAllocatorType;

   Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > rowLengths;
   matrix.getCompressedRowLengths( rowLengths );
   this->setDimensions( matrix.getRows(), matrix.getColumns() );

   // TODO: use getConstView when it works
   const auto matrixView = const_cast< RHSMatrix& >( matrix ).getView();
   auto values_view = this->values.getView();
   RHSIndexType padding_index = matrix.getPaddingIndex();
   this->values = 0.0;

   if( std::is_same< DeviceType, RHSDeviceType >::value )
   {
      const auto segments_view = this->segments.getView();
      auto f = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx_, RHSIndexType columnIdx, const RHSRealType& value, bool& compute ) mutable {
         if( value != 0.0 && columnIdx != padding_index )
            values_view[ segments_view.getGlobalIndex( rowIdx, columnIdx ) ] = value;
      };
      matrix.forAllRows( f );
   }
   else
   {
      const IndexType maxRowLength = max( rowLengths );
      const IndexType bufferRowsCount( 128 );
      const size_t bufferSize = bufferRowsCount * maxRowLength;
      Containers::Vector< RHSRealType, RHSDeviceType, RHSIndexType, RHSRealAllocatorType > matrixValuesBuffer( bufferSize );
      Containers::Vector< RHSIndexType, RHSDeviceType, RHSIndexType > matrixColumnsBuffer( bufferSize );
      Containers::Vector< RealType, DeviceType, IndexType, RealAllocatorType > thisValuesBuffer( bufferSize );
      Containers::Vector< IndexType, DeviceType, IndexType > thisColumnsBuffer( bufferSize );
      auto matrixValuesBuffer_view = matrixValuesBuffer.getView();
      auto matrixColumnsBuffer_view = matrixColumnsBuffer.getView();
      auto thisValuesBuffer_view = thisValuesBuffer.getView();
      auto thisColumnsBuffer_view = thisColumnsBuffer.getView();

      IndexType baseRow( 0 );
      const IndexType rowsCount = this->getRows();
      while( baseRow < rowsCount )
      {
         const IndexType lastRow = min( baseRow + bufferRowsCount, rowsCount );
         thisColumnsBuffer = padding_index;
         matrixColumnsBuffer_view = padding_index;

         ////
         // Copy matrix elements into buffer
         auto f1 = [=] __cuda_callable__ ( RHSIndexType rowIdx, RHSIndexType localIdx, RHSIndexType columnIndex, const RHSRealType& value, bool& compute ) mutable {
            if( columnIndex != padding_index )
            {
               const IndexType bufferIdx = ( rowIdx - baseRow ) * maxRowLength + localIdx;
               matrixColumnsBuffer_view[ bufferIdx ] = columnIndex;
               matrixValuesBuffer_view[ bufferIdx ] = value;
            }
         };
         matrix.forRows( baseRow, lastRow, f1 );

         ////
         // Copy the source matrix buffer to this matrix buffer
         thisValuesBuffer_view = matrixValuesBuffer_view;
         thisColumnsBuffer_view = matrixColumnsBuffer_view;

         ////
         // Copy matrix elements from the buffer to the matrix
         auto this_view = this->view;
         auto f2 = [=] __cuda_callable__ ( IndexType bufferColumnIdx, IndexType bufferRowIdx ) mutable {
            IndexType bufferIdx = bufferRowIdx * maxRowLength + bufferColumnIdx;
            IndexType columnIdx = thisColumnsBuffer_view[ bufferIdx ];
            if( columnIdx != padding_index )
               this_view( baseRow + bufferRowIdx, columnIdx ) = thisValuesBuffer_view[ bufferIdx ];
         };
         Algorithms::ParallelFor2D< DeviceType >::exec( ( IndexType ) 0, ( IndexType ) 0, ( IndexType ) maxRowLength, ( IndexType ) min( bufferRowsCount, this->getRows() - baseRow ), f2 );
         baseRow += bufferRowsCount;
      }
   }
   this->view = this->getView();
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
bool
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
operator==( const DenseMatrix< Real_, Device_, Index_, Organization >& matrix ) const
{
   return( this->getRows() == matrix.getRows() &&
           this->getColumns() == matrix.getColumns() &&
           this->getValues() == matrix.getValues() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
   template< typename Real_, typename Device_, typename Index_, typename RealAllocator_ >
bool
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
operator!=( const DenseMatrix< Real_, Device_, Index_, Organization >& matrix ) const
{
   return ! ( *this == matrix );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::save( const String& fileName ) const
{
   this->view.save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::save( File& file ) const
{
   this->view.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::load( File& file )
{
   Matrix< Real, Device, Index >::load( file );
   this->segments.load( file );
   this->view = this->getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
void DenseMatrix< Real, Device, Index, Organization, RealAllocator >::print( std::ostream& str ) const
{
   this->view.print( str );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
__cuda_callable__
Index
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::
getElementIndex( const IndexType row, const IndexType column ) const
{
   return this->segments.getGlobalIndex( row, column );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization,
          typename RealAllocator >
std::ostream& operator<< ( std::ostream& str, const DenseMatrix< Real, Device, Index, Organization, RealAllocator >& matrix )
{
   matrix.print( str );
   return str;
}

} // namespace Matrices
} // namespace TNL
