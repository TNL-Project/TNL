/***************************************************************************
                          DenseMatrixView.hpp  -  description
                             -------------------
    begin                : Nov 29, 2013
    copyright            : (C) 2013 by Tomas Oberhuber
    email                : tomas.oberhuber@fjfi.cvut.cz
 ***************************************************************************/

/* See Copyright Notice in tnl/Copyright */

#pragma once

#include <iomanip>
#include <functional>
#include <TNL/Assert.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Exceptions/NotImplementedError.h>

namespace TNL {
namespace Matrices {

#ifdef HAVE_CUDA
template< int tileDim,
          int tileRowBlockSize,
          typename ResultMatrix,
          typename Matrix1,
          typename Matrix2,
          typename Real,
          typename Index >
__global__ void
DenseMatrixProductKernel( ResultMatrix resultMatrix,
                          const Matrix1 matrixA,
                          const Matrix2 matrixB,
                          const Real matrixAMultiplicator,
                          const Real matrixBMultiplicator,
                          const Index gridIdx_x,
                          const Index gridIdx_y );

template< typename Real,
          typename Index,
          typename Matrix,
          ElementsOrganization Organization,
          int tileDim,
          int tileRowBlockSize,
          typename Device >
__global__ void
DenseTranspositionAlignedKernel( DenseMatrixView< Real, Device, Index >* resultMatrix,
                                 const Matrix* inputMatrix,
                                 const Real matrixMultiplicator,
                                 const Index gridIdx_x,
                                 const Index gridIdx_y );

template< typename Real,
          typename Index,
          ElementsOrganization Organization,
          typename Matrix,
          int tileDim,
          int tileRowBlockSize,
          typename Device >
__global__ void
DenseTranspositionNonAlignedKernel( DenseMatrixView< Real, Device, Index >* resultMatrix,
                                    const Matrix* inputMatrix,
                                    const Real matrixMultiplicator,
                                    const Index gridIdx_x,
                                    const Index gridIdx_y );

#endif

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::
DenseMatrixView()
{
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
DenseMatrixView< Real, Device, Index, Organization >::
DenseMatrixView( const IndexType rows,
                 const IndexType columns,
                 const ValuesViewType& values )
 : MatrixView< Real, Device, Index >( rows, columns, values )
{
   SegmentsType a( rows, columns );
   segments = a.getView();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::
getView() -> ViewType
{
   return ViewType( this->getRows(),
                    this->getColumns(),
                    this->getValues().getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
auto
DenseMatrixView< Real, Device, Index, Organization >::
getConstView() const -> ConstViewType
{
   return ConstViewType( this->getRows(),
                         this->getColumns(),
                         this->getValues().getConstView() );

}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
String
DenseMatrixView< Real, Device, Index, Organization >::
getSerializationType()
{
   return String( "Matrices::DenseMatrix< " ) +
          TNL::getSerializationType< RealType >() + ", [any_device], " +
          TNL::getSerializationType< IndexType >() + ", " +
          ( Organization ? "true" : "false" ) + ", [any_allocator] >";
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
String
DenseMatrixView< Real, Device, Index, Organization >::
getSerializationTypeVirtual() const
{
   return this->getSerializationType();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Vector >
void
DenseMatrixView< Real, Device, Index, Organization >::
getCompressedRowLengths( Vector& rowLengths ) const
{
   rowLengths.setSize( this->getRows() );
   rowLengths = 0;
   auto rowLengths_view = rowLengths.getView();
   auto fetch = [] __cuda_callable__ ( IndexType row, IndexType column, const RealType& value ) -> IndexType {
      return ( value != 0.0 );
   };
   auto keep = [=] __cuda_callable__ ( const IndexType rowIdx, const IndexType value ) mutable {
      rowLengths_view[ rowIdx ] = value;
   };
   this->allRowsReduction( fetch, std::plus<>{}, keep, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::
getRowLength( const IndexType row ) const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::
getMaxRowLength() const
{
   return this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::
getElementsCount() const
{
   return this->getRows() * this->getColumns();
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
Index
DenseMatrixView< Real, Device, Index, Organization >::
getNonzeroElementsCount() const
{
   const auto values_view = this->values.getConstView();
   auto fetch = [=] __cuda_callable__ ( const IndexType i ) -> IndexType {
      return ( values_view[ i ] != 0.0 );
   };
   return Algorithms::Reduction< DeviceType >::reduce( this->values.getSize(), std::plus<>{}, fetch, 0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void
DenseMatrixView< Real, Device, Index, Organization >::
setValue( const Real& value )
{
   this->values = value;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto
DenseMatrixView< Real, Device, Index, Organization >::
getRow( const IndexType& rowIdx ) const -> const RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getConstView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ auto
DenseMatrixView< Real, Device, Index, Organization >::
getRow( const IndexType& rowIdx ) -> RowView
{
   TNL_ASSERT_LT( rowIdx, this->getRows(), "Row index is larger than number of matrix rows." );
   return RowView( this->segments.getSegmentView( rowIdx ), this->values.getView() );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
Real& DenseMatrixView< Real, Device, Index, Organization >::operator()( const IndexType row,
                                                const IndexType column )
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
const Real& DenseMatrixView< Real, Device, Index, Organization >::operator()( const IndexType row,
                                                      const IndexType column ) const
{
   TNL_ASSERT_GE( row, 0, "Row index must be non-negative." );
   TNL_ASSERT_LT( row, this->getRows(), "Row index is out of bounds." );
   TNL_ASSERT_GE( column, 0, "Column index must be non-negative." );
   TNL_ASSERT_LT( column, this->getColumns(), "Column index is out of bounds." );

   return this->values.operator[]( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ void
DenseMatrixView< Real, Device, Index, Organization >::
setElement( const IndexType row,
            const IndexType column,
            const RealType& value )
{
   this->values.setElement( this->getElementIndex( row, column ), value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ void
DenseMatrixView< Real, Device, Index, Organization >::
addElement( const IndexType row,
            const IndexType column,
            const RealType& value,
            const RealType& thisElementMultiplicator )
{
   const IndexType elementIndex = this->getElementIndex( row, column );
   if( thisElementMultiplicator == 1.0 )
      this->values.setElement( elementIndex,
                               this->values.getElement( elementIndex ) + value );
   else
      this->values.setElement( elementIndex,
                               thisElementMultiplicator * this->values.getElement( elementIndex ) + value );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__ Real
DenseMatrixView< Real, Device, Index, Organization >::
getElement( const IndexType row,
            const IndexType column ) const
{
   return this->values.getElement( this->getElementIndex( row, column ) );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchValue >
void
DenseMatrixView< Real, Device, Index, Organization >::
rowsReduction( IndexType first, IndexType last, Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchValue& zero ) const
{
   const auto values_view = this->values.getConstView();
   auto fetch_ = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable -> decltype( fetch( IndexType(), IndexType(), RealType() ) ) {
         return fetch( rowIdx, columnIdx, values_view[ globalIdx ] );
      return zero;
   };
   this->segments.segmentsReduction( first, last, fetch_, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Fetch, typename Reduce, typename Keep, typename FetchReal >
void
DenseMatrixView< Real, Device, Index, Organization >::
allRowsReduction( Fetch& fetch, const Reduce& reduce, Keep& keep, const FetchReal& zero ) const
{
   this->rowsReduction( 0, this->getRows(), fetch, reduce, keep, zero );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forRows( IndexType first, IndexType last, Function& function ) const
{
   const auto values_view = this->values.getConstView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable {
      function( rowIdx, columnIdx, columnIdx, values_view[ globalIdx ], compute );
   };
   this->segments.forSegments( first, last, f );

}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forRows( IndexType first, IndexType last, Function& function )
{
   auto values_view = this->values.getView();
   auto f = [=] __cuda_callable__ ( IndexType rowIdx, IndexType columnIdx, IndexType globalIdx, bool& compute ) mutable {
      function( rowIdx, columnIdx, globalIdx, values_view[ globalIdx ], compute );
   };
   this->segments.forSegments( first, last, f );

}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forAllRows( Function& function ) const
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Function >
void
DenseMatrixView< Real, Device, Index, Organization >::
forAllRows( Function& function )
{
   this->forRows( 0, this->getRows(), function );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename InVector,
             typename OutVector >
void
DenseMatrixView< Real, Device, Index, Organization >::
vectorProduct( const InVector& inVector,
               OutVector& outVector,
               const RealType& matrixMultiplicator,
               const RealType& outVectorMultiplicator,
               const IndexType begin,
               IndexType end ) const
{
   TNL_ASSERT_EQ( this->getColumns(), inVector.getSize(), "Matrix columns count differs with input vector size." );
   TNL_ASSERT_EQ( this->getRows(), outVector.getSize(), "Matrix rows count differs with output vector size." );

   const auto inVectorView = inVector.getConstView();
   auto outVectorView = outVector.getView();
   const auto valuesView = this->values.getConstView();
   if( end == 0 )
      end = this->getRows();
   auto fetch = [=] __cuda_callable__ ( IndexType row, IndexType column, IndexType offset, bool& compute ) -> RealType {
      return valuesView[ offset ] * inVectorView[ column ];
   };
   auto keeper = [=] __cuda_callable__ ( IndexType row, const RealType& value ) mutable {
      outVectorView[ row ] = matrixMultiplicator * value + outVectorMultiplicator * outVectorView[ row ];
   };
   this->segments.segmentsReduction( begin, end, fetch, std::plus<>{}, keeper, ( RealType ) 0.0 );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Matrix >
void
DenseMatrixView< Real, Device, Index, Organization >::
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
          ElementsOrganization Organization >
   template< typename MatrixView1, typename MatrixView2, int tileDim >
void DenseMatrixView< Real, Device, Index, Organization >::getMatrixProduct( const MatrixView1& matrix1,
                                                              const MatrixView2& matrix2,
                                                              const RealType& matrix1Multiplicator,
                                                              const RealType& matrix2Multiplicator )
{
   TNL_ASSERT_EQ( matrix1.getColumns(), matrix2.getRows(), "" );
   TNL_ASSERT_EQ( this->getRows(), matrix1.getRows(), "" );
   TNL_ASSERT_EQ( this->getColumns(), matrix2.getColumns(), "" );

   if( std::is_same< Device, Devices::Host >::value )
      for( IndexType i = 0; i < this->getRows(); i += tileDim )
         for( IndexType j = 0; j < this->getColumns(); j += tileDim )
         {
            const IndexType tileRows = min( tileDim, this->getRows() - i );
            const IndexType tileColumns = min( tileDim, this->getColumns() - j );
            for( IndexType i1 = i; i1 < i + tileRows; i1++ )
               for( IndexType j1 = j; j1 < j + tileColumns; j1++ )
                  ( *this )( i1, j1 ) = 0.0;

            for( IndexType k = 0; k < matrix1.getColumns(); k += tileDim )
            {
               const IndexType lastK = min( k + tileDim, matrix1.getColumns() );
               for( IndexType i1 = 0; i1 < tileRows; i1++ )
                  for( IndexType j1 = 0; j1 < tileColumns; j1++ )
                     for( IndexType k1 = k; k1 < lastK; k1++ )
                        ( *this )( i + i1, j + j1 ) +=
                            matrix1Multiplicator * matrix1( i + i1, k1 ) *
                            matrix2Multiplicator * matrix2( k1, j + j1 );
            }
         }
   if( std::is_same< Device, Devices::Cuda >::value )
   {
#ifdef HAVE_CUDA
      dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
      const IndexType matrixProductCudaBlockSize( 256 );
      const IndexType rowTiles = roundUpDivision( this->getRows(), tileDim );
      const IndexType columnTiles = roundUpDivision( this->getColumns(), tileDim );
      const IndexType cudaBlockColumns( tileDim );
      const IndexType cudaBlockRows( matrixProductCudaBlockSize / tileDim );
      cudaBlockSize.x = cudaBlockColumns;
      cudaBlockSize.y = cudaBlockRows;
      const IndexType rowGrids = roundUpDivision( rowTiles, Cuda::getMaxGridSize() );
      const IndexType columnGrids = roundUpDivision( columnTiles, Cuda::getMaxGridSize() );

      for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ )
         {
            cudaGridSize.x = cudaGridSize.y = Cuda::getMaxGridSize();
            if( gridIdx_x == columnGrids - 1 )
               cudaGridSize.x = columnTiles % Cuda::getMaxGridSize();
            if( gridIdx_y == rowGrids - 1 )
               cudaGridSize.y = rowTiles % Cuda::getMaxGridSize();
            DenseMatrixProductKernel< tileDim, cudaBlockRows, DenseMatrixView< RealType, DeviceType, IndexType >,
                                      MatrixView1, MatrixView2, RealType, IndexType >
               <<< cudaGridSize, cudaBlockSize >>>
               ( *this, matrix1, matrix2, matrix1Multiplicator, matrix2Multiplicator, gridIdx_x, gridIdx_y );
         }
#endif
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename MatrixView, int tileDim >
void DenseMatrixView< Real, Device, Index, Organization >::getTransposition( const MatrixView& matrixView,
                                                              const RealType& matrixMultiplicator )
{
    TNL_ASSERT( this->getColumns() == matrixView.getRows() &&
                this->getRows() == matrixView.getColumns(),
                std::cerr << "This matrix columns: " << this->getColumns() << std::endl
                          << "This matrix rows: " << this->getRows() << std::endl
                          << "That matrix columns: " << matrixView.getColumns() << std::endl
                          << "That matrix rows: " << matrixView.getRows() << std::endl );

    if( std::is_same< Device, Devices::Host >::value )
    {
        const IndexType& rows = matrixView.getRows();
        const IndexType& columns = matrixView.getColumns();
        for( IndexType i = 0; i < rows; i += tileDim )
            for( IndexType j = 0; j < columns; j += tileDim )
                for( IndexType k = i; k < i + tileDim && k < rows; k++ )
                    for( IndexType l = j; l < j + tileDim && l < columns; l++ )
                        this->setElement( l, k, matrixMultiplicator * matrixView.getElement( k, l ) );
    }
    if( std::is_same< Device, Devices::Cuda >::value )
    {
#ifdef HAVE_CUDA
      dim3 cudaBlockSize( 0 ), cudaGridSize( 0 );
      const IndexType matrixProductCudaBlockSize( 256 );
      const IndexType rowTiles = roundUpDivision( matrixView.getRows(), tileDim );
      const IndexType columnTiles = roundUpDivision( matrixView.getColumns(), tileDim );
      const IndexType cudaBlockColumns( tileDim );
      const IndexType cudaBlockRows( matrixProductCudaBlockSize / tileDim );
      cudaBlockSize.x = cudaBlockColumns;
      cudaBlockSize.y = cudaBlockRows;
      const IndexType rowGrids = roundUpDivision( rowTiles, Cuda::getMaxGridSize() );
      const IndexType columnGrids = roundUpDivision( columnTiles, Cuda::getMaxGridSize() );
      const IndexType sharedMemorySize = tileDim*tileDim + tileDim*tileDim/Cuda::getNumberOfSharedMemoryBanks();

      DenseMatrixView* this_device = Cuda::passToDevice( *this );
      MatrixView* matrix_device = Cuda::passToDevice( matrixView );

      for( IndexType gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( IndexType gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ )
         {
            cudaGridSize.x = cudaGridSize.y = Cuda::getMaxGridSize();
            if( gridIdx_x == columnGrids - 1)
               cudaGridSize.x = columnTiles % Cuda::getMaxGridSize();
            if( gridIdx_y == rowGrids - 1 )
               cudaGridSize.y = rowTiles % Cuda::getMaxGridSize();
            if( ( gridIdx_x < columnGrids - 1 || matrixView.getColumns() % tileDim == 0 ) &&
                ( gridIdx_y < rowGrids - 1 || matrixView.getRows() % tileDim == 0 ) )
            {
               DenseTranspositionAlignedKernel< Real,
                                                         Index,
                                                         MatrixView,
                                                         Organization,
                                                         tileDim,
                                                         cudaBlockRows >
                                                     <<< cudaGridSize,
                                                         cudaBlockSize,
                                                         sharedMemorySize  >>>
                                                       ( this_device,
                                                         matrix_device,
                                                         matrixMultiplicator,
                                                         gridIdx_x,
                                                         gridIdx_y );
            }
            else
            {
               DenseTranspositionNonAlignedKernel< Real,
                                                         Index,
                                                         Organization,
                                                         MatrixView,
                                                         tileDim,
                                                         cudaBlockRows >
                                                     <<< cudaGridSize,
                                                         cudaBlockSize,
                                                         sharedMemorySize  >>>
                                                       ( this_device,
                                                         matrix_device,
                                                         matrixMultiplicator,
                                                         gridIdx_x,
                                                         gridIdx_y );
            }
            TNL_CHECK_CUDA_DEVICE;
         }
      Cuda::freeFromDevice( this_device );
      Cuda::freeFromDevice( matrix_device );
#endif
    }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
   template< typename Vector1, typename Vector2 >
void DenseMatrixView< Real, Device, Index, Organization >::performSORIteration( const Vector1& b,
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
          ElementsOrganization Organization >
DenseMatrixView< Real, Device, Index, Organization >&
DenseMatrixView< Real, Device, Index, Organization >::
operator=( const DenseMatrixView& matrix )
{
   MatrixView< Real, Device, Index >::operator=( matrix );
   this->segments = matrix.segments;
   return *this;
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void DenseMatrixView< Real, Device, Index, Organization >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void DenseMatrixView< Real, Device, Index, Organization >::save( File& file ) const
{
   MatrixView< Real, Device, Index >::save( file );
   this->segments.save( file );
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
void DenseMatrixView< Real, Device, Index, Organization >::print( std::ostream& str ) const
{
   for( IndexType row = 0; row < this->getRows(); row++ )
   {
      str <<"Row: " << row << " -> ";
      for( IndexType column = 0; column < this->getColumns(); column++ )
      {
         std::stringstream str_;
         str_ << std::setw( 4 ) << std::right << column << ":" << std::setw( 4 ) << std::left << this->getElement( row, column );
         str << std::setw( 10 ) << str_.str();
      }
      str << std::endl;
   }
}

template< typename Real,
          typename Device,
          typename Index,
          ElementsOrganization Organization >
__cuda_callable__
Index DenseMatrixView< Real, Device, Index, Organization >::getElementIndex( const IndexType row,
                                                              const IndexType column ) const
{
   return this->segments.getGlobalIndex( row, column );
}


#ifdef HAVE_CUDA
template< int tileDim,
          int tileRowBlockSize,
          typename ResultMatrix,
          typename Matrix1,
          typename Matrix2,
          typename Real,
          typename Index >
__global__ void
DenseMatrixProductKernel( ResultMatrix resultMatrix,
                          const Matrix1 matrixA,
                          const Matrix2 matrixB,
                          const Real matrixAMultiplicator,
                          const Real matrixBMultiplicator,
                          const Index gridIdx_x,
                          const Index gridIdx_y )
{
    typedef Index IndexType;
    typedef Real RealType;

    const IndexType resultTileRow = ( gridIdx_y*gridDim.y + blockIdx.y )*tileDim;
    const IndexType resultTileColumn = ( gridIdx_x*gridDim.x + blockIdx.x )*tileDim;

    const IndexType& lastRow = TNL::min( resultMatrix.getRows(), resultTileRow + tileDim );
    if (blockIdx.y == gridDim.y - 1 && resultTileRow + threadIdx.y >= lastRow) {
        return;
    }

    const IndexType& lastColumn = TNL::min( resultMatrix.getColumns(), resultTileColumn + tileDim );
    if (blockIdx.x == gridDim.x - 1 && resultTileColumn + threadIdx.x >= lastColumn) {
        return;
    }

    const IndexType& matrixBColumn = resultTileColumn + threadIdx.x;
    const IndexType& matrixAColumns = matrixA.getColumns();

    for ( IndexType row = resultTileRow + threadIdx.y; row < lastRow; row += tileRowBlockSize ) {
        RealType sum = 0.0;
        for( IndexType i = 0; i < matrixAColumns; i++ )
        {
             sum +=
             matrixAMultiplicator * matrixA( row, i ) *
             matrixBMultiplicator * matrixB( i, matrixBColumn );
        }
        resultMatrix(row, resultTileColumn + threadIdx.x) = sum;
    }
}

template< typename Real,
          typename Index,
          typename Matrix,
          ElementsOrganization Organization,
          int tileDim,
          int tileRowBlockSize,
          typename Device >
__global__ void
DenseTranspositionAlignedKernel( DenseMatrixView< Real, Device, Index >* resultMatrix,
                                 const Matrix* inputMatrix,
                                 const Real matrixMultiplicator,
                                 const Index gridIdx_x,
                                 const Index gridIdx_y )
{
   __shared__ Real tile[ tileDim*tileDim + tileDim*tileDim/Cuda::getNumberOfSharedMemoryBanks() ];

   const Index columns = inputMatrix->getColumns();
   const Index rows = inputMatrix->getRows();


   /****
    * Diagonal mapping of the CUDA blocks
    */
   Index blockIdx_x, blockIdx_y;
   if( columns == rows )
   {
      blockIdx_y = blockIdx.x;
      blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
   }
   else
   {
      Index bID = blockIdx.x + gridDim.x*blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   /****
    * Read the tile to the shared memory
    */
   const Index readRowPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.y;
   const Index readColumnPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.x;
   for( Index rowBlock = 0;
        rowBlock < tileDim;
        rowBlock += tileRowBlockSize )
   {
      tile[ Cuda::getInterleaving( threadIdx.x*tileDim + threadIdx.y + rowBlock ) ] =
               inputMatrix->getElement( readRowPosition + rowBlock, readColumnPosition );
   }
   __syncthreads();

   /****
    * Write the tile to the global memory
    */
   const Index writeRowPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.y;
   const Index writeColumnPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.x;
   for( Index rowBlock = 0;
        rowBlock < tileDim;
        rowBlock += tileRowBlockSize )
   {
      resultMatrix->setElement( writeRowPosition + rowBlock,
                                    writeColumnPosition,
                                    matrixMultiplicator * tile[ Cuda::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ] );

   }

}

template< typename Real,
          typename Index,
          ElementsOrganization Organization,
          typename Matrix,
          int tileDim,
          int tileRowBlockSize,
          typename Device >
__global__ void
DenseTranspositionNonAlignedKernel( DenseMatrixView< Real, Device, Index >* resultMatrix,
                                    const Matrix* inputMatrix,
                                    const Real matrixMultiplicator,
                                    const Index gridIdx_x,
                                    const Index gridIdx_y )
{
   __shared__ Real tile[ tileDim*tileDim + tileDim*tileDim/Cuda::getNumberOfSharedMemoryBanks() ];

   const Index columns = inputMatrix->getColumns();
   const Index rows = inputMatrix->getRows();

   /****
    * Diagonal mapping of the CUDA blocks
    */
   Index blockIdx_x, blockIdx_y;
   if( columns == rows )
   {
      blockIdx_y = blockIdx.x;
      blockIdx_x = (blockIdx.x+blockIdx.y)%gridDim.x;
   }
   else
   {
      Index bID = blockIdx.x + gridDim.x*blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   /****
    * Read the tile to the shared memory
    */
   const Index readRowPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.y;
   const Index readColumnPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.x;
   if( readColumnPosition < columns )
   {
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize )
      {
         if( readRowPosition + rowBlock < rows )
            tile[ Cuda::getInterleaving( threadIdx.x*tileDim + threadIdx.y + rowBlock ) ] =
               inputMatrix->getElement( readRowPosition + rowBlock, readColumnPosition );
      }
   }
   __syncthreads();

   /****
    * Write the tile to the global memory
    */
   const Index writeColumnPosition =
      ( gridIdx_y*gridDim.y + blockIdx_y )*tileDim + threadIdx.x;
   const Index writeRowPosition =
      ( gridIdx_x*gridDim.x + blockIdx_x )*tileDim + threadIdx.y;
   if( writeColumnPosition < rows )
   {
      for( Index rowBlock = 0;
           rowBlock < tileDim;
           rowBlock += tileRowBlockSize )
      {
         if( writeRowPosition + rowBlock < columns )
            resultMatrix->setElement( writeRowPosition + rowBlock,
                                          writeColumnPosition,
                                          matrixMultiplicator * tile[ Cuda::getInterleaving( (threadIdx.y + rowBlock)*tileDim + threadIdx.x ) ] );
      }
   }

}
#endif

} // namespace Matrices
} // namespace TNL
