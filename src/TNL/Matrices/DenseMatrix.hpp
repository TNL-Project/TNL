// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Backend.h>

#include "DenseMatrix.h"
#include "SparseOperations.h"

namespace TNL::Matrices {

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix( const RealAllocatorType& allocator )
: values( allocator )
{}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix( const DenseMatrix& matrix )
: values( matrix.values ), segments( matrix.segments )
{
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), segments.getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix( Index rows,
                                                                              Index columns,
                                                                              const RealAllocatorType& allocator )
: values( allocator )
{
   this->setDimensions( rows, columns );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Value >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::DenseMatrix(
   std::initializer_list< std::initializer_list< Value > > data,
   const RealAllocatorType& allocator )
: values( allocator )
{
   this->setElements( data );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Value >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setElements(
   std::initializer_list< std::initializer_list< Value > > data )
{
   Index rows = data.size();
   Index columns = 0;
   for( auto row : data )
      columns = max( columns, row.size() );
   this->setDimensions( rows, columns );
   if constexpr( std::is_same_v< Device, Devices::Cuda > ) {
      DenseMatrix< Real, Devices::Host, Index > hostDense( rows, columns );
      Index rowIdx = 0;
      for( auto row : data ) {
         Index columnIdx = 0;
         for( auto element : row )
            hostDense.setElement( rowIdx, columnIdx++, element );
         rowIdx++;
      }
      *this = hostDense;
   }
   else {
      Index rowIdx = 0;
      for( auto row : data ) {
         Index columnIdx = 0;
         for( auto element : row )
            this->setElement( rowIdx, columnIdx++, element );
         rowIdx++;
      }
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename MapIndex, typename MapValue >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setElements(
   const std::map< std::pair< MapIndex, MapIndex >, MapValue >& map )
{
   if constexpr( ! std::is_same_v< Device, Devices::Host > ) {
      DenseMatrix< Real, Devices::Host, Index, Organization > hostMatrix( this->getRows(), this->getColumns() );
      hostMatrix.setElements( map );
      *this = hostMatrix;
   }
   else {
      for( const auto& [ coordinates, value ] : map ) {
         const auto& [ rowIdx, columnIdx ] = coordinates;
         if( rowIdx >= this->getRows() )
            throw std::logic_error( "Wrong row index " + std::to_string( rowIdx ) + " in the input data structure." );
         if( columnIdx >= this->getColumns() )
            throw std::logic_error( "Wrong column index " + std::to_string( columnIdx ) + " in the input data structure." );
         this->setElement( rowIdx, columnIdx, value );
      }
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getView() -> ViewType
{
   return { this->getRows(), this->getColumns(), this->getValues().getView() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
auto
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getConstView() const -> ConstViewType
{
   return { this->getRows(), this->getColumns(), this->getValues().getConstView() };
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setDimensions( Index rows, Index columns )
{
   this->segments.setSegmentsSizes( rows, columns );
   this->values.setSize( this->segments.getStorageSize() );
   this->values = 0.0;
   // update the base
   Base::bind( rows, columns, values.getView(), segments.getView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Matrix_ >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setLike( const Matrix_& matrix )
{
   this->setDimensions( matrix.getRows(), matrix.getColumns() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RowCapacitiesVector >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::setRowCapacities( const RowCapacitiesVector& rowCapacities )
{
   TNL_ASSERT_EQ( rowCapacities.getSize(), this->getRows(), "" );
   TNL_ASSERT_LE( max( rowCapacities ), this->getColumns(), "" );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::reset()
{
   this->values.reset();
   this->segments.reset();
   // update the base
   Base::bind( 0, 0, values.getView(), segments.getView() );
}

template< int tileDim, typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
DenseMatrixProductKernel( ResultMatrix resultMatrix,
                          const Matrix1 matrixA,
                          const Matrix2 matrixB,
                          const typename ResultMatrix::RealType matrixMultiplicator,
                          TransposeState TransposeA,
                          TransposeState TransposeB )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;

   __shared__ RealType tileA[ tileDim ][ tileDim + 1 ];
   __shared__ RealType tileB[ tileDim ][ tileDim + 1 ];

   IndexType bx = blockIdx.x, by = blockIdx.y;
   IndexType tx = threadIdx.x, ty = threadIdx.y;

   IndexType row, col;
   constexpr auto organization = resultMatrix.getOrganization();

   // Adjust row and col based on the matrix organization
   if constexpr( organization == Algorithms::Segments::ElementsOrganization::ColumnMajorOrder ) {
      row = by * tileDim + ty;  // For column-major
      col = bx * tileDim + tx;
   }
   else {  // For row-major order
      row = bx * tileDim + ty;
      col = by * tileDim + tx;
   }

   RealType CValue = 0;

   IndexType widthA = ( TransposeA == TransposeState::None ) ? matrixA.getColumns() : matrixA.getRows();
   IndexType heightA = ( TransposeA == TransposeState::None ) ? matrixA.getRows() : matrixA.getColumns();
   IndexType widthB = ( TransposeB == TransposeState::None ) ? matrixB.getColumns() : matrixB.getRows();
   IndexType heightB = ( TransposeB == TransposeState::None ) ? matrixB.getRows() : matrixB.getColumns();

   for( IndexType m = 0; m < ( tileDim + widthA - 1 ) / tileDim; ++m ) {
      // Load tileA from matrix A
      if( m * tileDim + tx < widthA && row < heightA ) {
         tileA[ ty ][ tx ] =
            ( TransposeA == TransposeState::None ) ? matrixA( row, m * tileDim + tx ) : matrixA( m * tileDim + tx, row );
      }
      else {
         tileA[ ty ][ tx ] = 0.0;
      }

      // Load tileB from matrix B
      if( m * tileDim + ty < heightB && col < widthB ) {
         tileB[ ty ][ tx ] =
            ( TransposeB == TransposeState::None ) ? matrixB( m * tileDim + ty, col ) : matrixB( col, m * tileDim + ty );
      }
      else {
         tileB[ ty ][ tx ] = 0.0;
      }

      __syncthreads();

   // Perform the multiplication for the current tile
   #pragma unroll
      for( IndexType k = 0; k < tileDim; ++k ) {
         CValue += tileA[ ty ][ k ] * tileB[ k ][ tx ];
      }

      __syncthreads();
   }

   // Write the computed value back to the result matrix
   if( row < resultMatrix.getRows() && col < resultMatrix.getColumns() ) {
      resultMatrix( row, col ) = CValue * matrixMultiplicator;
   }
#endif  // __CUDACC__
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Matrix1, typename Matrix2, int tileDim >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getMatrixProduct( const Matrix1& matrix1,
                                                                                   const Matrix2& matrix2,
                                                                                   Real matrixMultiplicator,
                                                                                   TransposeState transposeA,
                                                                                   TransposeState transposeB )
{
   // Determine dimensions based on transpose states
   Index aRows = ( transposeA == TransposeState::None ) ? matrix1.getRows() : matrix1.getColumns();
   Index aCols = ( transposeA == TransposeState::None ) ? matrix1.getColumns() : matrix1.getRows();

   Index bRows = ( transposeB == TransposeState::None ) ? matrix2.getRows() : matrix2.getColumns();
   Index bCols = ( transposeB == TransposeState::None ) ? matrix2.getColumns() : matrix2.getRows();

   // Check for dimension compatibility
   if( aCols != bRows )
      throw std::invalid_argument( "invalid dimensions of input matrices" );

   // Adjust the dimensions of the result matrix
   setDimensions( aRows, bCols );

   if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > ) {
      for( Index i = 0; i < this->getRows(); i += tileDim )
         for( Index j = 0; j < this->getColumns(); j += tileDim ) {
            const Index tileRows = min( tileDim, this->getRows() - i );
            const Index tileColumns = min( tileDim, this->getColumns() - j );
            for( Index i1 = i; i1 < i + tileRows; i1++ )
               for( Index j1 = j; j1 < j + tileColumns; j1++ )
                  this->operator()( i1, j1 ) = 0.0;

            for( Index k = 0; k < matrix1.getColumns(); k += tileDim ) {
               const Index lastK = min( k + tileDim, matrix1.getColumns() );
               for( Index i1 = 0; i1 < tileRows; i1++ )
                  for( Index j1 = 0; j1 < tileColumns; j1++ )
                     for( Index k1 = k; k1 < lastK; k1++ )
                        this->operator()( i + i1, j + j1 ) +=
                           matrixMultiplicator * matrix1( i + i1, k1 ) * matrix2( k1, j + j1 );
            }
         }
   }
   if constexpr( std::is_same_v< Device, Devices::Cuda > ) {
      constexpr Index matrixProductCudaBlockSize = 256;
      constexpr Index cudaBlockRows = matrixProductCudaBlockSize / tileDim;
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = tileDim;
      launch_config.blockSize.y = cudaBlockRows;
      launch_config.dynamicSharedMemorySize = 3 * tileDim * tileDim;

      const Index rowTiles = roundUpDivision( this->getRows(), tileDim );
      const Index columnTiles = roundUpDivision( this->getColumns(), tileDim );
      const Index rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
      const Index columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

      for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
            launch_config.gridSize.x = Backend::getMaxGridXSize();
            launch_config.gridSize.y = Backend::getMaxGridYSize();
            if( gridIdx_x == columnGrids - 1 )
               launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
            if( gridIdx_y == rowGrids - 1 )
               launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

            constexpr auto kernel =
               DenseMatrixProductKernel< tileDim, ViewType, typename Matrix1::ConstViewType, typename Matrix2::ConstViewType >;
            Backend::launchKernelAsync( kernel,
                                        launch_config,
                                        getView(),
                                        matrix1.getConstView(),
                                        matrix2.getConstView(),
                                        matrixMultiplicator,
                                        transposeA,
                                        transposeB

            );
         }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< int tileDim, typename OutputMatrix, typename InputMatrix, typename Real, typename Index >
__global__
void
DenseTranspositionKernel( OutputMatrix resultMatrix, const InputMatrix inputMatrix, const Real matrixMultiplicator )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Real tile[ tileDim ][ tileDim + 1 ];

   constexpr auto organization = inputMatrix.getOrganization();

   Index row = blockIdx.y * tileDim + threadIdx.y;
   Index col = blockIdx.x * tileDim + threadIdx.x;

   // Checks ensuring that only valid indices are processed for both loading from the input matrix to the shared memory tile
   if constexpr( organization == Algorithms::Segments::ElementsOrganization::ColumnMajorOrder ) {
      // Column-major order
      if( row < inputMatrix.getRows() && col < inputMatrix.getColumns() ) {
         tile[ threadIdx.y ][ threadIdx.x ] = inputMatrix( row, col ) * matrixMultiplicator;
      }
   }
   else {  // Row-major order
      if( row < inputMatrix.getRows() && col < inputMatrix.getColumns() ) {
         tile[ threadIdx.x ][ threadIdx.y ] = inputMatrix( row, col ) * matrixMultiplicator;
      }
   }

   __syncthreads();

   // Adjust writing based on result matrix organization
   row = blockIdx.x * tileDim + threadIdx.y;
   col = blockIdx.y * tileDim + threadIdx.x;

   if constexpr( organization == Algorithms::Segments::ElementsOrganization::ColumnMajorOrder ) {
      // Column-major order
      if( row < resultMatrix.getRows() && col < resultMatrix.getColumns() ) {
         resultMatrix( row, col ) = tile[ threadIdx.x ][ threadIdx.y ];
      }
   }
   else {  // Row-major order
      if( row < resultMatrix.getRows() && col < resultMatrix.getColumns() ) {
         resultMatrix( row, col ) = tile[ threadIdx.y ][ threadIdx.x ];
      }
   }
#endif
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Matrix, int tileDim >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getTransposition( const Matrix& matrix,
                                                                                   Real matrixMultiplicator )
{
   setDimensions( matrix.getColumns(), matrix.getRows() );

   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      const Index& rows = matrix.getRows();
      const Index& columns = matrix.getColumns();
      for( Index i = 0; i < rows; i += tileDim )
         for( Index j = 0; j < columns; j += tileDim )
            for( Index k = i; k < i + tileDim && k < rows; k++ )
               for( Index l = j; l < j + tileDim && l < columns; l++ )
                  this->setElement( l, k, matrixMultiplicator * matrix.getElement( k, l ) );
   }
   if constexpr( std::is_same_v< Device, Devices::Cuda > ) {
      constexpr Index matrixProductCudaBlockSize = 256;
      constexpr Index cudaBlockRows = matrixProductCudaBlockSize / tileDim;
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = tileDim;
      launch_config.blockSize.y = cudaBlockRows;
      launch_config.dynamicSharedMemorySize = tileDim * tileDim + tileDim * tileDim / Backend::getNumberOfSharedMemoryBanks();

      const Index rowTiles = roundUpDivision( this->getRows(), tileDim );
      const Index columnTiles = roundUpDivision( this->getColumns(), tileDim );
      const Index rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
      const Index columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

      for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
            launch_config.gridSize.x = Backend::getMaxGridXSize();
            launch_config.gridSize.y = Backend::getMaxGridYSize();
            if( gridIdx_x == columnGrids - 1 )
               launch_config.gridSize.x = columnTiles % Backend::getMaxGridXSize();
            if( gridIdx_y == rowGrids - 1 )
               launch_config.gridSize.y = rowTiles % Backend::getMaxGridYSize();

            constexpr auto kernel = DenseTranspositionKernel< tileDim, ViewType, typename Matrix::ConstViewType, Real, Index >;
            Backend::launchKernelAsync( kernel, launch_config, getView(), matrix.getConstView(), matrixMultiplicator );
         }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrix< Real, Device, Index, Organization, RealAllocator >& matrix )
{
   return this->operator=( matrix.getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   DenseMatrix< Real, Device, Index, Organization, RealAllocator >&& matrix ) noexcept( false )
{
   this->values = std::move( matrix.values );
   this->segments = std::move( matrix.segments );
   // update the base
   Base::bind( matrix.getRows(), matrix.getColumns(), values.getView(), segments.getView() );
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal, typename RHSDevice, typename RHSIndex, typename RHSRealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrix< RHSReal, RHSDevice, RHSIndex, Organization, RHSRealAllocator >& matrix )
{
   return this->operator=( matrix.getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal, typename RHSDevice, typename RHSIndex >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, Organization >& matrix )
{
   this->setLike( matrix );
   this->values = matrix.getValues();
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal,
          typename RHSDevice,
          typename RHSIndex,
          ElementsOrganization RHSOrganization,
          typename RHSRealAllocator >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrix< RHSReal, RHSDevice, RHSIndex, RHSOrganization, RHSRealAllocator >& matrix )
{
   return this->operator=( matrix.getConstView() );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSReal, typename RHSDevice, typename RHSIndex, ElementsOrganization RHSOrganization >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=(
   const DenseMatrixView< RHSReal, RHSDevice, RHSIndex, RHSOrganization >& matrix )
{
   copyDenseToDenseMatrix( *this, matrix );
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename RHSMatrix >
DenseMatrix< Real, Device, Index, Organization, RealAllocator >&
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::operator=( const RHSMatrix& matrix )
{
   copySparseToDenseMatrix( *this, matrix );
   return *this;
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::save( const String& fileName ) const
{
   Object::save( fileName );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::load( const String& fileName )
{
   Object::load( fileName );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::save( File& file ) const
{
   file.save( &this->rows );
   file.save( &this->columns );
   file << values;
   segments.save( file );
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::load( File& file )
{
   Index rows = 0;
   Index columns = 0;
   file.load( &rows );
   file.load( &columns );
   file >> values;
   segments.load( file );
   // update the base
   Base::bind( rows, columns, values.getView(), segments.getView() );
}

}  // namespace TNL::Matrices
