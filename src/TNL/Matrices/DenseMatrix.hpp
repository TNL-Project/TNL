// Copyright (c) 2004-2023 Tomáš Oberhuber et al.
//
// This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
//
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

template< int tileDim, int tileRowBlockSize, typename ResultMatrix, typename Matrix1, typename Matrix2 >
__global__
void
DenseMatrixProductKernel( ResultMatrix resultMatrix,
                          const Matrix1 matrixA,
                          const Matrix2 matrixB,
                          const typename ResultMatrix::RealType matrixMultiplicator,
                          const typename ResultMatrix::IndexType gridIdx_x,
                          const typename ResultMatrix::IndexType gridIdx_y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   // Here we compute product C = A * B. To profit from the fast
   // shared memory we do it by tiles.

   using IndexType = typename ResultMatrix::IndexType;
   using RealType = typename ResultMatrix::RealType;
   __shared__ RealType tileA[ tileDim * tileDim ];
   __shared__ RealType tileB[ tileDim * tileDim ];
   __shared__ RealType tileC[ tileDim * tileDim ];

   const IndexType& matrixARows = matrixA.getRows();
   const IndexType& matrixAColumns = matrixA.getColumns();
   const IndexType& matrixBRows = matrixB.getRows();
   const IndexType& matrixBColumns = matrixB.getColumns();

   // Reset the tile C
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize )
      tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] = 0.0;

   // Compute the result tile coordinates
   const IndexType resultTileRow = ( gridIdx_y * gridDim.y + blockIdx.y ) * tileDim;
   const IndexType resultTileColumn = ( gridIdx_x * gridDim.x + blockIdx.x ) * tileDim;

   // Sum over the matrix tiles
   for( IndexType i = 0; i < matrixAColumns; i += tileDim ) {
      for( IndexType row = 0; row < tileDim; row += tileRowBlockSize ) {
         const IndexType matrixARow = resultTileRow + threadIdx.y + row;
         const IndexType matrixAColumn = i + threadIdx.x;
         if( matrixARow < matrixARows && matrixAColumn < matrixAColumns )
            tileA[ ( threadIdx.y + row ) * tileDim + threadIdx.x ] = matrixA( matrixARow, matrixAColumn );

         const IndexType matrixBRow = i + threadIdx.y + row;
         const IndexType matrixBColumn = resultTileColumn + threadIdx.x;
         if( matrixBRow < matrixBRows && matrixBColumn < matrixBColumns )
            tileB[ ( threadIdx.y + row ) * tileDim + threadIdx.x ] = matrixB( matrixBRow, matrixBColumn );
      }
      __syncthreads();

      const IndexType tileALastRow = TNL::min( tileDim, matrixARows - resultTileRow );
      const IndexType tileALastColumn = TNL::min( tileDim, matrixAColumns - i );
      // const IndexType tileBLastRow = TNL::min( tileDim, matrixBRows - i );
      // const IndexType tileBLastColumn = TNL::min( tileDim, matrixBColumns - resultTileColumn );

      for( IndexType row = 0; row < tileALastRow; row += tileRowBlockSize ) {
         RealType sum( 0.0 );
         for( IndexType j = 0; j < tileALastColumn; j++ )
            sum += matrixMultiplicator * tileA[ ( threadIdx.y + row ) * tileDim + j ] * tileB[ j * tileDim + threadIdx.x ];
         tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ] += sum;
      }
      __syncthreads();
   }

   // Write the result tile to the result matrix
   const IndexType& matrixCRows = resultMatrix.getRows();
   const IndexType& matrixCColumns = resultMatrix.getColumns();
   for( IndexType row = 0; row < tileDim; row += tileRowBlockSize ) {
      const IndexType matrixCRow = resultTileRow + row + threadIdx.y;
      const IndexType matrixCColumn = resultTileColumn + threadIdx.x;
      if( matrixCRow < matrixCRows && matrixCColumn < matrixCColumns )
         resultMatrix( matrixCRow, matrixCColumn ) = tileC[ ( row + threadIdx.y ) * tileDim + threadIdx.x ];
   }
#endif
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization, typename RealAllocator >
template< typename Matrix1, typename Matrix2, int tileDim >
void
DenseMatrix< Real, Device, Index, Organization, RealAllocator >::getMatrixProduct( const Matrix1& matrix1,
                                                                                   const Matrix2& matrix2,
                                                                                   Real matrixMultiplicator )
{
   TNL_ASSERT_EQ( matrix1.getColumns(), matrix2.getRows(), "invalid dimensions of input matrices" );
   setDimensions( matrix1.getRows(), matrix2.getColumns() );

   if constexpr( std::is_same_v< Device, Devices::Host > ) {
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

            constexpr auto kernel = DenseMatrixProductKernel< tileDim,
                                                              cudaBlockRows,
                                                              ViewType,
                                                              typename Matrix1::ConstViewType,
                                                              typename Matrix2::ConstViewType >;
            Backend::launchKernelAsync( kernel,
                                        launch_config,
                                        getView(),
                                        matrix1.getConstView(),
                                        matrix2.getConstView(),
                                        matrixMultiplicator,
                                        gridIdx_x,
                                        gridIdx_y );
         }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< int tileDim, int tileRowBlockSize, typename OutputMatrix, typename InputMatrix, typename Real, typename Index >
__global__
void
DenseTranspositionAlignedKernel( OutputMatrix resultMatrix,
                                 const InputMatrix inputMatrix,
                                 const Real matrixMultiplicator,
                                 const Index gridIdx_x,
                                 const Index gridIdx_y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Real tile[ tileDim * tileDim ];

   const Index columns = inputMatrix.getColumns();
   const Index rows = inputMatrix.getRows();

   // Diagonal mapping of the CUDA blocks
   Index blockIdx_x, blockIdx_y;
   if( columns == rows ) {
      blockIdx_y = blockIdx.x;
      blockIdx_x = ( blockIdx.x + blockIdx.y ) % gridDim.x;
   }
   else {
      Index bID = blockIdx.x + gridDim.x * blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   // Read the tile to the shared memory
   const Index readRowPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.y;
   const Index readColumnPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.x;
   for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
      tile[ Backend::getInterleaving( threadIdx.x * tileDim + threadIdx.y + rowBlock ) ] =
         inputMatrix( readRowPosition + rowBlock, readColumnPosition );
   }
   __syncthreads();

   // Write the tile to the global memory
   const Index writeRowPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.y;
   const Index writeColumnPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.x;
   for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
      resultMatrix( writeRowPosition + rowBlock, writeColumnPosition ) =
         matrixMultiplicator * tile[ Backend::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ];
   }
#endif
}

template< int tileDim, int tileRowBlockSize, typename OutputMatrix, typename InputMatrix, typename Real, typename Index >
__global__
void
DenseTranspositionNonAlignedKernel( OutputMatrix resultMatrix,
                                    const InputMatrix inputMatrix,
                                    const Real matrixMultiplicator,
                                    const Index gridIdx_x,
                                    const Index gridIdx_y )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Real tile[ tileDim * tileDim ];

   const Index columns = inputMatrix.getColumns();
   const Index rows = inputMatrix.getRows();

   // Diagonal mapping of the CUDA blocks
   Index blockIdx_x, blockIdx_y;
   if( columns == rows ) {
      blockIdx_y = blockIdx.x;
      blockIdx_x = ( blockIdx.x + blockIdx.y ) % gridDim.x;
   }
   else {
      Index bID = blockIdx.x + gridDim.x * blockIdx.y;
      blockIdx_y = bID % gridDim.y;
      blockIdx_x = ( ( bID / gridDim.y ) + blockIdx_y ) % gridDim.x;
   }

   // Read the tile to the shared memory
   const Index readRowPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.y;
   const Index readColumnPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.x;
   if( readColumnPosition < columns ) {
      // const Index readOffset = readRowPosition * columns + readColumnPosition;
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( readRowPosition + rowBlock < rows )
            tile[ Backend::getInterleaving( threadIdx.x * tileDim + threadIdx.y + rowBlock ) ] =
               inputMatrix( readRowPosition + rowBlock, readColumnPosition );
      }
   }
   __syncthreads();

   // Write the tile to the global memory
   const Index writeRowPosition = ( gridIdx_x * gridDim.x + blockIdx_x ) * tileDim + threadIdx.y;
   const Index writeColumnPosition = ( gridIdx_y * gridDim.y + blockIdx_y ) * tileDim + threadIdx.x;
   if( writeColumnPosition < rows ) {
      // const Index writeOffset = writeRowPosition * rows + writeColumnPosition;
      for( Index rowBlock = 0; rowBlock < tileDim; rowBlock += tileRowBlockSize ) {
         if( writeRowPosition + rowBlock < columns )
            resultMatrix( writeRowPosition + rowBlock, writeColumnPosition ) =
               matrixMultiplicator * tile[ Backend::getInterleaving( ( threadIdx.y + rowBlock ) * tileDim + threadIdx.x ) ];
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

            if( ( gridIdx_x < columnGrids - 1 || matrix.getColumns() % tileDim == 0 )
                && ( gridIdx_y < rowGrids - 1 || matrix.getRows() % tileDim == 0 ) )
            {
               constexpr auto kernel = DenseTranspositionAlignedKernel< tileDim,
                                                                        cudaBlockRows,
                                                                        ViewType,
                                                                        typename Matrix::ConstViewType,
                                                                        Real,
                                                                        Index >;
               Backend::launchKernelAsync(
                  kernel, launch_config, getView(), matrix.getConstView(), matrixMultiplicator, gridIdx_x, gridIdx_y );
            }
            else {
               constexpr auto kernel = DenseTranspositionNonAlignedKernel< tileDim,
                                                                           cudaBlockRows,
                                                                           ViewType,
                                                                           typename Matrix::ConstViewType,
                                                                           Real,
                                                                           Index >;
               Backend::launchKernelAsync(
                  kernel, launch_config, getView(), matrix.getConstView(), matrixMultiplicator, gridIdx_x, gridIdx_y );
            }
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
