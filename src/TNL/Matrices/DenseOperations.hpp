#pragma once

#include <TNL/Backend.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Hip.h>
#include <TNL/Devices/Cuda.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Devices/Host.h>

#include "DenseOperations.h"

namespace TNL::Matrices {

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

   IndexType row = by * tileDim + ty;
   IndexType col = bx * tileDim + tx;

   RealType CValue = 0;

   const IndexType& widthA = ( TransposeA == TransposeState::None ) ? matrixA.getColumns() : matrixA.getRows();
   const IndexType& heightA = ( TransposeA == TransposeState::None ) ? matrixA.getRows() : matrixA.getColumns();
   const IndexType& widthB = ( TransposeB == TransposeState::None ) ? matrixB.getColumns() : matrixB.getRows();
   const IndexType& heightB = ( TransposeB == TransposeState::None ) ? matrixB.getRows() : matrixB.getColumns();

   const IndexType numPhases = ( tileDim + widthA - 1 ) / tileDim;

   for( IndexType m = 0; m < numPhases; ++m ) {
      IndexType aCols = m * tileDim + tx;
      IndexType bRows = m * tileDim + ty;

      // Pre-determine if threads are within valid range
      bool loadTileA = aCols < widthA && row < heightA;
      bool loadTileB = bRows < heightB && col < widthB;

      // Load tileA from matrix A if within bounds, else initialize to 0
      tileA[ ty ][ tx ] =
         loadTileA ? ( TransposeA == TransposeState::None ? matrixA( row, aCols ) : matrixA( aCols, row ) ) : 0.0;

      // Load tileB from matrix B if within bounds, else initialize to 0
      tileB[ ty ][ tx ] =
         loadTileB ? ( TransposeB == TransposeState::None ? matrixB( bRows, col ) : matrixB( col, bRows ) ) : 0.0;

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

template< typename ResultMatrix, typename Matrix1, typename Matrix2, typename Real, int tileDim >
void
getMatrixProduct( ResultMatrix& resultMatrix,
                  const Matrix1& matrix1,
                  const Matrix2& matrix2,
                  Real matrixMultiplicator,
                  TransposeState transposeA,
                  TransposeState transposeB )
{
   using Index = typename ResultMatrix::IndexType;
   using Device = typename ResultMatrix::DeviceType;

   // Determine dimensions based on transpose states
   Index aRows = ( transposeA == TransposeState::None ) ? matrix1.getRows() : matrix1.getColumns();
   Index aCols = ( transposeA == TransposeState::None ) ? matrix1.getColumns() : matrix1.getRows();

   Index bRows = ( transposeB == TransposeState::None ) ? matrix2.getRows() : matrix2.getColumns();
   Index bCols = ( transposeB == TransposeState::None ) ? matrix2.getColumns() : matrix2.getRows();

   // Check for dimension compatibility
   if( aCols != bRows )
      throw std::invalid_argument( "invalid dimensions of input matrices" );

   // Adjust the dimensions of the result matrix
   resultMatrix.setDimensions( aRows, bCols );

   if constexpr( std::is_same_v< Device, Devices::Host > || std::is_same_v< Device, Devices::Sequential > ) {
      for( Index i = 0; i < resultMatrix.getRows(); i += tileDim )
         for( Index j = 0; j < resultMatrix.getColumns(); j += tileDim ) {
            const Index tileRows = min( tileDim, resultMatrix.getRows() - i );
            const Index tileColumns = min( tileDim, resultMatrix.getColumns() - j );
            for( Index i1 = i; i1 < i + tileRows; i1++ )
               for( Index j1 = j; j1 < j + tileColumns; j1++ )
                  resultMatrix.operator()( i1, j1 ) = 0.0;

            for( Index k = 0; k < matrix1.getColumns(); k += tileDim ) {
               const Index lastK = min( k + tileDim, matrix1.getColumns() );
               for( Index i1 = 0; i1 < tileRows; i1++ )
                  for( Index j1 = 0; j1 < tileColumns; j1++ )
                     for( Index k1 = k; k1 < lastK; k1++ )
                        resultMatrix.operator()( i + i1, j + j1 ) +=
                           matrixMultiplicator * matrix1( i + i1, k1 ) * matrix2( k1, j + j1 );
            }
         }
   }
   if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = tileDim;
      launch_config.blockSize.y = tileDim;
      launch_config.dynamicSharedMemorySize = 2 * tileDim * ( tileDim + 1 ) * sizeof( Real );

      const Index rowTiles = roundUpDivision( resultMatrix.getRows(), tileDim );
      const Index columnTiles = roundUpDivision( resultMatrix.getColumns(), tileDim );
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
                                                              typename ResultMatrix::ViewType,
                                                              typename Matrix1::ConstViewType,
                                                              typename Matrix2::ConstViewType >;
            Backend::launchKernelAsync( kernel,
                                        launch_config,
                                        resultMatrix.getView(),
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

   const Index matrixColumns = inputMatrix.getColumns();
   const Index matrixRows = inputMatrix.getRows();

   Index row = blockIdx.y * tileDim + threadIdx.y;
   Index col = blockIdx.x * tileDim + threadIdx.x;

   if( row < matrixRows && col < matrixColumns ) {
      tile[ threadIdx.y ][ threadIdx.x ] = inputMatrix( row, col ) * matrixMultiplicator;
   }

   __syncthreads();

   // Adjust writing based on result matrix organization
   row = blockIdx.x * tileDim + threadIdx.y;
   col = blockIdx.y * tileDim + threadIdx.x;

   if( row < matrixColumns && col < matrixRows ) {
      resultMatrix( row, col ) = tile[ threadIdx.x ][ threadIdx.y ];
   }

#endif
}

template< typename ResultMatrix, typename Matrix, typename Real, int tileDim >
void
getTransposition( ResultMatrix& resultMatrix, const Matrix& matrix, Real matrixMultiplicator )
{
   using Index = typename ResultMatrix::IndexType;
   using Device = typename ResultMatrix::DeviceType;
   resultMatrix.setDimensions( matrix.getColumns(), matrix.getRows() );

   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      const Index& rows = matrix.getRows();
      const Index& columns = matrix.getColumns();
      for( Index i = 0; i < rows; i += tileDim )
         for( Index j = 0; j < columns; j += tileDim )
            for( Index k = i; k < i + tileDim && k < rows; k++ )
               for( Index l = j; l < j + tileDim && l < columns; l++ )
                  resultMatrix.setElement( l, k, matrixMultiplicator * matrix.getElement( k, l ) );
   }
   if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = tileDim;
      launch_config.blockSize.y = tileDim;
      launch_config.dynamicSharedMemorySize = tileDim * ( tileDim + 1 ) * sizeof( Real );

      const Index rowTiles = roundUpDivision( resultMatrix.getRows(), tileDim );
      const Index columnTiles = roundUpDivision( resultMatrix.getColumns(), tileDim );
      const Index rowGrids = roundUpDivision( rowTiles, Backend::getMaxGridYSize() );
      const Index columnGrids = roundUpDivision( columnTiles, Backend::getMaxGridXSize() );

      for( Index gridIdx_x = 0; gridIdx_x < columnGrids; gridIdx_x++ )
         for( Index gridIdx_y = 0; gridIdx_y < rowGrids; gridIdx_y++ ) {
            launch_config.gridSize.x = Backend::getMaxGridXSize();
            launch_config.gridSize.y = Backend::getMaxGridYSize();
            if( gridIdx_x == columnGrids - 1 ) {
               auto remainder = columnTiles % Backend::getMaxGridXSize();
               launch_config.gridSize.x = ( remainder == 0 ) ? Backend::getMaxGridXSize() : remainder;
            }
            if( gridIdx_y == rowGrids - 1 ) {
               auto remainder = rowTiles % Backend::getMaxGridYSize();
               launch_config.gridSize.y = ( remainder == 0 ) ? Backend::getMaxGridYSize() : remainder;
            }

            constexpr auto kernel =
               DenseTranspositionKernel< tileDim, typename ResultMatrix::ViewType, typename Matrix::ConstViewType, Real, Index >;
            Backend::launchKernelAsync(
               kernel, launch_config, resultMatrix.getView(), matrix.getConstView(), matrixMultiplicator );
         }
      Backend::streamSynchronize( launch_config.stream );
   }
}

template< int tileDim, typename Matrix, typename Real, typename Index >
__global__
void
DenseInPlaceTranspositionKernel( Matrix matrix, const Real matrixMultiplicator )
{
#if defined( __CUDACC__ ) || defined( __HIP__ )
   __shared__ Real tile[ tileDim ][ tileDim + 1 ];

   const Index matrixColumns = matrix.getColumns();
   const Index matrixRows = matrix.getRows();

   Index xIndex = blockIdx.x * tileDim + threadIdx.x;
   Index yIndex = blockIdx.y * tileDim + threadIdx.y;

   if( xIndex < matrixColumns && yIndex < matrixRows ) {
      tile[ threadIdx.y ][ threadIdx.x ] = matrix( yIndex, xIndex ) * matrixMultiplicator;
   }

   __syncthreads();

   xIndex = blockIdx.y * tileDim + threadIdx.x;
   yIndex = blockIdx.x * tileDim + threadIdx.y;

   if( xIndex < matrixRows && yIndex < matrixColumns ) {
      matrix( yIndex, xIndex ) = tile[ threadIdx.x ][ threadIdx.y ];
   }
#endif
}

template< typename Matrix, typename Real, int tileDim >
void
getInPlaceTransposition( Matrix& matrix, Real matrixMultiplicator )
{
   using Index = typename Matrix::IndexType;
   using Device = typename Matrix::DeviceType;

   if( matrix.getRows() != matrix.getColumns() ) {
      throw std::logic_error( "In-place transposition on CPU only supports square matrices." );
   }
   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      const Index rows = matrix.getRows();
      const Index columns = matrix.getColumns();

      // Performing in-place transposition for square matrices
      for( Index i = 0; i < rows; ++i ) {
         for( Index j = i + 1; j < columns; ++j ) {
            Real temp = matrix.getElement( i, j );
            matrix.setElement( i, j, matrix.getElement( j, i ) );
            matrix.setElement( j, i, temp );
         }
      }
   }

   if constexpr( std::is_same_v< Device, Devices::Cuda > || std::is_same_v< Device, Devices::Hip > ) {
      Backend::LaunchConfiguration launch_config;
      launch_config.blockSize.x = tileDim;
      launch_config.blockSize.y = tileDim;
      launch_config.dynamicSharedMemorySize = tileDim * tileDim + tileDim * tileDim / Backend::getNumberOfSharedMemoryBanks();

      const Index rowTiles = roundUpDivision( matrix.getRows(), tileDim );
      const Index columnTiles = roundUpDivision( matrix.getColumns(), tileDim );
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

            auto kernel = DenseInPlaceTranspositionKernel< tileDim, decltype( matrix.getView() ), Real, Index >;
            Backend::launchKernelAsync( kernel, launch_config, matrix.getView(), matrixMultiplicator );
         }

      Backend::streamSynchronize( launch_config.stream );
   }
}

}  // namespace TNL::Matrices
