// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/ReductionLaunchConfigurations.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Benchmarks/Benchmark.h>
#include <TNL/Config/ParameterContainer.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/GPU.h>
#include <TNL/Devices/Host.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Matrices/MatrixReader.h>
#include <TNL/Matrices/SparseDenseOperations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/TypeInfo.h>

#include "ReferenceKernels.h"
#include "CuSparseSpMM.h"
#include "HipSparseSpMM.h"

namespace TNL::Benchmarks::SpMM {

/**
 * \brief Benchmark result extended with GFLOPS and the relative error of the
 * computed product with respect to the reference result.
 */
struct SpMMResult : public BenchmarkResult
{
   SpMMResult( double flops, std::function< double() > relativeError )
   : flops( flops ),
     relativeError( std::move( relativeError ) )
   {}

   [[nodiscard]] HeaderElements
   getTableHeader() const override
   {
      HeaderElements headers = BenchmarkResult::getTableHeader();
      headers.push_back( "GFLOPS" );
      headers.push_back( "rel. error" );
      return headers;
   }

   [[nodiscard]] RowElements
   getRowElements() const override
   {
      RowElements elements = BenchmarkResult::getRowElements();
      elements << flops / time / 1e9 << relativeError();
      return elements;
   }

   double flops;
   std::function< double() > relativeError;
};

// Tiles of columns for which the RowGroup kernel is instantiated.
inline constexpr int rowGroupTiles[] = { 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 32 };

template< typename Function >
void
dispatchTile( int K, Function&& f )
{
   switch( K ) {
      case 1:
         f( std::integral_constant< int, 1 >{} );
         break;
      case 2:
         f( std::integral_constant< int, 2 >{} );
         break;
      case 3:
         f( std::integral_constant< int, 3 >{} );
         break;
      case 4:
         f( std::integral_constant< int, 4 >{} );
         break;
      case 5:
         f( std::integral_constant< int, 5 >{} );
         break;
      case 6:
         f( std::integral_constant< int, 6 >{} );
         break;
      case 7:
         f( std::integral_constant< int, 7 >{} );
         break;
      case 8:
         f( std::integral_constant< int, 8 >{} );
         break;
      case 12:
         f( std::integral_constant< int, 12 >{} );
         break;
      case 16:
         f( std::integral_constant< int, 16 >{} );
         break;
      case 32:
         f( std::integral_constant< int, 32 >{} );
         break;
      default:
         throw std::invalid_argument( "dispatchTile: unsupported tile " + std::to_string( K ) );
   }
}

template< typename Function >
void
dispatchPowerOfTwo( int value, Function&& f )
{
   switch( value ) {
      case 1:
         f( std::integral_constant< int, 1 >{} );
         break;
      case 2:
         f( std::integral_constant< int, 2 >{} );
         break;
      case 4:
         f( std::integral_constant< int, 4 >{} );
         break;
      case 8:
         f( std::integral_constant< int, 8 >{} );
         break;
      case 16:
         f( std::integral_constant< int, 16 >{} );
         break;
      case 32:
         f( std::integral_constant< int, 32 >{} );
         break;
      default:
         throw std::invalid_argument( "dispatchPowerOfTwo: unsupported value " + std::to_string( value ) );
   }
}

/**
 * \brief Tiles of columns used with the RowGroup kernel for k dense columns.
 *
 * The fused variant (K == k) is used whenever it is instantiated, the tiled
 * variants are used for all tiles smaller than k.
 */
inline std::vector< int >
rowGroupTilesFor( int k )
{
   std::vector< int > tiles;
   for( int K : rowGroupTiles )
      if( K == k || ( K >= 4 && ( K & ( K - 1 ) ) == 0 && K < k ) )
         tiles.push_back( K );
   return tiles;
}

/**
 * \brief Generates the matrix of a finite difference stencil on a uniform grid
 * with Dirichlet boundary conditions.
 *
 * \param stencil is `laplace-2d` (5-point), `laplace-3d` (7-point) or `stencil-27` (27-point in 3D).
 */
template< typename Matrix >
void
generateStencilMatrix( Matrix& matrix, const std::string& stencil, int gridSize )
{
   using Index = typename Matrix::IndexType;
   using Real = typename Matrix::RealType;

   std::vector< std::array< int, 3 > > offsets;
   Real diagonal = 0;
   int dimension = 3;
   if( stencil == "laplace-2d" ) {
      dimension = 2;
      offsets = { { { 0, -1, 0 } }, { { -1, 0, 0 } }, { { 0, 0, 0 } }, { { 1, 0, 0 } }, { { 0, 1, 0 } } };
      diagonal = 4;
   }
   else if( stencil == "laplace-3d" ) {
      offsets = { { { 0, 0, -1 } }, { { 0, -1, 0 } }, { { -1, 0, 0 } }, { { 0, 0, 0 } },
                  { { 1, 0, 0 } },  { { 0, 1, 0 } },  { { 0, 0, 1 } } };
      diagonal = 6;
   }
   else if( stencil == "stencil-27" ) {
      for( int z = -1; z <= 1; z++ )
         for( int y = -1; y <= 1; y++ )
            for( int x = -1; x <= 1; x++ )
               offsets.push_back( { { x, y, z } } );
      diagonal = 26;
   }
   else
      throw std::invalid_argument( "Unknown stencil " + stencil );

   const Index nx = gridSize;
   const Index ny = gridSize;
   const Index nz = dimension == 2 ? 1 : gridSize;
   const Index size = nx * ny * nz;

   auto inside = [ & ]( Index x, Index y, Index z )
   {
      return x >= 0 && x < nx && y >= 0 && y < ny && z >= 0 && z < nz;
   };

   Containers::Vector< Index, Devices::Host, Index > capacities( size );
   for( Index z = 0; z < nz; z++ )
      for( Index y = 0; y < ny; y++ )
         for( Index x = 0; x < nx; x++ ) {
            Index count = 0;
            for( const auto& o : offsets )
               count += inside( x + o[ 0 ], y + o[ 1 ], z + o[ 2 ] );
            capacities[ ( z * ny + y ) * nx + x ] = count;
         }

   matrix.setDimensions( size, size );
   matrix.setRowCapacities( capacities );
   for( Index z = 0; z < nz; z++ )
      for( Index y = 0; y < ny; y++ )
         for( Index x = 0; x < nx; x++ ) {
            auto row = matrix.getRow( ( z * ny + y ) * nx + x );
            Index localIdx = 0;
            for( const auto& o : offsets )
               if( inside( x + o[ 0 ], y + o[ 1 ], z + o[ 2 ] ) ) {
                  const bool isDiagonal = o[ 0 ] == 0 && o[ 1 ] == 0 && o[ 2 ] == 0;
                  const Index column = ( ( z + o[ 2 ] ) * ny + y + o[ 1 ] ) * nx + x + o[ 0 ];
                  row.setElement( localIdx++, column, isDiagonal ? diagonal : Real( -1 ) );
               }
         }
}

//! \brief Value of the element (i, j) of the dense input matrix B.
template< typename Real, typename Index >
Real
denseInputValue( Index i, Index j )
{
   return static_cast< Real >( ( i * 7 + j * 13 ) % 17 - 8 ) / Real( 8 );
}

/**
 * \brief Data shared by all benchmarked implementations for one sparse matrix,
 * one number of dense columns and one dense layout.
 *
 * B and C are stored in plain vectors, the layout is given by `rowMajor`.
 * The reference result is stored in row-major order on the host.
 */
template< typename Real, typename Index >
struct SpMMProblem
{
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;

   Index rows = 0;
   Index columns = 0;
   Index nonzeros = 0;
   Index k = 0;
   bool rowMajor = true;
   HostVector reference;

   [[nodiscard]] Index
   leadingDimensionB() const
   {
      return rowMajor ? k : columns;
   }

   [[nodiscard]] Index
   leadingDimensionC() const
   {
      return rowMajor ? k : rows;
   }

   [[nodiscard]] HostVector
   makeB() const
   {
      HostVector B( columns * k );
      for( Index i = 0; i < columns; i++ )
         for( Index j = 0; j < k; j++ )
            B[ rowMajor ? i * k + j : j * columns + i ] = denseInputValue< Real >( i, j );
      return B;
   }

   //! \brief Returns the relative max-norm error of C with respect to the reference.
   template< typename Vector >
   [[nodiscard]] double
   relativeError( const Vector& C ) const
   {
      const HostVector hostC( C );
      double maxDiff = 0;
      double maxRef = 0;
      for( Index i = 0; i < rows; i++ )
         for( Index j = 0; j < k; j++ ) {
            const double ref = reference[ i * k + j ];
            const double value = hostC[ rowMajor ? i * k + j : j * rows + i ];
            if( std::isnan( value ) )
               return std::numeric_limits< double >::quiet_NaN();
            maxDiff = std::max( maxDiff, std::abs( value - ref ) );
            maxRef = std::max( maxRef, std::abs( ref ) );
         }
      return maxRef > 0 ? maxDiff / maxRef : maxDiff;
   }
};

/**
 * \brief Computes C = A * B on the host, parallelized over rows.
 *
 * Used both as the host implementation and to compute the reference result.
 */
template< typename Real, typename Index >
void
spmmHost( const CSRArrays< Real, Index >& A, const DenseArray< Real, Index >& B, DenseArray< Real, Index >& C, bool rowMajor )
{
   const Index k = C.columns;
   auto kernel = [ = ]( Index row ) mutable
   {
      auto index = [ rowMajor ]( Index i, Index j, Index ld )
      {
         return rowMajor ? i * ld + j : j * ld + i;
      };
      for( Index j = 0; j < k; j++ )
         C.data[ index( row, j, C.leadingDimension ) ] = 0;
      for( Index i = A.rowPointers[ row ]; i < A.rowPointers[ row + 1 ]; i++ ) {
         const Index column = A.columnIndexes[ i ];
         const Real value = A.values[ i ];
         for( Index j = 0; j < k; j++ )
            C.data[ index( row, j, C.leadingDimension ) ] += value * B.data[ index( column, j, B.leadingDimension ) ];
      }
   };
   Algorithms::parallelFor< Devices::Host >( Index( 0 ), A.rows, kernel );
}

template< typename Matrix >
CSRArrays< typename Matrix::RealType, typename Matrix::IndexType >
getCSRArrays( const Matrix& matrix )
{
   CSRArrays< typename Matrix::RealType, typename Matrix::IndexType > arrays;
   arrays.rowPointers = matrix.getSegments().getOffsets().getData();
   arrays.columnIndexes = matrix.getColumnIndexes().getData();
   arrays.values = matrix.getValues().getData();
   arrays.rows = matrix.getRows();
   arrays.columns = matrix.getColumns();
   arrays.nonzeros = matrix.getColumnIndexes().getSize();
   return arrays;
}

template< typename Real, typename Index, typename Vector >
DenseArray< Real, Index >
getDenseArray( Vector& data, Index rows, Index columns, Index leadingDimension )
{
   DenseArray< Real, Index > array;
   array.data = const_cast< Real* >( data.getData() );  // B is never written
   array.rows = rows;
   array.columns = columns;
   array.leadingDimension = leadingDimension;
   return array;
}

//! \brief Tells whether the implementation is selected by the `kernels` parameter.
inline bool
isSelected( const std::vector< std::string >& kernels, const std::string& kernel )
{
   return std::find( kernels.begin(), kernels.end(), "all" ) != kernels.end()
       || std::find( kernels.begin(), kernels.end(), kernel ) != kernels.end();
}

/**
 * \brief Returns the permutation of the rows sorted by their lengths in
 * descending order within windows of consecutive rows.
 *
 * Sorting all rows balances the work of the groups of threads best (row
 * swizzle of Sputnik), sorting within smaller windows preserves the locality
 * of the accesses to B (like the sorting scope of SELL-C-sigma).
 *
 * \param window is the number of rows in a window, all rows are sorted when it is not positive.
 */
template< typename Matrix >
Containers::Vector< typename Matrix::IndexType, Devices::Host, typename Matrix::IndexType >
getSortedRowOrder( const Matrix& matrix, typename Matrix::IndexType window )
{
   using Index = typename Matrix::IndexType;
   const Index rows = matrix.getRows();
   if( window <= 0 )
      window = rows;
   const auto offsets = matrix.getSegments().getOffsets().getConstView();
   auto longer = [ & ]( Index a, Index b )
   {
      return offsets[ a + 1 ] - offsets[ a ] > offsets[ b + 1 ] - offsets[ b ];
   };

   std::vector< Index > order( rows );
   std::iota( order.begin(), order.end(), Index( 0 ) );
   for( Index first = 0; first < rows; first += window )
      std::stable_sort( order.begin() + first, order.begin() + std::min( first + window, rows ), longer );

   Containers::Vector< Index, Devices::Host, Index > result( rows );
   for( Index i = 0; i < rows; i++ )
      result[ i ] = order[ i ];
   return result;
}

/**
 * \brief Returns the numbers of elements of the Ellpack and SlicedEllpack
 * formats of the matrix including the padding elements.
 *
 * \param alignment is the alignment of the number of rows of Ellpack.
 * \param sliceSize is the number of rows of a slice of SlicedEllpack.
 */
template< typename Matrix >
std::pair< std::size_t, std::size_t >
getPaddedSizes( const Matrix& matrix, typename Matrix::IndexType alignment, typename Matrix::IndexType sliceSize )
{
   using Index = typename Matrix::IndexType;
   const Index rows = matrix.getRows();
   const auto offsets = matrix.getSegments().getOffsets().getConstView();
   std::size_t maxRowLength = 0;
   std::size_t slicedEllpackSize = 0;
   for( Index first = 0; first < rows; first += sliceSize ) {
      std::size_t sliceMax = 0;
      for( Index row = first; row < std::min( first + sliceSize, rows ); row++ )
         sliceMax = std::max( sliceMax, std::size_t( offsets[ row + 1 ] - offsets[ row ] ) );
      maxRowLength = std::max( maxRowLength, sliceMax );
      slicedEllpackSize += sliceMax * sliceSize;
   }
   const std::size_t alignedRows = ( rows + alignment - 1 ) / alignment * alignment;
   return { alignedRows * maxRowLength, slicedEllpackSize };
}

#if defined( __CUDACC__ ) || defined( __HIP__ )

template< typename Device, typename Index, typename IndexAllocator >
using EllpackSegments = Algorithms::Segments::Ellpack< Device, Index, IndexAllocator >;

template< typename Device, typename Index, typename IndexAllocator >
using SlicedEllpackSegments = Algorithms::Segments::SlicedEllpack< Device, Index, IndexAllocator >;

/**
 * \brief The sparse matrix in the other formats than CSR benchmarked by the TNL SpMM.
 *
 * A format is empty when it is not selected by the `segments-formats` parameter or
 * when its padding would be too large.
 */
template< typename Real, typename Index >
struct SegmentsFormats
{
   template< template< typename, typename, typename > class Segments >
   using Matrix = Matrices::SparseMatrix< Real, Devices::GPU, Index, Matrices::GeneralMatrix, Segments >;

   std::optional< Matrix< Algorithms::Segments::AdaptiveCSR > > adaptiveCSR;
   std::optional< Matrix< EllpackSegments > > ellpack;
   std::optional< Matrix< SlicedEllpackSegments > > slicedEllpack;
};

/**
 * \brief Device data of the row bins for the Binned kernel.
 *
 * The number of threads per row of a bin is the smallest power of two (at
 * most 32) for which every thread processes at most `elementsPerThread`
 * elements of the row. The rows longer than `pieceSize` are split into pieces.
 */
template< typename Index >
struct RowBinsData
{
   using DeviceVector = Containers::Vector< Index, Devices::GPU, Index >;

   template< typename Matrix >
   RowBinsData( const Matrix& matrix, Index elementsPerThread, Index pieceSize )
   {
      const Index rows = matrix.getRows();
      const auto offsets = matrix.getSegments().getOffsets().getConstView();
      const auto order = getSortedRowOrder( matrix, Index( 0 ) );
      auto length = [ & ]( Index slot )
      {
         return offsets[ order[ slot ] + 1 ] - offsets[ order[ slot ] ];
      };

      std::vector< Index > pieceRows;
      std::vector< Index > pieceBegins;
      std::vector< Index > pieceEnds;
      Index slot = 0;
      for( ; slot < rows && length( slot ) > pieceSize; slot++ ) {
         const Index row = order[ slot ];
         for( Index begin = offsets[ row ]; begin < offsets[ row + 1 ]; begin += pieceSize ) {
            pieceRows.push_back( row );
            pieceBegins.push_back( begin );
            pieceEnds.push_back( std::min( begin + pieceSize, offsets[ row + 1 ] ) );
         }
      }
      bins.longRows = slot;
      bins.pieces = pieceRows.size();

      // the bin b uses 32 >> b threads per row, the rows are sorted in descending order
      for( int b = 0; b < RowBins< Index >::bins; b++ ) {
         bins.binBegins[ b ] = slot;
         const Index tpr = 32 >> b;
         const Index smallerTpr = tpr / 2;
         while( slot < rows && ( smallerTpr == 0 || length( slot ) > smallerTpr * elementsPerThread ) )
            slot++;
      }
      bins.binBegins[ RowBins< Index >::bins ] = rows;

      rowOrder = order;
      auto upload = [ & ]( DeviceVector& target, const std::vector< Index >& source )
      {
         Containers::Vector< Index, Devices::Host, Index > hostVector( source.size() );
         std::copy( source.begin(), source.end(), hostVector.getData() );
         target = hostVector;
      };
      upload( this->pieceRows, pieceRows );
      upload( this->pieceBegins, pieceBegins );
      upload( this->pieceEnds, pieceEnds );
      bins.rowOrder = rowOrder.getData();
      bins.pieceRows = this->pieceRows.getData();
      bins.pieceBegins = this->pieceBegins.getData();
      bins.pieceEnds = this->pieceEnds.getData();
   }

   DeviceVector rowOrder;
   DeviceVector pieceRows;
   DeviceVector pieceBegins;
   DeviceVector pieceEnds;
   RowBins< Index > bins;
};

#endif

template< typename Real, typename Index >
void
benchmarkHost(
   Benchmark& benchmark,
   const Matrices::SparseMatrix< Real, Devices::Host, Index >& matrix,
   const SpMMProblem< Real, Index >& problem )
{
   using HostVector = Containers::Vector< Real, Devices::Host, Index >;

   const HostVector B = problem.makeB();
   HostVector C( problem.rows * problem.k );
   const auto A = getCSRArrays( matrix );
   const auto arrayB = getDenseArray< Real >( B, problem.columns, problem.k, problem.leadingDimensionB() );
   auto arrayC = getDenseArray< Real >( C, problem.rows, problem.k, problem.leadingDimensionC() );
   const double flops = 2.0 * problem.nonzeros * problem.k;

   C.setValue( std::numeric_limits< Real >::quiet_NaN() );
   benchmark.setMetadataElement( { "kernel config", "" } );
   benchmark.setMetadataElement( { "row order", "natural" } );
   SpMMResult result(
      flops,
      [ & ]()
      {
         return problem.relativeError( C );
      } );
   auto compute = [ & ]()
   {
      spmmHost( A, arrayB, arrayC, problem.rowMajor );
   };
   benchmark.time< Devices::Host >( "RowWise", compute, result );
}

#if defined( __CUDACC__ ) || defined( __HIP__ )

template< typename Real, typename Index >
void
benchmarkGPU(
   Benchmark& benchmark,
   const Config::ParameterContainer& parameters,
   const Matrices::SparseMatrix< Real, Devices::GPU, Index >& matrix,
   const Containers::Vector< Index, Devices::GPU, Index >& sortedRowOrder,
   const std::vector< std::pair< std::string, RowBinsData< Index > > >& rowBins,
   const SegmentsFormats< Real, Index >& formats,
   const SpMMProblem< Real, Index >& problem )
{
   using GPUVector = Containers::Vector< Real, Devices::GPU, Index >;

   const auto kernels = parameters.getList< std::string >( "kernels" );
   const auto segmentsFormats = parameters.getList< std::string >( "segments-formats" );
   const auto& rowOrderParameter = parameters.getParameter< std::string >( "row-order" );

   // B is not modified, it is not const only for the DenseMatrixView used by the TNL SpMM
   GPUVector B( problem.makeB() );
   GPUVector C( problem.rows * problem.k );
   const auto A = getCSRArrays( matrix );
   const auto arrayB = getDenseArray< Real >( B, problem.columns, problem.k, problem.leadingDimensionB() );
   auto arrayC = getDenseArray< Real >( C, problem.rows, problem.k, problem.leadingDimensionC() );
   const double flops = 2.0 * problem.nonzeros * problem.k;

   // CSR arrays with the row orders used by the RowGroup and LanesOverColumns kernels
   std::vector< std::pair< std::string, CSRArrays< Real, Index > > > orderedA;
   if( rowOrderParameter == "natural" || rowOrderParameter == "all" )
      orderedA.emplace_back( "natural", A );
   if( rowOrderParameter == "sorted" || rowOrderParameter == "all" ) {
      orderedA.emplace_back( "sorted", A );
      orderedA.back().second.rowOrder = sortedRowOrder.getData();
   }

   std::string rowOrder = "natural";
   auto run = [ & ]( const std::string& performer, const std::string& config, auto&& compute )
   {
      C.setValue( std::numeric_limits< Real >::quiet_NaN() );
      benchmark.setMetadataElement( { "kernel config", config } );
      benchmark.setMetadataElement( { "row order", rowOrder } );
      SpMMResult result(
         flops,
         [ & ]()
         {
            return problem.relativeError( C );
         } );
      benchmark.time< Devices::GPU >( performer, compute, result );
   };

   // k independent SpMV operations of TNL (column-major only, the columns are contiguous vectors)
   if( ! problem.rowMajor && isSelected( kernels, "k-spmv" ) ) {
      using Algorithms::Segments::LaunchConfiguration;
      using Algorithms::Segments::ThreadsToSegmentsMapping;
      auto runKSpMV = [ & ]( const LaunchConfiguration& launchConfig, const std::string& config )
      {
         auto compute = [ & ]()
         {
            for( Index j = 0; j < problem.k; j++ ) {
               auto inVector = B.getConstView( j * problem.columns, ( j + 1 ) * problem.columns );
               auto outVector = C.getView( j * problem.rows, ( j + 1 ) * problem.rows );
               matrix.vectorProduct( inVector, outVector, launchConfig );
            }
         };
         run( "TNL k-SpMV", config, compute );
      };
      runKSpMV( LaunchConfiguration{}, "Default" );
      for( int TPS : { 1, 2, 4, 8, 16, 32 } )
         runKSpMV( LaunchConfiguration( ThreadsToSegmentsMapping::Uniform, TPS ), "Uniform TPS=" + std::to_string( TPS ) );
   }

   // the generic SpMM of TNL based on the multireduction in the segments of the sparse matrix
   if( isSelected( kernels, "segments" ) ) {
      using Algorithms::Segments::LaunchConfiguration;
      // runs the product with the sparse matrix in the given format, tiles of at most MaxTileSize columns
      // and all launch configurations of its segments, or only the Default one
      auto runSegments = [ & ]( const std::string& performer, const auto& sparseMatrix, auto maxTileSize, bool defaultOnly )
      {
         constexpr int MaxTileSize = decltype( maxTileSize )::value;
         auto launchConfigs = Algorithms::Segments::reductionLaunchConfigurations( sparseMatrix.getSegments() );
         if( defaultOnly )
            launchConfigs.resize( 1 );
         for( const auto& entry : launchConfigs ) {
            const LaunchConfiguration& launchConfig = entry.first;
            auto compute = [ & ]()
            {
               auto multiply = [ & ]( auto organization )
               {
                  constexpr auto Organization = decltype( organization )::value;
                  using DenseView = Matrices::DenseMatrixView< Real, Devices::GPU, Index, Organization >;
                  const DenseView viewB( problem.columns, problem.k, B.getView() );
                  DenseView viewC( problem.rows, problem.k, C.getView() );
                  Matrices::sparseDenseMatrixProduct< MaxTileSize >(
                     sparseMatrix, viewB, viewC, Real( 1 ), Real( 0 ), launchConfig );
               };
               if( problem.rowMajor )
                  multiply(
                     std::integral_constant< Algorithms::Segments::ElementsOrganization, Algorithms::Segments::RowMajorOrder >{} );
               else
                  multiply(
                     std::integral_constant<
                        Algorithms::Segments::ElementsOrganization,
                        Algorithms::Segments::ColumnMajorOrder >{} );
            };
            run( performer, entry.second, compute );
         }
      };
      using DefaultTileSize = std::integral_constant< int, 16 >;
      if( isSelected( segmentsFormats, "csr" ) )
         runSegments( "TNL SpMM", matrix, DefaultTileSize{}, false );
      // The adaptive kernel supports values of at most 32 bytes, the larger tiles would fall back to the CSR kernels.
      // The other launch configurations of AdaptiveCSR are the same as those of CSR.
      if( formats.adaptiveCSR )
         runSegments(
            "TNL SpMM AdaptiveCSR", *formats.adaptiveCSR, std::integral_constant< int, 32 / sizeof( Real ) >{}, true );
      if( formats.ellpack )
         runSegments( "TNL SpMM Ellpack", *formats.ellpack, DefaultTileSize{}, false );
      if( formats.slicedEllpack )
         runSegments( "TNL SpMM SlicedEllpack", *formats.slicedEllpack, DefaultTileSize{}, false );
   }

   for( const auto& entry : orderedA ) {
      rowOrder = entry.first;
      const auto& orderA = entry.second;

      // RowGroup kernels
      if( isSelected( kernels, "row-group" ) )
         for( int K : rowGroupTilesFor( problem.k ) )
            for( int TPR : { 1, 2, 4, 8, 16, 32 } )
               dispatchTile(
                  K,
                  [ & ]( auto K_ )
                  {
                     dispatchPowerOfTwo(
                        TPR,
                        [ & ]( auto TPR_ )
                        {
                           auto compute = [ & ]()
                           {
                              if( problem.rowMajor )
                                 spmmRowGroup< TPR_, K_, true >( orderA, arrayB, arrayC );
                              else
                                 spmmRowGroup< TPR_, K_, false >( orderA, arrayB, arrayC );
                           };
                           run( "RowGroup", "TPR=" + std::to_string( TPR ) + " K=" + std::to_string( K ), compute );
                        } );
                  } );

      // RowGroup kernels with one tile of columns per launch (column-major only), the columns of B used by
      // one launch fit better in the cache
      if( ! problem.rowMajor && isSelected( kernels, "row-group" ) )
         for( int K : { 1, 2, 4, 8 } ) {
            if( K >= problem.k )
               continue;
            for( int TPR : { 1, 2, 4, 8, 16, 32 } )
               dispatchTile(
                  K,
                  [ & ]( auto K_ )
                  {
                     dispatchPowerOfTwo(
                        TPR,
                        [ & ]( auto TPR_ )
                        {
                           auto compute = [ & ]()
                           {
                              spmmRowGroup< TPR_, K_, false >( orderA, arrayB, arrayC, Real( 1 ), Real( 0 ), Index( 1 ) );
                           };
                           run( "RowGroup", "TPR=" + std::to_string( TPR ) + " K=" + std::to_string( K ) + " T=1", compute );
                        } );
                  } );
         }

      // LanesOverColumns kernels (row-major only)
      if( problem.rowMajor && isSelected( kernels, "lanes-over-columns" ) ) {
         for( int G : { 4, 8, 16, 32 } )
            for( int V : { 1, 2, 4 } ) {
               if( G * V >= 2 * problem.k )
                  continue;
               dispatchPowerOfTwo(
                  G,
                  [ & ]( auto G_ )
                  {
                     dispatchPowerOfTwo(
                        V,
                        [ & ]( auto V_ )
                        {
                           if constexpr( G_ >= 4 && V_ <= 4 ) {
                              auto compute = [ & ]()
                              {
                                 spmmLanesOverColumns< G_, V_ >( orderA, arrayB, arrayC );
                              };
                              run( "LanesOverColumns", "G=" + std::to_string( G ) + " V=" + std::to_string( V ), compute );
                           }
                        } );
                  } );
            }
      }
   }
   rowOrder = "natural";

   // Binned kernels
   if( isSelected( kernels, "binned" ) ) {
      rowOrder = "sorted";
      for( int K : rowGroupTilesFor( problem.k ) )
         for( const auto& entry : rowBins ) {
            const auto& bins = entry.second.bins;
            dispatchTile(
               K,
               [ & ]( auto K_ )
               {
                  auto compute = [ & ]()
                  {
                     if( problem.rowMajor )
                        spmmBinned< K_, true >( A, bins, arrayB, arrayC );
                     else
                        spmmBinned< K_, false >( A, bins, arrayB, arrayC );
                  };
                  run( "Binned", "K=" + std::to_string( K ) + " " + entry.first, compute );
               } );
         }
      rowOrder = "natural";
   }

   // NnzSplit kernels (row-major only), the chunk size is a multiple of G
   if( problem.rowMajor && isSelected( kernels, "nnz-split" ) ) {
      for( int G : { 4, 8, 16, 32 } )
         for( int V : { 1, 2, 4 } )
            for( int S : { 4, 16 } ) {
               if( G * V >= 2 * problem.k )
                  continue;
               dispatchPowerOfTwo(
                  G,
                  [ & ]( auto G_ )
                  {
                     dispatchPowerOfTwo(
                        V,
                        [ & ]( auto V_ )
                        {
                           if constexpr( G_ >= 4 && V_ <= 4 ) {
                              const Index chunkSize = G * S;
                              auto compute = [ & ]()
                              {
                                 spmmNnzSplit< G_, V_ >( A, arrayB, arrayC, chunkSize );
                              };
                              run(
                                 "NnzSplit",
                                 "G=" + std::to_string( G ) + " V=" + std::to_string( V ) + " L=" + std::to_string( chunkSize ),
                                 compute );
                           }
                        } );
                  } );
            }
   }

   #if defined( __CUDACC__ ) && defined( HAVE_CUSPARSE )
   const std::pair< cusparseSpMMAlg_t, std::string > cuSparseAlgorithms[] = { { CUSPARSE_SPMM_ALG_DEFAULT, "ALG_DEFAULT" },
                                                                              { CUSPARSE_SPMM_CSR_ALG1, "CSR_ALG1" },
                                                                              { CUSPARSE_SPMM_CSR_ALG2, "CSR_ALG2" },
                                                                              { CUSPARSE_SPMM_CSR_ALG3, "CSR_ALG3" } };
   for( const auto& [ algorithm, name ] : cuSparseAlgorithms ) {
      if( ! isSelected( kernels, "vendor" ) )
         break;
      try {
         CuSparseSpMM< Real, Index > cuSparse( A, arrayB, arrayC, problem.rowMajor, algorithm );
         auto compute = [ & ]()
         {
            cuSparse.multiply();
         };
         run( "cuSPARSE", name, compute );
      }
      catch( const std::exception& e ) {
         benchmark.addErrorMessage( "cuSPARSE " + name + ": " + e.what() );
      }
   }
   #endif

   #if defined( __HIP__ ) && defined( HAVE_HIPSPARSE )
   const std::pair< hipsparseSpMMAlg_t, std::string > hipSparseAlgorithms[] = { { HIPSPARSE_SPMM_ALG_DEFAULT, "ALG_DEFAULT" },
                                                                                { HIPSPARSE_SPMM_CSR_ALG1, "CSR_ALG1" },
                                                                                { HIPSPARSE_SPMM_CSR_ALG2, "CSR_ALG2" },
                                                                                { HIPSPARSE_SPMM_CSR_ALG3, "CSR_ALG3" } };
   for( const auto& [ algorithm, name ] : hipSparseAlgorithms ) {
      if( ! isSelected( kernels, "vendor" ) )
         break;
      try {
         HipSparseSpMM< Real, Index > hipSparse( A, arrayB, arrayC, problem.rowMajor, algorithm );
         auto compute = [ & ]()
         {
            hipSparse.multiply();
         };
         run( "hipSPARSE", name, compute );
      }
      catch( const std::exception& e ) {
         benchmark.addErrorMessage( "hipSPARSE " + name + ": " + e.what() );
      }
   }
   #endif
}

#endif

/**
 * \brief Runs the benchmark for one sparse matrix with all requested numbers of
 * dense columns, dense layouts and devices.
 */
template< typename Real, typename Index >
void
benchmarkMatrix(
   Benchmark& benchmark,
   const Config::ParameterContainer& parameters,
   const Matrices::SparseMatrix< Real, Devices::Host, Index >& hostMatrix,
   const std::string& matrixName )
{
   const auto& device = parameters.getParameter< std::string >( "device" );
   const auto& layout = parameters.getParameter< std::string >( "dense-layout" );
   const auto denseColumns = parameters.getList< int >( "dense-columns" );

   const Index nonzeros = hostMatrix.getNonzeroElementsCount();
   std::cout << "Matrix " << matrixName << ": " << hostMatrix.getRows() << " x " << hostMatrix.getColumns() << ", " << nonzeros
             << " nonzeros\n";

#if defined( __CUDACC__ ) || defined( __HIP__ )
   Matrices::SparseMatrix< Real, Devices::GPU, Index > gpuMatrix;
   Containers::Vector< Index, Devices::GPU, Index > sortedRowOrder;
   std::vector< std::pair< std::string, RowBinsData< Index > > > rowBins;
   SegmentsFormats< Real, Index > formats;
   if( device == "cuda" || device == "hip" || device == "all" ) {
      gpuMatrix = hostMatrix;
      sortedRowOrder = getSortedRowOrder( hostMatrix, Index( parameters.getParameter< int >( "sorting-window" ) ) );
      constexpr Index pieceSize = 1024;
      for( Index elementsPerThread : { 2, 4 } )
         rowBins.emplace_back(
            "E=" + std::to_string( elementsPerThread ) + " P=" + std::to_string( pieceSize ),
            RowBinsData< Index >( hostMatrix, elementsPerThread, pieceSize ) );

      // the other formats for the TNL SpMM, the formats with padding are skipped when it is too large
      const auto segmentsFormats = parameters.getList< std::string >( "segments-formats" );
      const double maxPadding = parameters.getParameter< double >( "max-padding-ratio" ) * nonzeros;
      const auto [ ellpackSize, slicedEllpackSize ] = getPaddedSizes( hostMatrix, Index( 32 ), Index( 32 ) );
      if( isSelected( segmentsFormats, "adaptive-csr" ) )
         formats.adaptiveCSR.emplace() = gpuMatrix;
      if( isSelected( segmentsFormats, "ellpack" ) ) {
         if( ellpackSize <= maxPadding )
            formats.ellpack.emplace() = gpuMatrix;
         else
            std::cout << "Skipping Ellpack, it would have " << ellpackSize << " elements\n";
      }
      if( isSelected( segmentsFormats, "sliced-ellpack" ) ) {
         if( slicedEllpackSize <= maxPadding )
            formats.slicedEllpack.emplace() = gpuMatrix;
         else
            std::cout << "Skipping SlicedEllpack, it would have " << slicedEllpackSize << " elements\n";
      }
   }
#endif

   for( const int k : denseColumns ) {
      // the reference result is computed in row-major order
      SpMMProblem< Real, Index > problem;
      problem.rows = hostMatrix.getRows();
      problem.columns = hostMatrix.getColumns();
      problem.nonzeros = nonzeros;
      problem.k = k;
      {
         problem.rowMajor = true;
         const auto B = problem.makeB();
         problem.reference.setSize( problem.rows * k );
         auto arrayC = getDenseArray< Real >( problem.reference, problem.rows, problem.k, problem.leadingDimensionC() );
         spmmHost(
            getCSRArrays( hostMatrix ),
            getDenseArray< Real >( B, problem.columns, k, problem.leadingDimensionB() ),
            arrayC,
            true );
      }

      for( const bool rowMajor : { true, false } ) {
         if( ! ( layout == "all" || ( layout == "row-major" ) == rowMajor ) )
            continue;
         problem.rowMajor = rowMajor;

         benchmark.setMetadataColumns(
            Benchmark::MetadataColumns(
               {
                  { "matrix", matrixName },
                  { "rows", std::to_string( problem.rows ) },
                  { "columns", std::to_string( problem.columns ) },
                  { "nonzeros", std::to_string( nonzeros ) },
                  { "precision", getType< Real >() },
                  { "dense columns", std::to_string( k ) },
                  { "dense layout", rowMajor ? "row-major" : "column-major" },
               } ) );
         benchmark.setOperation( "SpMM" );
         // minimal memory traffic: A once, B once, C once
         benchmark.setDatasetSize(
            nonzeros * ( sizeof( Real ) + sizeof( Index ) ) + ( problem.rows + 1 ) * sizeof( Index )
            + ( problem.columns + problem.rows ) * std::size_t( k ) * sizeof( Real ) );
         benchmark.setOperationsPerLoop( 2 * std::size_t( nonzeros ) * k );

         if( device == "host" || device == "all" )
            benchmarkHost( benchmark, hostMatrix, problem );

#if defined( __CUDACC__ ) || defined( __HIP__ )
         if( device == "cuda" || device == "hip" || device == "all" )
            benchmarkGPU( benchmark, parameters, gpuMatrix, sortedRowOrder, rowBins, formats, problem );
#endif
      }
   }
}

template< typename Real, typename Index >
void
runBenchmark( Benchmark& benchmark, const Config::ParameterContainer& parameters )
{
   using HostMatrix = Matrices::SparseMatrix< Real, Devices::Host, Index >;

   const auto& inputFile = parameters.getParameter< std::string >( "input-file" );
   if( ! inputFile.empty() ) {
      HostMatrix matrix;
      Matrices::MatrixReader< HostMatrix >::readMtx( inputFile, matrix );
      benchmarkMatrix( benchmark, parameters, matrix, inputFile );
      return;
   }

   const auto& generator = parameters.getParameter< std::string >( "matrix-generator" );
   const auto gridSize2D = parameters.getParameter< int >( "grid-size-2d" );
   const auto gridSize3D = parameters.getParameter< int >( "grid-size-3d" );
   for( const std::string stencil : { "laplace-2d", "laplace-3d", "stencil-27" } ) {
      if( ! ( generator == "all" || generator == stencil ) )
         continue;
      const int gridSize = stencil == "laplace-2d" ? gridSize2D : gridSize3D;
      HostMatrix matrix;
      generateStencilMatrix( matrix, stencil, gridSize );
      benchmarkMatrix( benchmark, parameters, matrix, stencil + "-" + std::to_string( gridSize ) );
   }
}

}  // namespace TNL::Benchmarks::SpMM
