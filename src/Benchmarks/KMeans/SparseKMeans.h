// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Algorithms/scan.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Backend/KernelLaunch.h>
#include <TNL/Backend/LaunchHelpers.h>
#include <TNL/Backend/SharedMemory.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Functional.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Matrices/SparseDenseOperations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>
#include <TNL/Timer.h>

#if defined( __CUDACC__ )
   #include <cuda_fp16.h>
#elif defined( __HIP__ )
   #include <hip/hip_fp16.h>
#endif

namespace TNL::Benchmarks::KMeans {

/**
 * \brief Packs a distance and the index of a cluster into a key whose unsigned ordering is the ordering of the distances
 * (ties are broken by the smaller index), so that the nearest cluster is found by an atomic minimum of 64-bit integers.
 *
 * The distance is rounded to single precision.
 */
__cuda_callable__
inline std::uint64_t
packDistance( float distance, std::uint32_t cluster )
{
   std::uint32_t bits;
   std::memcpy( &bits, &distance, sizeof( bits ) );
   // the order of the negative floats is reversed, flipping all bits of them and the sign bit of the positive ones gives
   // the order of unsigned integers
   bits = ( bits & 0x80000000u ) != 0 ? ~bits : ( bits | 0x80000000u );
   return ( std::uint64_t( bits ) << 32 ) | cluster;
}

//! \brief Returns the distance packed by \ref packDistance.
__cuda_callable__
inline float
unpackDistance( std::uint64_t key )
{
   std::uint32_t bits = key >> 32;
   bits = ( bits & 0x80000000u ) != 0 ? ( bits & 0x7fffffffu ) : ~bits;
   float distance;
   std::memcpy( &distance, &bits, sizeof( distance ) );
   return distance;
}

//! \brief Returns the cluster index packed by \ref packDistance.
__cuda_callable__
inline std::uint32_t
unpackCluster( std::uint64_t key )
{
   return key & 0xffffffffu;
}

/**
 * \brief Atomically replaces `target` by `key` when `key` is smaller.
 *
 * On the host, all columns of a row of the sparse-dense product are stored by the same thread (or by the same thread
 * in successive passes over the tiles of the columns), so no atomic operation is needed.
 */
__cuda_callable__
inline void
atomicMinKey( std::uint64_t& target, std::uint64_t key )
{
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
   // the minimum decreases quickly, so most atomic operations are avoided by checking the current value first (a stale
   // value only causes a needless atomic operation)
   if( key < *static_cast< volatile std::uint64_t* >( &target ) )
      atomicMin( reinterpret_cast< unsigned long long* >( &target ), static_cast< unsigned long long >( key ) );
#else
   if( key < target )
      target = key;
#endif
}

//! \brief Method of the assignment of the points to the nearest centroids.
enum class Assignment : std::uint8_t
{
   //! The distances are computed by \ref TNL::Matrices::sparseDenseMatrixProductWithLambdas and the nearest centroid is
   //! found by the storer of the product, the matrix of the distances is not stored. The centroids are read by
   //! \ref TNL::Matrices::DirectFetch, i.e. by vectorized loads.
   Fused,
   //! The same as \ref Fused with the centroids read by a lambda function, i.e. element by element.
   FusedLambda,
   //! The same as \ref Fused with the centroids read from a copy of `C^T` in half precision, which halves the data read
   //! from the cache (only on GPUs). The distances are rounded, so a point with two nearly equally distant centroids may
   //! be assigned to the other one.
   FusedHalf,
   //! The dot products are stored in a dense matrix by \ref TNL::Matrices::sparseDenseMatrixProduct and the nearest
   //! centroid is found in its rows.
   Unfused
};

//! \brief Method of the update of the centroids.
enum class Update : std::uint8_t
{
   //! The sums of the points of the clusters are the product of the transposed data matrix and the indicator matrix of the
   //! clusters given by a lambda function, computed by \ref TNL::Matrices::sparseDenseMatrixProductWithLambdas.
   SpMM,
   //! The elements of the points are added to the sums of their clusters by atomic operations.
   Atomic,
   //! The sums are computed by the rows of the transposed data matrix, i.e. separately for every feature, so that the sums
   //! of one feature are accumulated in fast memory (in shared memory on GPUs) and written once. On GPUs, the rows are
   //! split into pieces processed by groups of 32 threads, the pieces of the long rows are added by atomic operations.
   Histogram
};

//! \brief Configuration of the fused assignment, see \ref SparseKMeans::tuneAssignment.
struct AssignmentConfiguration
{
   //! The number of clusters processed at once, 0 means all.
   int tileSize = 0;
   //! The number of lanes of the product on GPUs, 0 means the default configuration of the product.
   int lanes = 0;
   //! The number of columns per lane of the product on GPUs.
   int columnsPerLane = 0;

   [[nodiscard]] std::string
   toString() const
   {
      const std::string tile = "T=" + ( tileSize > 0 ? std::to_string( tileSize ) : std::string( "K" ) );
      if( lanes == 0 )
         return tile + " default";
      return tile + " L=" + std::to_string( lanes ) + " V=" + std::to_string( columnsPerLane );
   }
};

#if defined( __CUDACC__ ) || defined( __HIP__ )
namespace detail {

/**
 * \brief Computes the sums of the clusters of the pieces of the rows of `X^T`, see \ref Update::Histogram.
 *
 * Every piece is processed by a group of 32 threads with its own `K * copies` sums in shared memory, the threads use
 * `copies` copies of the sums to reduce the conflicts of the atomic operations. The means of the short rows (one piece)
 * are written to `CT`, the sums of the long rows are added to `longSums`.
 */
template< typename Real, typename Index, typename Label >
__global__
void
histogramUpdateKernel(
   const Index* offsets,
   const Index* points,
   const Real* values,
   const Index* pieceRows,
   const Index* pieceBegins,
   const Index* pieceLongRows,
   Index pieces,
   Index pieceSize,
   const Label* labels,
   const Index* counts,
   Index K,
   int copies,
   Real* CT,
   Real* longSums )
{
   constexpr int groupSize = 32;
   const auto group = cg::tiled_partition< groupSize >( cg::this_thread_block() );
   const Index groupsPerBlock = blockDim.x / groupSize;
   const int lane = group.thread_rank();
   const int copy = lane % copies;
   Real* sums = Backend::getSharedMemory< Real >() + threadIdx.x / groupSize * K * copies;
   for( Index piece = blockIdx.x * groupsPerBlock + threadIdx.x / groupSize; piece < pieces;
        piece += gridDim.x * groupsPerBlock )
   {
      for( Index i = lane; i < K * copies; i += groupSize )
         sums[ i ] = 0;
      group.sync();
      const Index feature = pieceRows[ piece ];
      const Index begin = pieceBegins[ piece ];
      const Index end = min( begin + pieceSize, offsets[ feature + 1 ] );
      for( Index i = begin + lane; i < end; i += groupSize )
         atomicAdd( &sums[ Index( labels[ points[ i ] ] ) * copies + copy ], values[ i ] );
      group.sync();
      const Index longRow = pieceLongRows[ piece ];
      for( Index cluster = lane; cluster < K; cluster += groupSize ) {
         Real sum = 0;
         for( int c = 0; c < copies; c++ )
            sum += sums[ cluster * copies + c ];
         if( longRow < 0 ) {
            const Index count = counts[ cluster ];
            if( count > 0 )
               CT[ feature * K + cluster ] = sum / count;
         }
         else
            atomicAdd( &longSums[ longRow * K + cluster ], sum );
      }
      group.sync();
   }
}

}  // namespace detail
#endif

/**
 * \brief Lloyd's algorithm of the k-means clustering of sparse data.
 *
 * The points are the rows of the sparse matrix `X` with `n` rows and `d` columns. One iteration assigns every point to
 * the nearest centroid and replaces every centroid by the mean of its points, the centroid of an empty cluster is kept.
 * The squared distance is computed as `|x|^2 - 2 x.c + |c|^2`, the dot products of all points and centroids are the
 * sparse-dense product `X * C^T`.
 *
 * The centroids are stored twice: `C^T` (`d x K`, row-major) is the dense matrix of the product, so that the centroids
 * of a feature are consecutive, and `C` (`K x d`, row-major) is used for the norms of the centroids on the host.
 *
 * The methods with lambda functions are public, because nvcc does not allow extended lambdas in private methods.
 */
template< typename Real, typename Device, typename Index >
class SparseKMeans
{
public:
   using MatrixType = Matrices::SparseMatrix< Real, Device, Index >;
   using VectorType = Containers::Vector< Real, Device, Index >;
   using IndexVectorType = Containers::Vector< Index, Device, Index >;
   using KeysType = Containers::Array< std::uint64_t, Device, Index >;
   using DenseMatrixType = Matrices::DenseMatrix< Real, Device, Index, Algorithms::Segments::RowMajorOrder >;

   SparseKMeans( const MatrixType& X, Index clusters )
   : X( X ),
     clusters( clusters )
   {
      if( clusters <= 0 || clusters > X.getRows() )
         throw std::invalid_argument( "SparseKMeans: wrong number of clusters" );
      const Index n = X.getRows();
      const Index d = X.getColumns();
      XT.getTransposition( X );
      squaredNorms.setSize( n );
      keys.setSize( n );
      labels.setSize( n );
      distances.setSize( n );
      counts.setSize( clusters );
      centroidNorms.setSize( clusters );
      centroidsT.setDimensions( d, clusters );
      centroids.setDimensions( clusters, d );
      computeSquaredNorms();
      if constexpr( isGPU() ) {
         setOnes();
         setPieces();
         if( clusters <= std::numeric_limits< std::uint16_t >::max() + 1 )
            shortLabels.setSize( n );
      }
   }

   //! \brief Returns whether the computation runs on a GPU.
   [[nodiscard]] static constexpr bool
   isGPU()
   {
      return ! std::is_same_v< Device, Devices::Host > && ! std::is_same_v< Device, Devices::Sequential >;
   }

   /**
    * \brief Splits the rows of `X^T` into the pieces of at most \ref pieceSize elements for \ref Update::Histogram on GPUs.
    *
    * `pieceLongRows` is the index of the row in `longRows` for the pieces of the rows with more than one piece, and -1
    * otherwise.
    */
   void
   setPieces()
   {
      Containers::Vector< Index, Devices::Host, Index > offsets;
      offsets = XT.getSegments().getOffsets();
      std::vector< Index > rows;
      std::vector< Index > begins;
      std::vector< Index > rowsOfPieces;
      std::vector< Index > longs;
      for( Index feature = 0; feature < XT.getRows(); feature++ ) {
         const bool isLong = offsets[ feature + 1 ] - offsets[ feature ] > pieceSize;
         for( Index begin = offsets[ feature ]; begin < offsets[ feature + 1 ]; begin += pieceSize ) {
            rows.push_back( feature );
            begins.push_back( begin );
            rowsOfPieces.push_back( isLong ? Index( longs.size() ) : Index( -1 ) );
         }
         if( isLong )
            longs.push_back( feature );
      }
      pieceRows = Containers::Vector< Index, Devices::Host, Index >( rows );
      pieceBegins = Containers::Vector< Index, Devices::Host, Index >( begins );
      pieceLongRows = Containers::Vector< Index, Devices::Host, Index >( rowsOfPieces );
      longRows = Containers::Vector< Index, Devices::Host, Index >( longs );
      longSums.setSize( longRows.getSize() * clusters );
   }

   //! \brief Sets the sparse matrix with one row of `d` ones used for the norms of the centroids on GPUs.
   void
   setOnes()
   {
      const Index d = X.getColumns();
      ones.setDimensions( 1, d );
      ones.setRowCapacities( IndexVectorType( 1, d ) );
      auto onesView = ones.getView();
      Algorithms::parallelFor< Device >(
         0,
         d,
         [ = ] __cuda_callable__( Index feature ) mutable
         {
            onesView.getRow( 0 ).setElement( feature, feature, Real( 1 ) );
         } );
   }

   //! \brief Computes the squared norms of the points.
   void
   computeSquaredNorms()
   {
      auto normsView = squaredNorms.getView();
      Matrices::reduceAllRows(
         X,
         [] __cuda_callable__( Index, Index, const Real& value ) -> Real
         {
            return value * value;
         },
         TNL::Plus{},
         [ = ] __cuda_callable__( Index row, const Real& value ) mutable
         {
            normsView[ row ] = value;
         },
         Real( 0 ) );
   }

   /**
    * \brief Sets the centroids.
    *
    * \param initial The dense matrix with `K` rows and `d` columns, every row is one centroid.
    */
   void
   setCentroids( const DenseMatrixType& initial )
   {
      if( initial.getRows() != clusters || initial.getColumns() != X.getColumns() )
         throw std::invalid_argument( "SparseKMeans: wrong dimensions of the centroids" );
      centroids = initial;
      centroidsValid = true;
      iteration = 0;
      reordered = false;
      reorderingPaysOff = tunedReorderingPaysOff.value_or( false );
      const Index K = clusters;
      const Index d = X.getColumns();
      const auto C = centroids.getConstView();
      auto CT = centroidsT.getView();
      Algorithms::parallelFor< Device >(
         0,
         d * K,
         [ = ] __cuda_callable__( Index idx ) mutable
         {
            const Index feature = idx / K;
            const Index cluster = idx % K;
            CT( feature, cluster ) = C( cluster, feature );
         } );
   }

   /**
    * \brief Computes the squared norms of the centroids.
    *
    * On the host, the rows of `C` are reduced. On GPUs, this would use only `K` threads for the rows of `d` elements, so
    * the norms are computed as the product of the row of ones and the squares of `C^T`, the very long row is split
    * among many threads by the sparse-dense product.
    */
   void
   computeCentroidNorms()
   {
      auto normsView = centroidNorms.getView();
      if constexpr( ! std::is_same_v< Device, Devices::Host > ) {
         const Index K = clusters;
         const auto CT = centroidsT.getConstView().getValues().getData();
         auto square = [] __cuda_callable__( const Real& value, Index, Index ) -> Real
         {
            return value * value;
         };
         const auto fetchB = Matrices::directFetch< Algorithms::Segments::RowMajorOrder >( CT, K, square );
         auto storeC = [ = ] __cuda_callable__( Index, int cluster, const Real& value ) mutable
         {
            normsView[ cluster ] = value;
         };
         Matrices::sparseDenseMatrixProductWithLambdas< Algorithms::Segments::RowMajorOrder >( ones, K, fetchB, storeC );
         return;
      }
      Matrices::reduceAllRows(
         centroids,
         [] __cuda_callable__( Index, Index, const Real& value ) -> Real
         {
            return value * value;
         },
         TNL::Plus{},
         [ = ] __cuda_callable__( Index row, const Real& value ) mutable
         {
            normsView[ row ] = value;
         },
         Real( 0 ) );
   }

#if defined( __CUDACC__ ) || defined( __HIP__ )
   /**
    * \brief Copies `C^T` to \ref halfCentroidsT and computes the squared norms of the rounded centroids, see
    * \ref Assignment::FusedHalf.
    */
   void
   setHalfCentroids()
   {
      const Index K = clusters;
      halfCentroidsT.setSize( centroidsT.getValues().getSize() );
      const auto CT = centroidsT.getConstView().getValues();
      auto halfCT = halfCentroidsT.getView();
      Algorithms::parallelFor< Device >(
         0,
         halfCT.getSize(),
         [ = ] __cuda_callable__( Index i ) mutable
         {
            halfCT[ i ] = __float2half( CT[ i ] );
         } );
      auto normsView = centroidNorms.getView();
      auto square = [] __cuda_callable__( const __half& value, Index, Index ) -> Real
      {
         const Real rounded = __half2float( value );
         return rounded * rounded;
      };
      auto storeC = [ = ] __cuda_callable__( Index, int cluster, const Real& value ) mutable
      {
         normsView[ cluster ] = value;
      };
      Matrices::sparseDenseMatrixProductWithLambdas(
         ones, K, Matrices::directFetch< Algorithms::Segments::RowMajorOrder >( halfCentroidsT.getData(), K, square ), storeC );
   }
#endif

   /**
    * \brief Assigns every point to the nearest centroid, see \ref Assignment::Fused.
    *
    * \param method \ref Assignment::Fused, \ref Assignment::FusedLambda or \ref Assignment::FusedHalf.
    */
   void
   assignFused( Assignment method = Assignment::Fused )
   {
      if( method == Assignment::FusedHalf ) {
#if defined( __CUDACC__ ) || defined( __HIP__ )
         if constexpr( isGPU() )
            setHalfCentroids();
         else
#endif
            throw std::invalid_argument( "SparseKMeans: the assignment in half precision runs only on GPUs" );
      }
      else
         computeCentroidNorms();
      keys.setValue( std::numeric_limits< std::uint64_t >::max() );
      const Index K = clusters;
      const auto norms = centroidNorms.getConstView();
      auto keysView = keys.getView();
      const MatrixType& A = getAssignedPoints();
      // the clusters are processed in tiles, see tuneAssignment
      const Index configuredTileSize = getAssignmentConfiguration( reordered ).tileSize;
      const Index tileSize = configuredTileSize > 0 ? configuredTileSize : K;
      for( Index first = 0; first < K; first += tileSize ) {
         const auto CT = centroidsT.getConstView().getValues().getData() + first;
         auto fetchB = [ = ] __cuda_callable__( Index feature, Index cluster ) -> Real
         {
            return CT[ feature * K + cluster ];
         };
         auto storeC = [ = ] __cuda_callable__( Index row, int cluster, const Real& dot ) mutable
         {
            atomicMinKey( keysView[ row ], packDistance( norms[ first + cluster ] - 2 * dot, first + cluster ) );
         };
         const Index columns = TNL::min( tileSize, K - first );
         if( method == Assignment::Fused )
            multiply( A, columns, Matrices::directFetch< Algorithms::Segments::RowMajorOrder >( CT, K ), storeC );
         else if( method == Assignment::FusedLambda )
            Matrices::sparseDenseMatrixProductWithLambdas< Algorithms::Segments::RowMajorOrder >( A, columns, fetchB, storeC );
#if defined( __CUDACC__ ) || defined( __HIP__ )
         else if constexpr( isGPU() ) {
            auto toReal = [] __cuda_callable__( const __half& value, Index, Index ) -> Real
            {
               return __half2float( value );
            };
            multiply(
               A,
               columns,
               Matrices::directFetch< Algorithms::Segments::RowMajorOrder >( halfCentroidsT.getData() + first, K, toReal ),
               storeC );
         }
#endif
      }
      finishAssignment();
   }

   //! \brief The numbers of lanes and columns per lane of the product of the fused assignment selected by
   //! \ref tuneAssignment on GPUs.
   static constexpr std::array< std::pair< int, int >, 10 > laneConfigurations{
      { { 8, 1 }, { 4, 2 }, { 16, 1 }, { 8, 2 }, { 32, 1 }, { 16, 2 }, { 8, 4 }, { 32, 2 }, { 16, 4 }, { 32, 4 } }
   };

   /**
    * \brief Computes the product of the fused assignment with the numbers of lanes and columns per lane of the current
    * \ref AssignmentConfiguration, by the default configuration of the product when they are zero.
    */
   template< int I = 0, typename FetchB, typename StoreC >
   void
   multiply( const MatrixType& A, Index k, const FetchB& fetchB, const StoreC& storeC ) const
   {
      if constexpr( isGPU() && I < laneConfigurations.size() ) {
         constexpr int lanes = laneConfigurations[ I ].first;
         constexpr int columnsPerLane = laneConfigurations[ I ].second;
         const AssignmentConfiguration& configuration = getAssignmentConfiguration( reordered );
         if( configuration.lanes == lanes && configuration.columnsPerLane == columnsPerLane )
            Matrices::sparseDenseMatrixProductWithLambdasLanesOverColumns< lanes, columnsPerLane >( A, k, fetchB, storeC );
         else
            multiply< I + 1 >( A, k, fetchB, storeC );
      }
      else
         Matrices::sparseDenseMatrixProductWithLambdas( A, k, fetchB, storeC );
   }

   //! \brief Sets the number of clusters processed at once by \ref assignFused, 0 means all, see \ref tuneAssignment.
   void
   setAssignmentTileSize( Index tileSize )
   {
      for( auto& configuration : assignmentConfigurations )
         configuration = AssignmentConfiguration{ static_cast< int >( tileSize ) };
      tunedReorderingPaysOff.reset();
   }

   //! \brief Returns the configuration of \ref assignFused for the points in the original order or sorted by
   //! \ref reorder.
   [[nodiscard]] const AssignmentConfiguration&
   getAssignmentConfiguration( bool sorted ) const
   {
      return assignmentConfigurations[ sorted ? 1 : 0 ];
   }

   /**
    * \brief Selects the configuration of \ref assignFused by measuring the assignment.
    *
    * The products with tiles of fewer columns of `C^T` read fewer data per element of `X`, which fit better in the cache,
    * but they read `X` more times. The best tile size depends on the data, so the tiles of `K`, 64 and 32 clusters are
    * measured with the current centroids and the fastest one is used. Then, on GPUs, the products with the numbers of
    * lanes and columns per lane from \ref laneConfigurations, which compute from a quarter of the tile to the whole tile
    * at once, are measured with the selected tile size (except \ref Assignment::FusedLambda). The best configuration
    * of the points sorted by their clusters differs, so on GPUs, it is selected in the same way after \ref reorder. Then
    * the sorting followed by the assignment is measured, which tells whether the sorting pays off for the negative
    * reordering period (see \ref setReorderingPeriod), so that it does not need to be tried in the iteration 1. The
    * result of the assignment does not depend on the configuration.
    *
    * \param method The method of the assignment, see \ref assignFused.
    */
   void
   tuneAssignment( Assignment method = Assignment::Fused, int loops = 3 )
   {
      reordered = false;
      const double unsortedTime = tuneAssignmentConfiguration( method, loops );
      if constexpr( isGPU() ) {
         reorder();
         tuneAssignmentConfiguration( method, loops );
         Timer timer;
         timer.start();
         for( int i = 0; i < loops; i++ ) {
            reorder();
            assignFused( method );
         }
         timer.stop();
         tunedReorderingPaysOff = timer.getRealTime() < unsortedTime;
         reorderingPaysOff = *tunedReorderingPaysOff;
         reordered = false;
      }
   }

   /**
    * \brief Selects the configuration of \ref assignFused for the current order of the points, see \ref tuneAssignment.
    *
    * \return The time of `loops` assignments with the selected configuration.
    */
   double
   tuneAssignmentConfiguration( Assignment method, int loops )
   {
      AssignmentConfiguration& configuration = assignmentConfigurations[ reordered ? 1 : 0 ];
      auto measure = [ & ]()
      {
         assignFused( method );
         Timer timer;
         timer.start();
         for( int i = 0; i < loops; i++ )
            assignFused( method );
         timer.stop();
         return timer.getRealTime();
      };
      std::vector< Index > candidates{ clusters };
      for( Index tileSize : { 64, 32 } )
         if( tileSize < clusters )
            candidates.push_back( tileSize );
      configuration = {};
      double bestTime = std::numeric_limits< double >::max();
      Index best = clusters;
      for( Index tileSize : candidates ) {
         configuration.tileSize = static_cast< int >( tileSize );
         const double time = measure();
         if( time < bestTime ) {
            bestTime = time;
            best = tileSize;
         }
      }
      configuration.tileSize = static_cast< int >( best );
      if( isGPU() && method != Assignment::FusedLambda ) {
         Index tileColumns = 1;
         while( tileColumns < best )
            tileColumns *= 2;
         std::pair< int, int > bestLanes{ 0, 0 };
         for( const auto& [ lanes, columnsPerLane ] : laneConfigurations ) {
            if( lanes * columnsPerLane > tileColumns || 4 * lanes * columnsPerLane < tileColumns )
               continue;
            configuration.lanes = lanes;
            configuration.columnsPerLane = columnsPerLane;
            const double time = measure();
            if( time < bestTime ) {
               bestTime = time;
               bestLanes = { lanes, columnsPerLane };
            }
         }
         configuration.lanes = bestLanes.first;
         configuration.columnsPerLane = bestLanes.second;
      }
      return bestTime;
   }

   //! \brief Assigns every point to the nearest centroid, see \ref Assignment::Unfused.
   void
   assignUnfused()
   {
      computeCentroidNorms();
      if( dots.getRows() != X.getRows() || dots.getColumns() != clusters )
         dots.setDimensions( X.getRows(), clusters );
      Matrices::sparseDenseMatrixProduct( getAssignedPoints(), centroidsT, dots );
      const Index K = clusters;
      const auto D = dots.getConstView();
      const auto norms = centroidNorms.getConstView();
      auto keysView = keys.getView();
      Algorithms::parallelFor< Device >(
         0,
         X.getRows(),
         [ = ] __cuda_callable__( Index row ) mutable
         {
            std::uint64_t best = std::numeric_limits< std::uint64_t >::max();
            for( Index cluster = 0; cluster < K; cluster++ ) {
               const std::uint64_t key = packDistance( norms[ cluster ] - 2 * D( row, cluster ), cluster );
               best = key < best ? key : best;
            }
            keysView[ row ] = best;
         } );
      finishAssignment();
   }

   /**
    * \brief Extracts the labels and the distances from the keys.
    *
    * The keys are in the order of the rows of the assigned matrix, the labels and the distances are in the order of the
    * points, see \ref reorder.
    */
   void
   finishAssignment()
   {
      const auto keysView = keys.getConstView();
      const auto norms = squaredNorms.getConstView();
      const auto orderView = order.getConstView();
      const bool sorted = reordered;
      auto labelsView = labels.getView();
      auto distancesView = distances.getView();
      Algorithms::parallelFor< Device >(
         0,
         X.getRows(),
         [ = ] __cuda_callable__( Index row ) mutable
         {
            const Index point = sorted ? orderView[ row ] : row;
            labelsView[ point ] = unpackCluster( keysView[ row ] );
            distancesView[ point ] = norms[ point ] + unpackDistance( keysView[ row ] );
         } );
      if( shortLabels.getSize() > 0 ) {
         auto shortLabelsView = shortLabels.getView();
         Algorithms::parallelFor< Device >(
            0,
            X.getRows(),
            [ = ] __cuda_callable__( Index row ) mutable
            {
               const Index point = sorted ? orderView[ row ] : row;
               shortLabelsView[ point ] = unpackCluster( keysView[ row ] );
            } );
      }
   }

   //! \brief Returns the matrix of the points used by the assignment, i.e. \ref X or its rows sorted by \ref reorder.
   [[nodiscard]] const MatrixType&
   getAssignedPoints() const
   {
      return reordered ? sortedX : X;
   }

   /**
    * \brief Sorts the points by their clusters for the assignment.
    *
    * The points of one cluster have similar features, so the rows of `C^T` read for the consecutive points of the sorted
    * matrix are more likely in the cache. The sorted matrix is used only by the assignment, `order[ row ]` is the point of
    * its row.
    */
   void
   reorder()
   {
      const Index n = X.getRows();
      countPoints();
      positions = counts;
      Algorithms::inplaceExclusiveScan( positions );
      order.setSize( n );
      const auto labelsView = labels.getConstView();
      auto positionsView = positions.getView();
      auto orderView = order.getView();
      Algorithms::parallelFor< Device >(
         0,
         n,
         [ = ] __cuda_callable__( Index point ) mutable
         {
            orderView[ Algorithms::AtomicOperations< Device >::add( positionsView[ labelsView[ point ] ], Index( 1 ) ) ] =
               point;
         } );
      // the rows of X are copied in the new order, X has the CSR format, so only the offsets of the rows differ
      if( sortedX.getRows() != n )
         sortedX = X;
      const auto offsets = X.getSegments().getOffsets().getConstView();
      auto sortedOffsets = sortedX.getSegments().getOffsets().getView();
      Algorithms::parallelFor< Device >(
         0,
         n + 1,
         [ = ] __cuda_callable__( Index row ) mutable
         {
            sortedOffsets[ row ] = row < n ? offsets[ orderView[ row ] + 1 ] - offsets[ orderView[ row ] ] : 0;
         } );
      Algorithms::inplaceExclusiveScan( sortedOffsets );
      const Index* columns = X.getColumnIndexes().getData();
      const Real* values = X.getValues().getData();
      Matrices::forAllElements(
         sortedX,
         [ = ] __cuda_callable__( Index row, Index localIdx, Index& column, Real& value )
         {
            const Index source = offsets[ orderView[ row ] ] + localIdx;
            column = columns[ source ];
            value = values[ source ];
         } );
      reordered = true;
   }

   /**
    * \brief Sets how often the points are sorted by their clusters for the assignment, see \ref reorder.
    *
    * \param period The points are sorted before the iterations 1, 1 + period, 1 + 2 * period, ... (counted from 0 after
    *        \ref setCentroids), 0 means never. A negative period means that the points are sorted before every iteration
    *        when it pays off, which is measured by \ref tuneAssignment, or otherwise, the assignment of the iteration 1
    *        with the sorted points (including the sorting) is compared with the assignment of the iteration 0. It is used
    *        only on GPUs.
    */
   void
   setReorderingPeriod( int period )
   {
      reorderingPeriod = period;
   }

   //! \brief Counts the points of the clusters.
   void
   countPoints()
   {
      counts.setValue( 0 );
      const auto labelsView = labels.getConstView();
      auto countsView = counts.getView();
      Algorithms::parallelFor< Device >(
         0,
         X.getRows(),
         [ = ] __cuda_callable__( Index row ) mutable
         {
            Algorithms::AtomicOperations< Device >::add( countsView[ labelsView[ row ] ], Index( 1 ) );
         } );
   }

   //! \brief Replaces the centroids by the means of their points, see \ref Update::SpMM.
   void
   updateSpMM()
   {
      countPoints();
      const Index K = clusters;
      const Index d = X.getColumns();
      const auto labelsView = labels.getConstView();
      const auto countsView = counts.getConstView();
      auto CT = centroidsT.getView().getValues().getData();
      auto C = centroids.getView().getValues().getData();
      // the indicator matrix of the clusters, Z( i, j ) = 1 when the point i is in the cluster j
      auto fetchZ = [ = ] __cuda_callable__( Index point, Index cluster ) -> Real
      {
         return labelsView[ point ] == cluster ? Real( 1 ) : Real( 0 );
      };
      auto storeC = [ = ] __cuda_callable__( Index feature, int cluster, const Real& sum ) mutable
      {
         const Index count = countsView[ cluster ];
         if( count > 0 ) {
            const Real value = sum / count;
            CT[ feature * K + cluster ] = value;
            C[ cluster * d + feature ] = value;
         }
      };
      Matrices::sparseDenseMatrixProductWithLambdas< Algorithms::Segments::RowMajorOrder >( XT, K, fetchZ, storeC );
   }

   //! \brief Replaces the centroids by the means of their points, see \ref Update::Atomic.
   void
   updateAtomic()
   {
      // the centroids of the empty clusters are kept in C
      syncCentroids();
      countPoints();
      const Index K = clusters;
      const Index d = X.getColumns();
      centroidsT.getValues().setValue( 0 );
      const auto labelsView = labels.getConstView();
      const auto countsView = counts.getConstView();
      auto CT = centroidsT.getView().getValues().getData();
      auto C = centroids.getView().getValues().getData();
      Matrices::forAllElements(
         X,
         [ = ] __cuda_callable__( Index row, Index, Index column, const Real& value )
         {
            Algorithms::AtomicOperations< Device >::add( CT[ column * K + labelsView[ row ] ], value );
         } );
      // the centroids of the empty clusters are kept in C
      Algorithms::parallelFor< Device >(
         0,
         d * K,
         [ = ] __cuda_callable__( Index idx ) mutable
         {
            const Index feature = idx / K;
            const Index cluster = idx % K;
            const Index count = countsView[ cluster ];
            if( count > 0 ) {
               const Real value = CT[ idx ] / count;
               CT[ idx ] = value;
               C[ cluster * d + feature ] = value;
            }
            else
               CT[ idx ] = C[ cluster * d + feature ];
         } );
   }

   //! \brief Replaces the centroids by the means of their points, see \ref Update::Histogram.
   void
   updateHistogram()
   {
      countPoints();
      const Index K = clusters;
      const Index d = X.getColumns();
      const auto countsView = counts.getConstView();
      auto CT = centroidsT.getView().getValues().getData();
      if constexpr( ! isGPU() ) {
         // every row of X^T is processed by one thread, so the sums are accumulated directly in the row of C^T
         const auto XTView = XT.getConstView();
         const auto labelsView = labels.getConstView();
         auto C = centroids.getView().getValues().getData();
         Algorithms::parallelFor< Device >(
            0,
            d,
            [ = ]( Index feature ) mutable
            {
               thread_local std::vector< Real > sums;
               sums.assign( K, 0 );
               const auto row = XTView.getRow( feature );
               for( Index i = 0; i < row.getSize(); i++ )
                  sums[ labelsView[ row.getColumnIndex( i ) ] ] += row.getValue( i );
               for( Index cluster = 0; cluster < K; cluster++ ) {
                  const Index count = countsView[ cluster ];
                  if( count > 0 ) {
                     const Real value = sums[ cluster ] / count;
                     CT[ feature * K + cluster ] = value;
                     C[ cluster * d + feature ] = value;
                  }
               }
            } );
      }
      else {
#if defined( __CUDACC__ ) || defined( __HIP__ )
         if( shortLabels.getSize() == 0 )
            throw std::invalid_argument( "SparseKMeans: too many clusters for the histogram update" );
         // more copies of the sums reduce the conflicts of the atomic operations in shared memory for few clusters
         const int copies = K <= 32 ? 4 : 1;
         const std::size_t groupMemory = std::size_t( K ) * copies * sizeof( Real );
         constexpr int groupSize = 32;
         constexpr std::size_t maxSharedMemory = 48 * 1024;
         if( groupMemory > maxSharedMemory )
            throw std::invalid_argument( "SparseKMeans: too many clusters for the histogram update" );
         const int groupsPerBlock = std::min( std::size_t( 8 ), maxSharedMemory / groupMemory );
         const Index pieces = pieceRows.getSize();
         longSums.setValue( 0 );
         Backend::LaunchConfiguration config;
         config.blockSize.x = groupsPerBlock * groupSize;
         config.gridSize.x =
            std::min( roundUpDivision( pieces, Index( groupsPerBlock ) ), Index( Backend::getMaxGridXSize() ) );
         config.dynamicSharedMemorySize = groupsPerBlock * groupMemory;
         Backend::launchKernelAsync(
            detail::histogramUpdateKernel< Real, Index, std::uint16_t >,
            config,
            XT.getSegments().getOffsets().getData(),
            XT.getColumnIndexes().getData(),
            XT.getValues().getData(),
            pieceRows.getData(),
            pieceBegins.getData(),
            pieceLongRows.getData(),
            pieces,
            pieceSize,
            shortLabels.getData(),
            counts.getData(),
            K,
            copies,
            CT,
            longSums.getData() );
         const auto longRowsView = longRows.getConstView();
         const auto longSumsView = longSums.getConstView();
         Algorithms::parallelFor< Device >(
            0,
            longRows.getSize() * K,
            [ = ] __cuda_callable__( Index i ) mutable
            {
               const Index cluster = i % K;
               const Index count = countsView[ cluster ];
               if( count > 0 )
                  CT[ longRowsView[ i / K ] * K + cluster ] = longSumsView[ i ] / count;
            } );
         centroidsValid = false;
#endif
      }
   }

   //! \brief Copies `C^T` to `C` when \ref Update::Histogram on GPUs has updated only `C^T`.
   void
   syncCentroids() const
   {
      if( centroidsValid )
         return;
      const Index K = clusters;
      const Index d = X.getColumns();
      const auto CT = centroidsT.getConstView();
      auto C = centroids.getView();
      Algorithms::parallelFor< Device >(
         0,
         d * K,
         [ = ] __cuda_callable__( Index idx ) mutable
         {
            const Index cluster = idx / d;
            const Index feature = idx % d;
            C( cluster, feature ) = CT( feature, cluster );
         } );
      centroidsValid = true;
   }

   //! \brief Performs one iteration of Lloyd's algorithm.
   void
   iterate( Assignment assignment, Update update )
   {
      Timer timer;
      timer.start();
      if( reorderingPeriod > 0 && iteration > 0 && ( iteration - 1 ) % reorderingPeriod == 0 )
         reorder();
      // on the host, the sorting did not pay off in the benchmarks (the caches are large), so it is not tried
      const bool adaptive = reorderingPeriod < 0 && isGPU();
      // whether the sorting pays off is known from tuneAssignment or it is tried in the iteration 1
      const bool trial = adaptive && ! tunedReorderingPaysOff.has_value();
      if( adaptive && iteration >= 1 && ( reorderingPaysOff || ( trial && iteration == 1 ) ) )
         reorder();
      if( assignment == Assignment::Unfused )
         assignUnfused();
      else
         assignFused( assignment );
      if( trial && iteration <= 1 ) {
         // the assignment ends by a synchronous kernel, so the timer measures its run
         timer.stop();
         if( iteration == 0 )
            unsortedAssignmentTime = timer.getRealTime();
         else {
            reorderingPaysOff = timer.getRealTime() < unsortedAssignmentTime;
            reordered = reorderingPaysOff;
         }
      }
      iteration++;
      if( update == Update::SpMM )
         updateSpMM();
      else if( update == Update::Atomic )
         updateAtomic();
      else
         updateHistogram();
   }

   //! \brief Returns the sum of the squared distances of the points to their centroids from the last assignment.
   [[nodiscard]] Real
   getInertia() const
   {
      return TNL::sum( distances );
   }

   [[nodiscard]] const IndexVectorType&
   getLabels() const
   {
      return labels;
   }

   [[nodiscard]] const DenseMatrixType&
   getCentroids() const
   {
      syncCentroids();
      return centroids;
   }

   //! \brief The maximal number of elements of a piece of a row of `X^T` in \ref Update::Histogram on GPUs.
   static constexpr Index pieceSize = 256;

protected:
   const MatrixType& X;
   MatrixType XT, ones;
   Index clusters;
   VectorType squaredNorms, distances, centroidNorms, longSums;
   KeysType keys;
   IndexVectorType labels, counts, pieceRows, pieceBegins, pieceLongRows, longRows;
   //! The labels for \ref Update::Histogram on GPUs, they are smaller, so that more of them fit in the cache.
   Containers::Array< std::uint16_t, Device, Index > shortLabels;
   DenseMatrixType centroidsT, dots;
#if defined( __CUDACC__ ) || defined( __HIP__ )
   //! The copy of `C^T` in half precision for \ref Assignment::FusedHalf.
   Containers::Array< __half, Device, Index > halfCentroidsT;
#endif
   //! `C` is not updated by \ref Update::Histogram on GPUs, see \ref syncCentroids.
   mutable DenseMatrixType centroids;
   mutable bool centroidsValid = true;
   //! The points sorted by their clusters for the assignment, see \ref reorder.
   MatrixType sortedX;
   IndexVectorType order, positions;
   bool reordered = false;
   int reorderingPeriod = 0;
   //! The time of the assignment of the iteration 0 and whether the sorting pays off for the negative reordering period,
   //! see \ref setReorderingPeriod.
   double unsortedAssignmentTime = 0;
   bool reorderingPaysOff = false;
   //! Whether the sorting pays off as measured by \ref tuneAssignment.
   std::optional< bool > tunedReorderingPaysOff;
   int iteration = 0;
   //! The configurations of \ref assignFused for the points in the original order and sorted by \ref reorder, see
   //! \ref tuneAssignment.
   AssignmentConfiguration assignmentConfigurations[ 2 ];
};

}  // namespace TNL::Benchmarks::KMeans
