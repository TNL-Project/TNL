// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include <TNL/Algorithms/AtomicOperations.h>
#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Functional.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseMatrixView.h>
#include <TNL/Matrices/reduce.h>
#include <TNL/Matrices/SparseDenseOperations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Matrices/traverse.h>

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
   Atomic
};

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
      if constexpr( ! std::is_same_v< Device, Devices::Host > )
         setOnes();
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

   /**
    * \brief Assigns every point to the nearest centroid, see \ref Assignment::Fused.
    *
    * \param direct Whether the centroids are read by \ref TNL::Matrices::DirectFetch
    *        (\ref Assignment::Fused) or by a lambda function (\ref Assignment::FusedLambda).
    */
   void
   assignFused( bool direct = true )
   {
      computeCentroidNorms();
      keys.setValue( std::numeric_limits< std::uint64_t >::max() );
      const Index K = clusters;
      const auto CT = centroidsT.getConstView().getValues().getData();
      const auto norms = centroidNorms.getConstView();
      auto keysView = keys.getView();
      auto fetchB = [ = ] __cuda_callable__( Index feature, Index cluster ) -> Real
      {
         return CT[ feature * K + cluster ];
      };
      auto storeC = [ = ] __cuda_callable__( Index row, int cluster, const Real& dot ) mutable
      {
         atomicMinKey( keysView[ row ], packDistance( norms[ cluster ] - 2 * dot, cluster ) );
      };
      if( direct )
         Matrices::sparseDenseMatrixProductWithLambdas(
            X, K, Matrices::directFetch< Algorithms::Segments::RowMajorOrder >( CT, K ), storeC );
      else
         Matrices::sparseDenseMatrixProductWithLambdas< Algorithms::Segments::RowMajorOrder >( X, K, fetchB, storeC );
      finishAssignment();
   }

   //! \brief Assigns every point to the nearest centroid, see \ref Assignment::Unfused.
   void
   assignUnfused()
   {
      computeCentroidNorms();
      if( dots.getRows() != X.getRows() || dots.getColumns() != clusters )
         dots.setDimensions( X.getRows(), clusters );
      Matrices::sparseDenseMatrixProduct( X, centroidsT, dots );
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

   //! \brief Extracts the labels and the distances from the keys.
   void
   finishAssignment()
   {
      const auto keysView = keys.getConstView();
      const auto norms = squaredNorms.getConstView();
      auto labelsView = labels.getView();
      auto distancesView = distances.getView();
      Algorithms::parallelFor< Device >(
         0,
         X.getRows(),
         [ = ] __cuda_callable__( Index row ) mutable
         {
            labelsView[ row ] = unpackCluster( keysView[ row ] );
            distancesView[ row ] = norms[ row ] + unpackDistance( keysView[ row ] );
         } );
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

   //! \brief Performs one iteration of Lloyd's algorithm.
   void
   iterate( Assignment assignment, Update update )
   {
      if( assignment == Assignment::Fused )
         assignFused();
      else if( assignment == Assignment::FusedLambda )
         assignFused( false );
      else
         assignUnfused();
      if( update == Update::SpMM )
         updateSpMM();
      else
         updateAtomic();
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
      return centroids;
   }

protected:
   const MatrixType& X;
   MatrixType XT, ones;
   Index clusters;
   VectorType squaredNorms, distances, centroidNorms;
   KeysType keys;
   IndexVectorType labels, counts;
   DenseMatrixType centroidsT, centroids, dots;
};

}  // namespace TNL::Benchmarks::KMeans
