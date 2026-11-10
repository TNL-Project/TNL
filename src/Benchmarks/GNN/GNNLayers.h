// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

#if defined( __CUDACC__ ) && defined( HAVE_CUBLAS )
   #include <cublas_v2.h>
#endif
#if defined( HAVE_EIGEN )
   #include <Eigen/Core>
#endif

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/DenseOperations.h>
#include <TNL/Matrices/SparseDenseOperations.h>
#include <TNL/Matrices/SparseMatrix.h>

namespace TNL::Benchmarks::GNN {

/**
 * \brief Returns a pseudo-random number from `[-1, 1)` determined by `seed`, `i` and `j`.
 *
 * It is used for the features and the weights, so that the reference implementations in `gnn-reference.py` compute
 * the same model on the same data without exchanging files (the same function is implemented there with NumPy).
 */
__cuda_callable__
inline double
hashUniform( std::uint64_t seed, std::uint64_t i, std::uint64_t j )
{
   std::uint64_t x = seed * 0x9E3779B97F4A7C15ull + i * 0xBF58476D1CE4E5B9ull + j * 0x94D049BB133111EBull;
   x ^= x >> 30;
   x *= 0xBF58476D1CE4E5B9ull;
   x ^= x >> 27;
   x *= 0x94D049BB133111EBull;
   x ^= x >> 31;
   return double( x >> 11 ) * 0x1.0p-52 - 1.0;
}

//! \brief Dense row-major matrix of the node features.
template< typename Real, typename Device, typename Index >
using FeaturesType = Matrices::DenseMatrix< Real, Device, Index, Algorithms::Segments::RowMajorOrder >;

/**
 * \brief Fills the matrix by `scale * hashUniform( seed, row, column )`.
 *
 * The function is public, because nvcc does not allow extended lambdas in private methods.
 */
template< typename Real, typename Device, typename Index >
void
fillRandom( FeaturesType< Real, Device, Index >& matrix, std::uint64_t seed, double scale )
{
   const Index columns = matrix.getColumns();
   auto values = matrix.getValues().getView();
   Algorithms::parallelFor< Device >( 0,
                                      matrix.getRows() * columns,
                                      [ = ] __cuda_callable__( Index idx ) mutable
                                      {
                                         values[ idx ] = scale * hashUniform( seed, idx / columns, idx % columns );
                                      } );
}

//! \brief Implementation of the dense matrix product of the linear transformations.
enum class Gemm : std::uint8_t
{
   //! \ref TNL::Matrices::getMatrixProduct
   TNL,
   //! cuBLAS on GPUs, Eigen on the host (when they are available)
   Vendor
};

#if defined( __CUDACC__ ) && defined( HAVE_CUBLAS )
//! \brief Returns the cuBLAS handle used by \ref linear.
inline cublasHandle_t
getCublasHandle()
{
   static cublasHandle_t handle = []
   {
      cublasHandle_t h;
      if( cublasCreate( &h ) != CUBLAS_STATUS_SUCCESS )
         throw std::runtime_error( "cublasCreate failed" );
      return h;
   }();
   return handle;
}
#endif

/**
 * \brief Linear transformation `X * W` of the features, the weights are `fin x fout` row-major.
 *
 * The rows of the row-major dense matrices of TNL are not padded, so the arrays are passed to cuBLAS and Eigen
 * directly.
 */
template< typename Real, typename Device, typename Index >
void
linear( const FeaturesType< Real, Device, Index >& X,
        const FeaturesType< Real, Device, Index >& W,
        FeaturesType< Real, Device, Index >& result,
        Gemm gemm )
{
   if( gemm == Gemm::TNL ) {
      Matrices::getMatrixProduct( result, X, W, Real( 1 ) );
      return;
   }
   const Index m = X.getRows();
   const Index k = X.getColumns();
   const Index n = W.getColumns();
   if( result.getRows() != m || result.getColumns() != n )
      result.setDimensions( m, n );
#if defined( __CUDACC__ ) && defined( HAVE_CUBLAS )
   if constexpr( std::is_same_v< Device, Devices::GPU > ) {
      // the row-major product C = X * W is the column-major product C^T = W^T * X^T
      const Real alpha = 1;
      const Real beta = 0;
      cublasStatus_t status;
      if constexpr( std::is_same_v< Real, float > )
         status = cublasSgemm( getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, W.getValues().getData(), n,
                               X.getValues().getData(), k, &beta, result.getValues().getData(), n );
      else
         status = cublasDgemm( getCublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, W.getValues().getData(), n,
                               X.getValues().getData(), k, &beta, result.getValues().getData(), n );
      if( status != CUBLAS_STATUS_SUCCESS )
         throw std::runtime_error( "cublasGemm failed" );
      return;
   }
#endif
#if defined( HAVE_EIGEN )
   if constexpr( std::is_same_v< Device, Devices::Host > ) {
      using Matrix = Eigen::Matrix< Real, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor >;
      const Eigen::Map< const Matrix > mapX( X.getValues().getData(), m, k );
      const Eigen::Map< const Matrix > mapW( W.getValues().getData(), k, n );
      Eigen::Map< Matrix > mapC( result.getValues().getData(), m, n );
      mapC.noalias() = mapX * mapW;
      return;
   }
#endif
   throw std::runtime_error( "no vendor implementation of the dense matrix product for this device" );
}

/**
 * \brief Graph convolutional layer of Kipf and Welling, `H' = act( A_hat * H * W + b )`.
 *
 * `A_hat = D^{-1/2} ( A + I ) D^{-1/2}` is stored in the values of the sparse matrix (see \ref GCN::normalize), the
 * product with `T = H * W` is computed by \ref TNL::Matrices::sparseDenseMatrixProductWithLambdas with `T` given by the
 * dense matrix and the bias and the activation applied by the lambda function storing the result.
 */
template< typename Real, typename Device, typename Index >
struct GCNLayer
{
   using MatrixType = Matrices::SparseMatrix< Real, Device, Index >;
   using DenseType = FeaturesType< Real, Device, Index >;
   using VectorType = Containers::Vector< Real, Device, Index >;

   GCNLayer( Index inputs, Index outputs, std::uint64_t seed, bool relu )
   : weights( inputs, outputs ),
     bias( outputs ),
     relu( relu )
   {
      // Glorot uniform initialization, the bias is small
      fillRandom( weights, 100 + seed, std::sqrt( 6.0 / ( inputs + outputs ) ) );
      DenseType b( 1, outputs );
      fillRandom( b, 300 + seed, 0.1 );
      bias = b.getValues();
   }

   //! \brief Computes the linear transformation of the features.
   void
   transform( const DenseType& H, Gemm gemm )
   {
      linear( H, weights, transformed, gemm );
   }

   //! \brief Computes the aggregation of the transformed features, the bias and the activation.
   void
   aggregate( const MatrixType& Ahat, DenseType& output )
   {
      if( output.getRows() != Ahat.getRows() || output.getColumns() != weights.getColumns() )
         output.setDimensions( Ahat.getRows(), weights.getColumns() );
      const Index outputs = weights.getColumns();
      auto out = output.getValues().getView();
      const auto b = bias.getConstView();
      const bool applyRelu = relu;
      auto store = [ = ] __cuda_callable__( Index row, int j, const Real& value ) mutable
      {
         const Real v = value + b[ j ];
         out[ row * outputs + j ] = applyRelu && v < 0 ? Real( 0 ) : v;
      };
      Matrices::sparseDenseMatrixProductWithLambdas( Ahat, outputs, transformed.getConstView(), store );
   }

   DenseType weights;
   VectorType bias;
   bool relu;
   DenseType transformed;
};

/**
 * \brief GraphSAGE layer with the mean aggregation, `H' = act( mean_{N(i)}( H ) * W_l + H * W_r + b )`.
 *
 * It is the layer `SAGEConv` of PyTorch Geometric. The mean is the product with the adjacency matrix with the values
 * `1 / deg( i )` (see \ref SAGE::normalize). Since the mean and the linear transformation commute, the features are
 * transformed first, and the root term, the bias and the activation are applied by the lambda function storing the
 * result of the product.
 */
template< typename Real, typename Device, typename Index >
struct SAGELayer
{
   using MatrixType = Matrices::SparseMatrix< Real, Device, Index >;
   using DenseType = FeaturesType< Real, Device, Index >;
   using VectorType = Containers::Vector< Real, Device, Index >;

   SAGELayer( Index inputs, Index outputs, std::uint64_t seed, bool relu )
   : weightsNeighbors( inputs, outputs ),
     weightsRoot( inputs, outputs ),
     bias( outputs ),
     relu( relu )
   {
      fillRandom( weightsNeighbors, 100 + seed, std::sqrt( 6.0 / ( inputs + outputs ) ) );
      fillRandom( weightsRoot, 200 + seed, std::sqrt( 6.0 / ( inputs + outputs ) ) );
      DenseType b( 1, outputs );
      fillRandom( b, 300 + seed, 0.1 );
      bias = b.getValues();
   }

   //! \brief Computes the linear transformations of the features.
   void
   transform( const DenseType& H, Gemm gemm )
   {
      linear( H, weightsNeighbors, transformed, gemm );
      linear( H, weightsRoot, root, gemm );
   }

   //! \brief Computes the aggregation of the transformed features, the root term, the bias and the activation.
   void
   aggregate( const MatrixType& Amean, DenseType& output )
   {
      if( output.getRows() != Amean.getRows() || output.getColumns() != weightsNeighbors.getColumns() )
         output.setDimensions( Amean.getRows(), weightsNeighbors.getColumns() );
      const Index outputs = weightsNeighbors.getColumns();
      auto out = output.getValues().getView();
      const auto r = root.getValues().getConstView();
      const auto b = bias.getConstView();
      const bool applyRelu = relu;
      auto store = [ = ] __cuda_callable__( Index row, int j, const Real& value ) mutable
      {
         const Real v = value + r[ row * outputs + j ] + b[ j ];
         out[ row * outputs + j ] = applyRelu && v < 0 ? Real( 0 ) : v;
      };
      Matrices::sparseDenseMatrixProductWithLambdas( Amean, outputs, transformed.getConstView(), store );
   }

   DenseType weightsNeighbors, weightsRoot;
   VectorType bias;
   bool relu;
   DenseType transformed, root;
};

/**
 * \brief Two-layer graph neural network for the inference.
 *
 * \tparam Layer \ref GCNLayer or \ref SAGELayer.
 */
template< typename Layer >
struct TwoLayerModel
{
   using MatrixType = typename Layer::MatrixType;
   using DenseType = typename Layer::DenseType;
   using IndexType = typename MatrixType::IndexType;

   TwoLayerModel( IndexType inputs, IndexType hidden, IndexType outputs )
   : first( inputs, hidden, 0, true ),
     second( hidden, outputs, 1, false )
   {}

   //! \brief Computes the output of the model for the features `X`.
   void
   forward( const MatrixType& A, const DenseType& X, DenseType& output, Gemm gemm )
   {
      first.transform( X, gemm );
      first.aggregate( A, hiddenFeatures );
      second.transform( hiddenFeatures, gemm );
      second.aggregate( A, output );
   }

   //! \brief Computes only the aggregations of both layers, the transformed features from the last \ref forward are used.
   void
   aggregate( const MatrixType& A, DenseType& output )
   {
      first.aggregate( A, hiddenFeatures );
      second.aggregate( A, output );
   }

   Layer first, second;
   DenseType hiddenFeatures;
};

/**
 * \brief Returns the normalized adjacency matrix of GCN, `D^{-1/2} ( A + I ) D^{-1/2}`, where `D` is the diagonal matrix
 * of the degrees of `A + I`.
 *
 * The pattern of `A` is used, its values are ignored. It runs on the host, it is a preprocessing.
 */
template< typename Real, typename Index >
Matrices::SparseMatrix< Real, Devices::Host, Index >
normalizeGCN( const Matrices::SparseMatrix< Real, Devices::Host, Index >& A )
{
   const Index n = A.getRows();
   Containers::Vector< Index, Devices::Host, Index > rowLengths( n );
   std::vector< double > inverseSqrtDegree( n );
   for( Index row = 0; row < n; row++ ) {
      const auto r = A.getRow( row );
      Index length = 0;
      bool diagonal = false;
      for( Index localIdx = 0; localIdx < r.getSize(); localIdx++ )
         if( r.getColumnIndex( localIdx ) >= 0 ) {
            length++;
            diagonal = diagonal || r.getColumnIndex( localIdx ) == row;
         }
      rowLengths[ row ] = diagonal ? length : length + 1;
      inverseSqrtDegree[ row ] = 1.0 / std::sqrt( double( rowLengths[ row ] ) );
   }
   Matrices::SparseMatrix< Real, Devices::Host, Index > Ahat( n, n );
   Ahat.setRowCapacities( rowLengths );
   for( Index row = 0; row < n; row++ ) {
      const auto r = A.getRow( row );
      auto out = Ahat.getRow( row );
      Index localOut = 0;
      bool diagonal = false;
      for( Index localIdx = 0; localIdx < r.getSize(); localIdx++ ) {
         const Index column = r.getColumnIndex( localIdx );
         if( column < 0 )
            continue;
         diagonal = diagonal || column == row;
         out.setElement( localOut++, column, inverseSqrtDegree[ row ] * inverseSqrtDegree[ column ] );
      }
      // the order of the elements in the rows does not matter for the product
      if( ! diagonal )
         out.setElement( localOut++, row, inverseSqrtDegree[ row ] * inverseSqrtDegree[ row ] );
   }
   return Ahat;
}

/**
 * \brief Returns the adjacency matrix for the mean aggregation of GraphSAGE, the values of the row `i` are `1 / deg( i )`.
 *
 * The pattern of `A` is used, its values are ignored. It runs on the host, it is a preprocessing.
 */
template< typename Real, typename Index >
Matrices::SparseMatrix< Real, Devices::Host, Index >
normalizeSAGE( const Matrices::SparseMatrix< Real, Devices::Host, Index >& A )
{
   Matrices::SparseMatrix< Real, Devices::Host, Index > Amean;
   Amean = A;
   for( Index i = 0; i < Amean.getRows(); i++ ) {
      auto row = Amean.getRow( i );
      Index length = 0;
      for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ )
         if( row.getColumnIndex( localIdx ) >= 0 )
            length++;
      for( Index localIdx = 0; localIdx < row.getSize(); localIdx++ )
         if( row.getColumnIndex( localIdx ) >= 0 )
            row.setValue( localIdx, Real( 1.0 / length ) );
   }
   return Amean;
}

}  // namespace TNL::Benchmarks::GNN
