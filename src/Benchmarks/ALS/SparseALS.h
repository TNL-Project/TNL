// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Algorithms/reduce.h>
#include <TNL/Backend/Functions.h>
#include <TNL/Containers/Array.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseDenseOperations.h>
#include <TNL/Matrices/SparseMatrix.h>
#include <TNL/Timer.h>

namespace TNL::Benchmarks::ALS {

namespace detail {

/**
 * \brief A group of `Lanes` consecutive threads of a warp processing one row or one element together.
 *
 * The sums are computed by warp shuffles, so all threads of the warp must call \ref sum. With one lane (on the host),
 * \ref sum returns its argument.
 */
template< int Lanes >
struct LaneGroup
{
   [[nodiscard]] static constexpr int
   getSize()
   {
      return Lanes;
   }

   [[nodiscard]] __cuda_callable__
   int
   getRank() const
   {
      return lane;
   }

   //! \brief Returns the sum of `value` over the lanes of the group.
   template< typename T >
   [[nodiscard]] __cuda_callable__
   T
   sum( T value ) const
   {
#if defined( __CUDA_ARCH__ ) || defined( __HIP_DEVICE_COMPILE__ )
      for( int mask = Lanes / 2; mask > 0; mask /= 2 )
         value += Backend::warp_shuffle_xor( value, mask );
#endif
      return value;
   }

   int lane;
};

//! \brief Returns the part of the dot product of the vectors `a` and `b` with `k` elements computed by one lane of
//! `group`.
template< int Lanes, typename Real, typename Index >
[[nodiscard]] __cuda_callable__
Real
partialDot( const LaneGroup< Lanes >& group, const Real* a, const Real* b, Index k )
{
   Real result = 0;
   for( Index j = group.getRank(); j < k; j += Lanes )
      result += a[ j ] * b[ j ];
   return result;
}

//! \brief Consecutive elements loaded by one vectorized load of 16 bytes.
template< typename Real >
struct alignas( 16 ) Packet
{
   static constexpr int size = 16 / sizeof( Real );
   Real values[ size ];
};

//! \brief The same as \ref partialDot with the elements loaded by \ref Packet, `a` and `b` must be aligned to 16 bytes and
//! `k` must be a multiple of the size of the packet.
template< int Lanes, typename Real, typename Index >
[[nodiscard]] __cuda_callable__
Real
partialDotPacked( const LaneGroup< Lanes >& group, const Real* a, const Real* b, Index k )
{
   using PacketType = Packet< Real >;
   const auto* packetsA = reinterpret_cast< const PacketType* >( a );
   const auto* packetsB = reinterpret_cast< const PacketType* >( b );
   Real result = 0;
   for( Index j = group.getRank(); j < k / PacketType::size; j += Lanes ) {
      const PacketType packetA = packetsA[ j ];
      const PacketType packetB = packetsB[ j ];
      for( int i = 0; i < PacketType::size; i++ )
         result += packetA.values[ i ] * packetB.values[ i ];
   }
   return result;
}

/**
 * \brief Returns the dot product of the vectors `a` and `b` with `k` elements computed by one thread with independent
 * partial sums of the consecutive elements, so that the compiler vectorizes it (the sum of floating point numbers in one
 * variable is not vectorized without `-ffast-math`).
 */
template< typename Real, typename Index >
[[nodiscard]] __cuda_callable__
Real
vectorizedDot( const Real* a, const Real* b, Index k )
{
   constexpr int width = 8;
   Real sums[ width ] = {};
   Index j = 0;
   for( ; j + width <= k; j += width )
      for( int i = 0; i < width; i++ )
         sums[ i ] += a[ j + i ] * b[ j + i ];
   for( ; j < k; j++ )
      sums[ 0 ] += a[ j ] * b[ j ];
   Real result = 0;
   for( int i = 0; i < width; i++ )
      result += sums[ i ];
   return result;
}

//! \brief Returns the dot product of the vectors `a` and `b` with `k` elements computed by the lanes of `group`.
template< int Lanes, typename Real, typename Index >
[[nodiscard]] __cuda_callable__
Real
dot( const LaneGroup< Lanes >& group, const Real* a, const Real* b, Index k )
{
   if constexpr( Lanes == 1 )
      return vectorizedDot( a, b, k );
   else
      return group.sum( partialDot( group, a, b, k ) );
}

/**
 * \brief Returns the element `j` of the product of the symmetric Gram matrix `G` (`k x k`) and the vector `x`.
 *
 * On GPUs, the consecutive columns `j` are computed by consecutive threads, so `G` is read by columns, which are
 * consecutive for consecutive `j`. On the host, one thread computes the consecutive columns of a row and `G` is read by
 * rows, so that the dot product is vectorized.
 */
template< bool ByColumns, typename Real, typename Index >
[[nodiscard]] __cuda_callable__
Real
gramProduct( const Real* G, const Real* x, Index j, Index k )
{
   if constexpr( ! ByColumns )
      return vectorizedDot( x, &G[ j * k ], k );
   Real result = 0;
   for( Index l = 0; l < k; l++ )
      result += x[ l ] * G[ l * k + j ];
   return result;
}

}  // namespace detail

/**
 * \brief The weights of the elements of the confidence matrix computed by \ref SparseALS::sampledProduct.
 *
 * `c` is the confidence and `t` the dot product of the factors of the user and the item.
 */
enum class Weight : std::uint8_t
{
   //! `c - (c - 1) t`, the weights of the residual of the conjugate gradient method.
   Residual,
   //! `(c - 1) t`, the weights of the product of the matrix of the conjugate gradient method.
   Product,
   //! `c + ((c - 1) t - 2 c) t`, the contribution of the element to the loss.
   Loss
};

/**
 * \brief The numbers of lanes and columns per lane of the products of \ref SparseALS on GPUs, see
 * \ref TNL::Matrices::sparseDenseMatrixProductWithLambdasLanesOverColumns.
 */
struct ProductConfiguration
{
   //! The number of lanes, 0 means the default configuration of the product.
   int lanes = 0;
   //! The number of columns per lane.
   int columnsPerLane = 0;

   [[nodiscard]] std::string
   toString() const
   {
      if( lanes == 0 )
         return "default";
      return "L=" + std::to_string( lanes ) + " V=" + std::to_string( columnsPerLane );
   }
};

/**
 * \brief Alternating least squares for implicit feedback (Hu, Koren and Volinsky) with the conjugate gradient method,
 * the same algorithm as `implicit.als.AlternatingLeastSquares` of the implicit library.
 *
 * The confidences `c_ui > 0` are the elements of the sparse matrix `R` (`m` users times `n` items), the preferences are
 * `p_ui = 1` for the elements of `R` and 0 otherwise, the confidences of the other pairs are 1. The factors of the users
 * (`X`, `m x k`) and the items (`Y`, `n x k`) minimize
 *
 *    sum_{u,i} c_ui (p_ui - x_u . y_i)^2 + lambda (|X|^2 + |Y|^2).
 *
 * One iteration solves the factors of the users with the fixed factors of the items and then vice versa. The factor of
 * the user `u` is the solution of
 *
 *    (Y^T Y + lambda I + Y^T (C_u - I) Y) x_u = Y^T C_u p_u,
 *
 * which is approximated by a few steps of the conjugate gradient method starting from the current factor. The systems of
 * all users are solved at once: the residuals and the products of the matrices with the search directions `P` are
 *
 *    r_u = sum_i (c_ui - (c_ui - 1) y_i . x_u) y_i - G x_u,
 *    A p_u = sum_i (c_ui - 1) (y_i . p_u) y_i + G p_u,
 *
 * where `G = Y^T Y + lambda I`. The weights of the sums are the sampled dense-dense product of the factors on the
 * elements of `R` (\ref sampledProduct), they are stored as the values of a copy of `R`. The sums are the sparse-dense
 * product of this matrix and `Y` computed by \ref TNL::Matrices::sparseDenseMatrixProductWithLambdas, whose storer adds
 * the product with `G`. The steps of the conjugate gradient method are computed separately for every user.
 *
 * The methods with lambda functions are public, because nvcc does not allow extended lambdas in private methods.
 */
template< typename Real, typename Device, typename Index >
class SparseALS
{
public:
   using MatrixType = Matrices::SparseMatrix< Real, Device, Index >;
   using VectorType = Containers::Vector< Real, Device, Index >;
   using IndexVectorType = Containers::Vector< Index, Device, Index >;
   using FlagsType = Containers::Array< std::uint8_t, Device, Index >;
   using DenseMatrixType = Matrices::DenseMatrix< Real, Device, Index, Algorithms::Segments::RowMajorOrder >;

   //! \brief The squared norm of the residual below which the conjugate gradient method stops, the same as in implicit.
   static constexpr Real smallResidual = 1e-20;

   /**
    * \brief The confidence matrix of one side of the problem (`R` for the users, `R^T` for the items) with the auxiliary
    * data of its products.
    */
   struct Side
   {
      //! The confidences, the rows are the users (items) and the columns are the items (users).
      MatrixType C;
      //! The copy of `C` whose values are the weights computed by \ref sampledProduct.
      MatrixType W;
      //! The rows of the elements of `C`, for the sampled product on GPUs.
      IndexVectorType rowIndexes;
      //! The configuration of the products, see \ref tuneProducts.
      ProductConfiguration configuration;
   };

   SparseALS( const MatrixType& R, Index factors, Real regularization, int cgSteps = 3 )
   : factors( factors ),
     regularization( regularization ),
     cgSteps( cgSteps )
   {
      if( factors <= 0 )
         throw std::invalid_argument( "SparseALS: wrong number of factors" );
      users.C = R;
      items.C.getTransposition( R );
      for( Side* side : { &users, &items } ) {
         side->W = side->C;
         setRowIndexes( *side );
      }
      const Index maxRows = TNL::max( R.getRows(), R.getColumns() );
      residuals.setSize( maxRows * factors );
      directions.setSize( maxRows * factors );
      products.setSize( maxRows * factors );
      residualNorms.setSize( maxRows );
      active.setSize( maxRows );
      gram.setSize( factors * factors );
      userFactors.setDimensions( R.getRows(), factors );
      itemFactors.setDimensions( R.getColumns(), factors );
   }

   //! \brief Returns whether the computation runs on a GPU.
   [[nodiscard]] static constexpr bool
   isGPU()
   {
      return ! std::is_same_v< Device, Devices::Host > && ! std::is_same_v< Device, Devices::Sequential >;
   }

   //! \brief Sets the rows of the elements of the confidence matrix of `side`.
   void
   setRowIndexes( Side& side )
   {
      side.rowIndexes.setSize( side.C.getValues().getSize() );
      const auto offsets = side.C.getSegments().getOffsets().getConstView();
      auto rowIndexesView = side.rowIndexes.getView();
      Algorithms::parallelFor< Device >(
         0,
         side.C.getRows(),
         [ = ] __cuda_callable__( Index row ) mutable
         {
            for( Index i = offsets[ row ]; i < offsets[ row + 1 ]; i++ )
               rowIndexesView[ i ] = row;
         } );
   }

   //! \brief Sets the factors of the users and the items.
   void
   setFactors( const DenseMatrixType& X, const DenseMatrixType& Y )
   {
      userFactors = X;
      itemFactors = Y;
   }

   [[nodiscard]] const DenseMatrixType&
   getUserFactors() const
   {
      return userFactors;
   }

   [[nodiscard]] const DenseMatrixType&
   getItemFactors() const
   {
      return itemFactors;
   }

   [[nodiscard]] Index
   getFactors() const
   {
      return factors;
   }

   [[nodiscard]] const Side&
   getSide( bool itemsSide ) const
   {
      return itemsSide ? items : users;
   }

   /**
    * \brief Computes `G = F^T F + shift * I` for the factors `F` (`rows x k`, row-major) and stores it in
    * `result` (`k x k`, row-major).
    *
    * The rows of `F` are split into chunks, the Gram matrices of the chunks are summed. On GPUs, every element of the
    * Gram matrix of a chunk is computed by one thread.
    */
   void
   computeGram( const DenseMatrixType& F, Real shift, VectorType& result )
   {
      const Index rows = F.getRows();
      const Index k = factors;
      const Index entries = k * k;
      const Index chunks = isGPU()
                            ? TNL::min( Index( 128 ), TNL::max( Index( 1 ), rows / 256 ) )
                            : TNL::max( Index( 1 ), TNL::min( rows / 64, Index( 4 * Devices::Host::getMaxThreadsCount() ) ) );
      const Index chunkSize = roundUpDivision( rows, chunks );
      chunkGrams.setSize( chunks * entries );
      const auto f = F.getConstView().getValues().getData();
      auto partial = chunkGrams.getView();
      if constexpr( isGPU() ) {
         Algorithms::parallelFor< Device >(
            0,
            chunks * entries,
            [ = ] __cuda_callable__( Index index ) mutable
            {
               const Index chunk = index / entries;
               const Index a = index % entries / k;
               const Index b = index % k;
               const Index end = TNL::min( rows, ( chunk + 1 ) * chunkSize );
               Real sum = 0;
               for( Index row = chunk * chunkSize; row < end; row++ )
                  sum += f[ row * k + a ] * f[ row * k + b ];
               partial[ index ] = sum;
            } );
      }
      else {
         // every thread computes one row of the Gram matrix of a chunk, the inner loop is vectorized (there must be enough
         // iterations, so that parallelFor runs in parallel)
         Algorithms::parallelFor< Device >(
            0,
            chunks * k,
            [ = ] __cuda_callable__( Index index ) mutable
            {
               const Index chunk = index / k;
               const Index a = index % k;
               // the pointers do not alias, so that the inner loop can be vectorized
               Real* __restrict__ g = &partial[ chunk * entries + a * k ];
               const Real* __restrict__ rowsOfChunk = f;
               for( Index b = 0; b < k; b++ )
                  g[ b ] = 0;
               const Index end = TNL::min( rows, ( chunk + 1 ) * chunkSize );
               for( Index row = chunk * chunkSize; row < end; row++ ) {
                  const Real fa = rowsOfChunk[ row * k + a ];
                  for( Index b = 0; b < k; b++ )
                     g[ b ] += fa * rowsOfChunk[ row * k + b ];
               }
            } );
      }
      result.setSize( entries );
      auto resultView = result.getView();
      const auto partialView = chunkGrams.getConstView();
      Algorithms::parallelFor< Device >(
         0,
         entries,
         [ = ] __cuda_callable__( Index index ) mutable
         {
            Real sum = 0;
            for( Index chunk = 0; chunk < chunks; chunk++ )
               sum += partialView[ chunk * entries + index ];
            if( index / k == index % k )
               sum += shift;
            resultView[ index ] = sum;
         } );
   }

   /**
    * \brief Computes the weights of the elements of the confidence matrix of `side` from the dot products of the rows
    * of `P` and `Y` and stores them as the values of `side.W`.
    *
    * On GPUs, every element is processed by a group of `Lanes` threads.
    */
   template< int Lanes >
   void
   sampledProduct( Side& side, const Real* P, const Real* Y, Weight weight )
   {
      const Index k = factors;
      const Index elements = side.C.getValues().getSize();
      const auto rowIndexes = side.rowIndexes.getConstView();
      const auto columns = side.C.getColumnIndexes().getConstView();
      const auto confidences = side.C.getValues().getConstView();
      auto weights = side.W.getValues().getView();
      // the rows of the factors are aligned to the packets when their size is a multiple of the packet size
      const bool packed = isGPU() && k % detail::Packet< Real >::size == 0;
      // all lanes of a warp must take part in the shuffles of the groups
      const Index threads = roundUpDivision( elements * Lanes, Index( 32 ) ) * 32;
      Algorithms::parallelFor< Device >(
         0,
         threads,
         [ = ] __cuda_callable__( Index thread ) mutable
         {
            const detail::LaneGroup< Lanes > group{ static_cast< int >( thread % Lanes ) };
            const Index element = thread / Lanes;
            const Index column = element < elements ? columns[ element ] : Index( -1 );
            Real t = 0;
            if( column >= 0 ) {
               const Real* p = &P[ rowIndexes[ element ] * k ];
               const Real* y = &Y[ column * k ];
               // not constexpr-if, which cannot capture variables in extended lambdas
               if( Lanes == 1 )
                  t = detail::vectorizedDot( p, y, k );
               else
                  t = packed ? detail::partialDotPacked( group, p, y, k ) : detail::partialDot( group, p, y, k );
            }
            t = group.sum( t );
            if( column < 0 || group.getRank() != 0 )
               return;
            const Real c = confidences[ element ];
            if( weight == Weight::Residual )
               weights[ element ] = c - ( c - 1 ) * t;
            else if( weight == Weight::Product )
               weights[ element ] = ( c - 1 ) * t;
            else
               weights[ element ] = c + ( ( c - 1 ) * t - 2 * c ) * t;
         } );
   }

   //! \brief Calls \ref sampledProduct with the number of lanes suitable for the number of factors.
   void
   sampledProduct( Side& side, const Real* P, const Real* Y, Weight weight )
   {
      if constexpr( isGPU() ) {
         if( factors <= 32 )
            sampledProduct< 8 >( side, P, Y, weight );
         else if( factors <= 64 )
            sampledProduct< 16 >( side, P, Y, weight );
         else
            sampledProduct< 32 >( side, P, Y, weight );
      }
      else
         sampledProduct< 1 >( side, P, Y, weight );
   }

   //! \brief The numbers of lanes and columns per lane of the products selected by \ref tuneProducts on GPUs.
   static constexpr std::array< std::pair< int, int >, 10 > laneConfigurations{
      { { 8, 1 }, { 4, 2 }, { 16, 1 }, { 8, 2 }, { 32, 1 }, { 16, 2 }, { 8, 4 }, { 32, 2 }, { 16, 4 }, { 32, 4 } }
   };

   //! \brief Computes the product of `side.W` and `Y` with the configuration of `side`.
   template< int I = 0, typename StoreC >
   void
   multiply( const Side& side, const Real* Y, const StoreC& storeC ) const
   {
      const auto fetchB = Matrices::directFetch< Algorithms::Segments::RowMajorOrder >( Y, factors );
      if constexpr( isGPU() && I < laneConfigurations.size() ) {
         constexpr int lanes = laneConfigurations[ I ].first;
         constexpr int columnsPerLane = laneConfigurations[ I ].second;
         if( side.configuration.lanes == lanes && side.configuration.columnsPerLane == columnsPerLane )
            Matrices::sparseDenseMatrixProductWithLambdasLanesOverColumns< lanes, columnsPerLane >(
               side.W, factors, fetchB, storeC );
         else
            multiply< I + 1 >( side, Y, storeC );
      }
      else
         Matrices::sparseDenseMatrixProductWithLambdas( side.W, factors, fetchB, storeC );
   }

   /**
    * \brief Computes the residuals `r_u = sum_i (c_ui - (c_ui - 1) y_i . x_u) y_i - G x_u` of the rows of `side` and sets
    * the search directions to them.
    */
   void
   computeResiduals( Side& side, const Real* X, const Real* Y )
   {
      if( fused ) {
         forRowsWith< FusedRow >(
            side,
            FusedRowData{ factors,
                          Weight::Residual,
                          side.C.getSegments().getOffsets().getData(),
                          side.C.getColumnIndexes().getData(),
                          side.C.getValues().getData(),
                          gram.getData(),
                          Y,
                          X,
                          residuals.getData(),
                          directions.getData() } );
         return;
      }
      sampledProduct( side, X, Y, Weight::Residual );
      const Index k = factors;
      const auto G = gram.getConstView();
      auto R = residuals.getView();
      auto P = directions.getView();
      auto storeC = [ = ] __cuda_callable__( Index row, int j, const Real& value ) mutable
      {
         const Real gx = detail::gramProduct< isGPU() >( G.getData(), &X[ row * k ], j, k );
         R[ row * k + j ] = P[ row * k + j ] = value - gx;
      };
      multiply( side, Y, storeC );
   }

   //! \brief Computes the products `A p_u = sum_i (c_ui - 1) (y_i . p_u) y_i + G p_u` for the rows of `side`.
   void
   computeProducts( Side& side, const Real* Y )
   {
      const Real* P = directions.getConstView().getData();
      if( fused ) {
         forRowsWith< FusedRow >(
            side,
            FusedRowData{ factors,
                          Weight::Product,
                          side.C.getSegments().getOffsets().getData(),
                          side.C.getColumnIndexes().getData(),
                          side.C.getValues().getData(),
                          gram.getData(),
                          Y,
                          P,
                          products.getData(),
                          nullptr } );
         return;
      }
      sampledProduct( side, P, Y, Weight::Product );
      const Index k = factors;
      const auto G = gram.getConstView();
      auto AP = products.getView();
      auto storeC = [ = ] __cuda_callable__( Index row, int j, const Real& value ) mutable
      {
         AP[ row * k + j ] = value + detail::gramProduct< isGPU() >( G.getData(), &P[ row * k ], j, k );
      };
      multiply( side, Y, storeC );
   }

   /**
    * \brief Calls `function( group, row )` for every row of `side`, on GPUs with a group of 32 threads per row.
    */
   template< typename Function >
   void
   forRows( const Side& side, Function function )
   {
      const Index rows = side.C.getRows();
      if constexpr( isGPU() ) {
         constexpr int lanes = 32;
         Algorithms::parallelFor< Device >(
            0,
            rows * lanes,
            [ = ] __cuda_callable__( Index thread ) mutable
            {
               function( detail::LaneGroup< lanes >{ static_cast< int >( thread % lanes ) }, thread / lanes );
            } );
      }
      else {
         // the rows are distributed dynamically, the work of the rows of the fused products is proportional to their
         // lengths
#ifdef HAVE_OPENMP
         if( Devices::Host::isOMPEnabled() && rows > 512 ) {
   #pragma omp parallel for schedule( dynamic, 64 )
            for( Index row = 0; row < rows; row++ )
               function( detail::LaneGroup< 1 >{ 0 }, row );
            return;
         }
#endif
         for( Index row = 0; row < rows; row++ )
            function( detail::LaneGroup< 1 >{ 0 }, row );
      }
   }

   /**
    * \brief Starts the conjugate gradient method: computes the squared norms of the residuals and marks the rows whose
    * residuals are not small. The factors of the empty rows are set to zero.
    */
   template< int Lanes >
   struct StartRow
   {
      __cuda_callable__
      void
      operator()( const detail::LaneGroup< Lanes >& group, Index row ) const
      {
         const Real norm = detail::dot( group, &R[ row * k ], &R[ row * k ], k );
         if( group.getRank() == 0 )
            norms[ row ] = norm;
         const bool empty = offsets[ row ] == offsets[ row + 1 ];
         if( empty )
            for( Index j = group.getRank(); j < k; j += Lanes )
               X[ row * k + j ] = 0;
         if( group.getRank() == 0 )
            flags[ row ] = ! empty && norm >= smallResidual;
      }

      Index k;
      const Index* offsets;
      const Real* R;
      Real* X;
      Real* norms;
      std::uint8_t* flags;
   };

   //! \brief One step of the conjugate gradient method for the active rows.
   template< int Lanes >
   struct UpdateRow
   {
      __cuda_callable__
      void
      operator()( const detail::LaneGroup< Lanes >& group, Index row ) const
      {
         if( ! flags[ row ] )
            return;
         Real* x = &X[ row * k ];
         Real* r = &R[ row * k ];
         Real* p = &P[ row * k ];
         const Real* ap = &AP[ row * k ];
         const Real oldNorm = norms[ row ];
         const Real alpha = oldNorm / detail::dot( group, p, ap, k );
         Real newNorm = 0;
         for( Index j = group.getRank(); j < k; j += Lanes ) {
            x[ j ] += alpha * p[ j ];
            r[ j ] -= alpha * ap[ j ];
            newNorm += r[ j ] * r[ j ];
         }
         newNorm = group.sum( newNorm );
         if( newNorm < smallResidual ) {
            if( group.getRank() == 0 )
               flags[ row ] = 0;
            return;
         }
         const Real beta = newNorm / oldNorm;
         for( Index j = group.getRank(); j < k; j += Lanes )
            p[ j ] = r[ j ] + beta * p[ j ];
         if( group.getRank() == 0 )
            norms[ row ] = newNorm;
      }

      Index k;
      const Real* AP;
      Real* X;
      Real* R;
      Real* P;
      Real* norms;
      std::uint8_t* flags;
   };

   //! \brief The data of \ref FusedRow.
   struct FusedRowData
   {
      Index k;
      //! \ref Weight::Residual or \ref Weight::Product.
      Weight weight;
      const Index* offsets;
      const Index* columns;
      const Real* confidences;
      const Real* G;
      const Real* Y;
      //! The vectors `v_u` (the factors `X` for the residuals, the search directions `P` for the products).
      const Real* V;
      //! The results.
      Real* out;
      //! The copy of the results (the search directions for the residuals) or `nullptr`.
      Real* copy;
   };

   /**
    * \brief Computes the residual (\ref Weight::Residual) or the product (\ref Weight::Product) of the row `u` in one pass
    * over its elements, i.e. the sampled product and the sparse-dense product fused (FusedMM):
    *
    *    out_u = sum_i w( c_ui, y_i . v_u ) y_i -+ G v_u.
    *
    * Every factor `y_i` is read only once, the dot product and the weighted sum use it from the cache. On the host,
    * every row is processed by one thread. On GPUs, by a group of `Lanes` threads, each of them accumulates at most
    * \ref maxColumnsPerLane elements of the sum, see \ref isFusedSupported.
    */
   template< int Lanes >
   struct FusedRow
   {
      static constexpr int maxColumnsPerLane = 4;

      __cuda_callable__
      void
      operator()( const detail::LaneGroup< Lanes >& group, Index row ) const
      {
         const Index k = data.k;
         const Real* v = &data.V[ row * k ];
         Real* out = &data.out[ row * k ];
         const bool residual = data.weight == Weight::Residual;
         if constexpr( Lanes == 1 ) {
            // the pointers do not alias, so that the loops are vectorized
            Real* __restrict__ sum = out;
            for( Index j = 0; j < k; j++ )
               sum[ j ] = 0;
            for( Index element = data.offsets[ row ]; element < data.offsets[ row + 1 ]; element++ ) {
               const Real* __restrict__ y = &data.Y[ data.columns[ element ] * k ];
               const Real w = getWeight( data.confidences[ element ], detail::vectorizedDot( v, y, k ), residual );
               for( Index j = 0; j < k; j++ )
                  sum[ j ] += w * y[ j ];
            }
         }
         else {
            Real sum[ maxColumnsPerLane ] = {};
            for( Index element = data.offsets[ row ]; element < data.offsets[ row + 1 ]; element++ ) {
               const Real* y = &data.Y[ data.columns[ element ] * k ];
               const Real t = group.sum( detail::partialDot( group, v, y, k ) );
               const Real w = getWeight( data.confidences[ element ], t, residual );
               for( int l = 0; l < maxColumnsPerLane; l++ ) {
                  const Index j = group.getRank() + l * Lanes;
                  if( j < k )
                     sum[ l ] += w * y[ j ];
               }
            }
            for( int l = 0; l < maxColumnsPerLane; l++ ) {
               const Index j = group.getRank() + l * Lanes;
               if( j < k )
                  out[ j ] = sum[ l ];
            }
         }
         for( Index j = group.getRank(); j < k; j += Lanes ) {
            const Real gv = detail::gramProduct< isGPU() >( data.G, v, j, k );
            out[ j ] = residual ? out[ j ] - gv : out[ j ] + gv;
            if( data.copy != nullptr )
               data.copy[ row * k + j ] = out[ j ];
         }
      }

      [[nodiscard]] __cuda_callable__
      static Real
      getWeight( Real c, Real t, bool residual )
      {
         return residual ? c - ( c - 1 ) * t : ( c - 1 ) * t;
      }

      FusedRowData data;
   };

   //! \brief Returns whether the fused products (\ref FusedRow) support the number of factors.
   [[nodiscard]] bool
   isFusedSupported() const
   {
      return ! isGPU() || factors <= 32 * FusedRow< 32 >::maxColumnsPerLane;
   }

   //! \brief Selects the fused products (\ref FusedRow) instead of the sampled and sparse-dense products.
   void
   setFused( bool value )
   {
      if( value && ! isFusedSupported() )
         throw std::invalid_argument( "SparseALS: the fused products do not support the number of factors" );
      fused = value;
   }

   //! \brief Calls `function` with the instance of `RowFunction` for the number of lanes of \ref forRows.
   template< template< int > class RowFunction, typename... Args >
   void
   forRowsWith( const Side& side, Args... args )
   {
      if constexpr( isGPU() )
         forRows( side, RowFunction< 32 >{ args... } );
      else
         forRows( side, RowFunction< 1 >{ args... } );
   }

   //! \brief Solves the factors `X` of the rows of `side` with the fixed factors `Y` of its columns.
   void
   solve( Side& side, DenseMatrixType& XMatrix, const DenseMatrixType& YMatrix )
   {
      computeGram( YMatrix, regularization, gram );
      Real* X = XMatrix.getValues().getData();
      const Real* Y = YMatrix.getConstView().getValues().getData();
      computeResiduals( side, X, Y );
      const Index* offsets = side.C.getSegments().getOffsets().getData();
      forRowsWith< StartRow >( side, factors, offsets, residuals.getData(), X, residualNorms.getData(), active.getData() );
      for( int step = 0; step < cgSteps; step++ ) {
         computeProducts( side, Y );
         forRowsWith< UpdateRow >(
            side,
            factors,
            products.getData(),
            X,
            residuals.getData(),
            directions.getData(),
            residualNorms.getData(),
            active.getData() );
      }
   }

   //! \brief Performs one iteration: solves the factors of the users and then the factors of the items.
   void
   iterate()
   {
      solve( users, userFactors, itemFactors );
      solve( items, itemFactors, userFactors );
   }

   /**
    * \brief Selects the configurations of the products of both sides by measuring them on GPUs.
    *
    * The products with the numbers of lanes and columns per lane from \ref laneConfigurations, which compute from a
    * quarter of the factors to all factors at once, are compared with the default configuration of the product.
    */
   void
   tuneProducts( int loops = 3 )
   {
      if constexpr( isGPU() ) {
         // the products are measured with the current factors, which are not changed
         Index columns = 1;
         while( columns < factors )
            columns *= 2;
         for( const bool itemsSide : { false, true } ) {
            Side& side = itemsSide ? items : users;
            const DenseMatrixType& X = itemsSide ? itemFactors : userFactors;
            const DenseMatrixType& Y = itemsSide ? userFactors : itemFactors;
            computeGram( Y, regularization, gram );
            const Real* XData = X.getConstView().getValues().getData();
            const Real* YData = Y.getConstView().getValues().getData();
            computeResiduals( side, XData, YData );
            auto measure = [ & ]()
            {
               computeProducts( side, YData );
               Timer timer;
               timer.start();
               for( int i = 0; i < loops; i++ )
                  computeProducts( side, YData );
               timer.stop();
               return timer.getRealTime();
            };
            side.configuration = {};
            ProductConfiguration best;
            double bestTime = measure();
            for( const auto& [ lanes, columnsPerLane ] : laneConfigurations ) {
               if( lanes * columnsPerLane > columns || 4 * lanes * columnsPerLane < columns )
                  continue;
               side.configuration = { lanes, columnsPerLane };
               const double time = measure();
               if( time < bestTime ) {
                  bestTime = time;
                  best = side.configuration;
               }
            }
            side.configuration = best;
         }
      }
   }

   /**
    * \brief Returns the loss divided by the sum of the confidences of all pairs of users and items, the same as
    * `calculate_training_loss` of implicit.
    *
    * The sum over all pairs is computed as the sum over the elements of `R` and `sum_u x_u^T Y^T Y x_u`, which is the
    * Frobenius product of `X^T X` and `Y^T Y`.
    */
   [[nodiscard]] double
   getLoss()
   {
      const Real* X = userFactors.getValues().getData();
      const Real* Y = itemFactors.getConstView().getValues().getData();
      sampledProduct( users, X, Y, Weight::Loss );
      const double elementsLoss = TNL::sum( users.W.getValues() );
      VectorType userGram;
      VectorType itemGram;
      computeGram( userFactors, 0, userGram );
      computeGram( itemFactors, 0, itemGram );
      const double allPairs = TNL::sum( userGram * itemGram );
      double norms = 0;
      const Index k = factors;
      for( Index j = 0; j < k; j++ )
         norms += userGram.getElement( j * k + j ) + itemGram.getElement( j * k + j );
      const double confidences = TNL::sum( users.C.getValues() );
      const double rows = users.C.getRows();
      const double columns = users.C.getColumns();
      const double elements = users.C.getValues().getSize();
      return ( elementsLoss + allPairs + regularization * norms ) / ( confidences + rows * columns - elements );
   }

protected:
   Index factors;
   Real regularization;
   int cgSteps;
   //! Whether the products are computed by \ref FusedRow.
   bool fused = false;

   Side users;
   Side items;

   DenseMatrixType userFactors;
   DenseMatrixType itemFactors;

   //! The vectors of the conjugate gradient method of all rows of one side (`rows x k`, row-major).
   VectorType residuals;
   VectorType directions;
   VectorType products;
   //! The squared norms of the residuals.
   VectorType residualNorms;
   //! Whether the conjugate gradient method continues for the row.
   FlagsType active;

   //! `Y^T Y + lambda I` of the current side.
   VectorType gram;
   //! The Gram matrices of the chunks of the rows, see \ref computeGram.
   VectorType chunkGrams;
};

}  // namespace TNL::Benchmarks::ALS
