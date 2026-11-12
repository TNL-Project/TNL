// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>

#include <TNL/Algorithms/parallelFor.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Containers/VectorView.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>
#include <TNL/Matrices/DenseMatrix.h>
#include <TNL/Matrices/SparseDenseOperations.h>
#include <TNL/Matrices/SparseMatrix.h>

namespace TNL::Benchmarks::SVM {

//! \brief The methods of the computation of the gradient, see \ref LinearSVMGradient.
enum class Method : std::uint8_t
{
   //! Every classifier is processed separately: `K` sparse matrix-vector products `X w_c`, the derivatives of the loss,
   //! and `K` products `X^T g_c`.
   SpMV,
   //! All classifiers at once: the sparse-dense products `S = X W` and `X^T G` and separate kernels for the derivatives
   //! of the loss and the regularization.
   SpMM,
   //! The same as \ref SpMM, the storer of the first product evaluates the derivatives of the loss and the storer of the
   //! second product adds the regularization, so the scores `S` are not stored.
   Fused
};

/**
 * \brief The derivative of the squared hinge loss `cost * max(0, 1 - y s)^2` with respect to the score `s` of the
 * sample `i` and the class `c`, where `y = 1` for the class of the sample and `y = -1` otherwise (one-vs-rest).
 */
template< typename Real, typename Index >
struct SquaredHingeDerivative
{
   [[nodiscard]] __cuda_callable__
   Real
   operator()( Index i, Index c, Real s ) const
   {
      const Real y = labels[ i ] == c ? 1 : -1;
      const Real margin = 1 - y * s;
      return margin > 0 ? -2 * cost * y * margin : Real( 0 );
   }

   const Index* labels;
   Real cost;
};

/**
 * \brief The gradient of the primal problem of the multiclass linear SVM with the squared hinge loss (L2-loss SVM,
 * one-vs-rest, the default multiclass solver of LIBLINEAR).
 *
 * The samples are the rows of the sparse matrix `X` (`n x d`), their classes are `labels`. The weights of the `K`
 * classifiers are the columns of `W` (`d x K`). The objective is
 *
 *    sum_c ( |w_c|^2 / 2 + cost sum_i max(0, 1 - y_ic x_i . w_c)^2 ),
 *
 * where `y_ic = 1` when the sample `i` belongs to the class `c` and `-1` otherwise, and its gradient is
 *
 *    W + X^T G,   G_ic = -2 cost y_ic max(0, 1 - y_ic s_ic),   S = X W.
 *
 * The products with `X^T` use the explicitly transposed matrix, so that both products are reductions in the rows of a
 * sparse matrix. See \ref Method for the methods of the computation.
 *
 * The methods with lambda functions are public, because nvcc does not allow extended lambdas in private methods.
 */
template< typename Real, typename Device, typename Index >
class LinearSVMGradient
{
public:
   using MatrixType = Matrices::SparseMatrix< Real, Device, Index >;
   using VectorType = Containers::Vector< Real, Device, Index >;
   using IndexVectorType = Containers::Vector< Index, Device, Index >;
   using DenseMatrixType = Matrices::DenseMatrix< Real, Device, Index, Algorithms::Segments::RowMajorOrder >;

   LinearSVMGradient( const MatrixType& X, const IndexVectorType& labels, Index classes, Real cost = 1 )
   : X( X ),
     labels( labels ),
     classes( classes ),
     cost( cost )
   {
      if( labels.getSize() != X.getRows() )
         throw std::invalid_argument( "LinearSVMGradient: wrong number of labels" );
      XT.getTransposition( X );
      const Index n = X.getRows();
      const Index d = X.getColumns();
      scores.setDimensions( n, classes );
      derivatives.setDimensions( n, classes );
      gradient.setDimensions( d, classes );
      weights.setDimensions( d, classes );
      scoreColumns.setSize( n * classes );
      derivativeColumns.setSize( n * classes );
      gradientColumns.setSize( d * classes );
      weightColumns.setSize( d * classes );
   }

   [[nodiscard]] Index
   getClasses() const
   {
      return classes;
   }

   //! \brief Sets the weights `W` (`d x K`, row-major), they are also copied by columns for \ref Method::SpMV.
   void
   setWeights( const DenseMatrixType& W )
   {
      weights = W;
      const Index d = X.getColumns();
      const Index K = classes;
      const auto w = weights.getConstView().getValues().getConstView();
      auto columns = weightColumns.getView();
      Algorithms::parallelFor< Device >(
         0,
         d * K,
         [ = ] __cuda_callable__( Index index ) mutable
         {
            const Index feature = index / K;
            const Index c = index % K;
            columns[ c * d + feature ] = w[ index ];
         } );
   }

   [[nodiscard]] SquaredHingeDerivative< Real, Index >
   getDerivative() const
   {
      return { labels.getData(), cost };
   }

   /**
    * \brief Computes the derivatives of the loss `G` from the scores `S`, both `n x K` stored by rows (`byColumns` is
    * false) or by columns.
    */
   void
   computeDerivatives( const Real* S, Real* G, bool byColumns )
   {
      const Index n = X.getRows();
      const Index K = classes;
      const auto derivative = getDerivative();
      Algorithms::parallelFor< Device >(
         0,
         n * K,
         [ = ] __cuda_callable__( Index index ) mutable
         {
            const Index i = byColumns ? index % n : index / K;
            const Index c = byColumns ? index / n : index % K;
            G[ index ] = derivative( i, c, S[ index ] );
         } );
   }

   //! \brief Computes the gradient by \ref Method::SpMV.
   void
   computeBySpMV()
   {
      const Index n = X.getRows();
      const Index d = X.getColumns();
      for( Index c = 0; c < classes; c++ ) {
         auto s = scoreColumns.getView( c * n, ( c + 1 ) * n );
         X.vectorProduct( weightColumns.getConstView( c * d, ( c + 1 ) * d ), s );
      }
      computeDerivatives( scoreColumns.getData(), derivativeColumns.getData(), true );
      for( Index c = 0; c < classes; c++ ) {
         auto g = gradientColumns.getView( c * d, ( c + 1 ) * d );
         XT.vectorProduct( derivativeColumns.getConstView( c * n, ( c + 1 ) * n ), g );
      }
      gradientColumns += weightColumns;
   }

   //! \brief Computes the gradient by \ref Method::SpMM.
   void
   computeBySpMM()
   {
      Matrices::sparseDenseMatrixProduct( X, weights, scores );
      computeDerivatives( scores.getValues().getData(), derivatives.getValues().getData(), false );
      Matrices::sparseDenseMatrixProduct( XT, derivatives, gradient );
      gradient.getValues() += weights.getValues();
   }

   //! \brief Computes the gradient by \ref Method::Fused.
   void
   computeFused()
   {
      const Index K = classes;
      const auto derivative = getDerivative();
      auto G = derivatives.getValues().getView();
      auto storeDerivative = [ = ] __cuda_callable__( Index i, int c, const Real& s ) mutable
      {
         G[ i * K + c ] = derivative( i, c, s );
      };
      Matrices::sparseDenseMatrixProductWithLambdas( X, K, weights, storeDerivative );
      const auto W = weights.getConstView().getValues().getConstView();
      auto gradientView = gradient.getValues().getView();
      auto storeGradient = [ = ] __cuda_callable__( Index feature, int c, const Real& value ) mutable
      {
         gradientView[ feature * K + c ] = value + W[ feature * K + c ];
      };
      Matrices::sparseDenseMatrixProductWithLambdas( XT, K, derivatives, storeGradient );
   }

   //! \brief Computes the gradient by the given method.
   void
   compute( Method method )
   {
      if( method == Method::SpMV )
         computeBySpMV();
      else if( method == Method::SpMM )
         computeBySpMM();
      else
         computeFused();
   }

   //! \brief Returns the Frobenius norm of the gradient computed by `method`.
   [[nodiscard]] double
   getGradientNorm( Method method ) const
   {
      if( method == Method::SpMV )
         return l2Norm( gradientColumns );
      return l2Norm( gradient.getValues() );
   }

protected:
   const MatrixType& X;
   MatrixType XT;
   IndexVectorType labels;
   Index classes;
   Real cost;

   //! The matrices of \ref Method::SpMM and \ref Method::Fused (stored by rows).
   DenseMatrixType weights;
   DenseMatrixType scores;
   DenseMatrixType derivatives;
   DenseMatrixType gradient;

   //! The matrices of \ref Method::SpMV stored by columns, so that the columns are vectors.
   VectorType weightColumns;
   VectorType scoreColumns;
   VectorType derivativeColumns;
   VectorType gradientColumns;
};

}  // namespace TNL::Benchmarks::SVM
