// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <tuple>
#include <utility>

#include "TNL/Containers/Expressions/ExpressionTemplates.h"
#include "TNL/Matrices/Factorization/QR/QR.h"
#include "TNL/Backend/DeviceInfo.h"
#include "TNL/Math.h"

namespace TNL::Matrices::Eigen {

/**
 * \brief Performs QR iteration on a given matrix to compute its eigenvalues and eigenvectors, using a specified QR factorization method.
 *
 * This function implements the QR iteration algorithm, a numerical method used to compute the eigenvalues and eigenvectors of a matrix.
 * It iteratively applies QR factorization, followed by the multiplication of the resulting R and Q matrices in reverse order. The process
 * is repeated until the matrix converges to an upper triangular form, where the diagonal elements approximate the eigenvalues of the
 * original matrix. The accumulated product of Q matrices converges to the matrix of eigenvectors.
 *
 * \tparam T The data type of the matrix elements (e.g., float, double), which determines the precision of the computations.
 * \tparam Device The computational device where the data is stored and operations are performed (e.g., Host, GPU).
 * \tparam MatrixType The type of the matrix (e.g., dense matrix, sparse matrix), which must support basic matrix operations like
 * getMatrixProduct.
 *
 * \param matrix The matrix for which the eigenvalues and eigenvectors are to be computed. It should be a square matrix.
 * \param precision The precision threshold for the convergence check. The iteration stops when all off-diagonal elements
 * of the current matrix are below this threshold.
 * \param QRtype The type of QR factorization method to be used, specified by an enumeration. This can be one of the QR
 * factorization methods like Gram-Schmidt, Givens, or Householder.
 *
 * \return A pair of matrices where the first element is the converged matrix approximating the upper triangular matrix with
 * eigenvalues on its diagonal, and the second element is the matrix of eigenvectors corresponding to these eigenvalues.
 *
 * \note The convergence of this method depends on the matrix having distinct eigenvalues. The method is more efficient
 * for matrices that are close to upper triangular form.
 * \note The precision parameter controls the accuracy of the eigenvalues and eigenvectors computed. Lower values
 * lead to more accurate results but may require more iterations.
 */
template< typename T, typename Device, typename MatrixType >
static std::pair< MatrixType, MatrixType >
QRiteration( MatrixType matrix, const T& precision, const TNL::Matrices::Factorization::QR::QRfactorizationType& QRtype )
{
   TNL_ASSERT_EQ( matrix.getRows(), matrix.getColumns(), "QR iteration is possible only for square matrices" );
   typename MatrixType::IndexType size = matrix.getColumns();
   MatrixType Q( size, size );
   MatrixType R( size, size );
   MatrixType Q_acc( size, size );
   for( int i = 0; i < size; i++ )
      Q_acc.setElement( i, i, 1 );
   while( true ) {
      TNL::Matrices::Factorization::QR::QRfactorization( matrix, Q, R, QRtype );
      matrix.getMatrixProduct( R, Q );
      MatrixType Q_acc_pom( size, size );
      Q_acc_pom.getMatrixProduct( Q_acc, Q );
      Q_acc = Q_acc_pom;
      bool converged = true;
      for( int i = 0; i < size - 1; i++ ) {
         if( abs( matrix.getElement( i + 1, i ) ) >= precision ) {
            converged = false;
            break;
         }
      }

      if( converged ) {
         break;
      }
   }
   return std::make_pair( matrix, Q_acc );
}

}  //namespace TNL::Matrices::Eigen
