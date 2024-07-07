// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <tuple>
#include <utility>

#include "TNL/Containers/Expressions/ExpressionTemplates.h"
#include "TNL/Matrices/Factorization/QR/QR.h"
#include "TNL/Math.h"

namespace TNL::Matrices::Eigen {

/**
 * \brief Computes the eigenvalues and eigenvectors of a matrix using the QR iteration algorithm with a specified QR
 * factorization method.
 *
 * QR iteration is a sophisticated numerical technique employed to find the eigenvalues and eigenvectors of a matrix. This
 * method involves the repeated application of QR factorization to the matrix, followed by the recombination of the factorized
 * matrices in reverse order (RQ instead of QR). Through successive iterations, the matrix transforms into an upper triangular
 * matrix whose diagonal elements are the eigenvalues of the original matrix. Simultaneously, the product of Q matrices across
 * iterations converges to the eigenvector matrix of the original matrix.
 *
 * \tparam T Data type of the matrix elements (e.g., float, double), influencing the precision of computations.
 * \tparam Device Computational device (e.g., CPU, GPU) for data storage and operations.
 * \tparam MatrixType Type of matrix (e.g., dense, sparse) involved in the computation.
 *
 * \param matrix The square matrix for which to compute eigenvalues and eigenvectors. Non-square matrices are not supported.
 * \param epsilon The convergence threshold for the algorithm. The iteration process is considered complete when all
 * off-diagonal elements in the current matrix are smaller than this value.
 * \param QRtype Enumerated type specifying the QR factorization method to use (e.g., Gram-Schmidt, Givens rotations,
 * Householder reflections), impacting the algorithm's efficiency and numerical stability.
 * \param maxIterations (Optional) The maximum number of iterations to perform, defaulting to 10000. If this limit is
 * reached before convergence, the function terminates, returning the last computed matrices and an iteration count of -1.
 *
 * \return A tuple comprising three elements:
 *         - The matrix converged to an upper triangular form, with eigenvalues on the diagonal (of type MatrixType).
 *         - The matrix of eigenvectors corresponding to the eigenvalues (of type MatrixType).
 *         - The number of iterations conducted (of type int), where -1 indicates that the iteration limit was reached
 *           without achieving convergence.
 *
 * \exception std::logic_error Thrown if the matrix is not square, as QR iteration requires a square matrix to function
 * correctly.
 *
 * \note This algorithm assumes that the input matrix has distinct eigenvalues for effective convergence. Matrices that are
 * already close to upper triangular form may see faster convergence.
 * \note The specified precision (epsilon) controls the accuracy of the computed eigenvalues and eigenvectors. A smaller epsilon
 * results in higher precision but may necessitate a greater number of iterations to achieve convergence.
 */
template< typename Real, typename Device, typename MatrixType >
std::tuple< MatrixType, MatrixType, int >
QRalgorithm( MatrixType matrix,
             const Real& epsilon,
             const TNL::Matrices::Factorization::QR::QRfactorizationType& QRtype,
             const int& maxIterations = 10000 )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Power iteration is possible only for square matrices" );
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   using IndexType = typename MatrixType::IndexType;
   IndexType size = matrix.getColumns();
   MatrixType Q( size, size );
   MatrixType R( size, size );
   MatrixType accQ( size, size );
   int iterations = 0;
   for( int i = 0; i < size; i++ )
      accQ.setElement( i, i, 1 );
   while( true ) {
      TNL::Matrices::Factorization::QR::QRfactorization( matrix, Q, R, QRtype );
      matrix.getMatrixProduct( R, Q );
      MatrixType Q_acc_temp( size, size );
      Q_acc_temp.getMatrixProduct( accQ, Q );
      accQ = std::move( Q_acc_temp );
      bool converged = true;
      iterations++;
      for( IndexType i = 0; i < size - 1; i++ ) {
         if( isnan( matrix.getElement( i + 1, i ) ) ) {
            return std::make_tuple( matrix, accQ, -1 );
         }
         if( abs( matrix.getElement( i + 1, i ) ) >= epsilon ) {
            converged = false;
            break;
         }
      }

      if( converged ) {
         return std::make_tuple( matrix, accQ, iterations );
      }
      if( iterations == maxIterations ) {
         return std::make_tuple( matrix, accQ, 0 );
      }
   }
   return std::make_tuple( matrix, accQ, iterations );
}

}  //namespace TNL::Matrices::Eigen
