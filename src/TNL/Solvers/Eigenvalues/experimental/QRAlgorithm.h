// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <tuple>
#include <utility>

#include <TNL/Containers/Expressions/ExpressionTemplates.h>
#include <TNL/Matrices/Factorization/QR/QR.h>
#include <TNL/Math.h>

namespace TNL::Solvers::Eigenvalues::experimental {

/**
 * \brief Computes the eigenvalues and eigenvectors of a matrix using the QR iteration algorithm with a specified QR
 * factorization method.
 *
 * This function implements the QR algorithm to find the eigenvalues and eigenvectors of a square matrix by repeatedly applying
 * QR factorization and recombining the factorized matrices. The matrix converges to an upper triangular form with eigenvalues
 * on the diagonal, and the product of Q matrices across iterations converges to the eigenvector matrix.
 *
 * \tparam MatrixType Type of matrix, which defines both the data type of the matrix elements (e.g., float, double) and
 * the computational device (e.g., CPU, GPU). The matrix must define a nested type `RealType`, representing the data type of the
 * elements.
 *
 * \param matrix The square matrix for which to compute eigenvalues and eigenvectors. Non-square matrices are not supported. The
 * data type and computational device are determined by the matrix type.
 * \param epsilon The convergence threshold for the algorithm. The iteration process is considered complete when all
 * off-diagonal elements in the current matrix are smaller than this value. This should be of type `MatrixType::RealType`.
 * \param QRMethod An enum class specifying the QR factorization method to use (e.g., Gram-Schmidt, Givens rotations,
 * Householder reflections). This choice impacts the algorithm's efficiency and numerical stability.
 * \param maxIterations (Optional) The maximum number of iterations to perform, defaulting to 10000. If this limit is reached
 * before convergence, the function terminates, returning the last computed matrices and an iteration count of -1.
 *
 * \return A tuple comprising three elements:
 *         - The matrix converged to an upper triangular form, with eigenvalues on the diagonal (of type `MatrixType`).
 *         - The matrix of eigenvectors corresponding to the eigenvalues (of type `MatrixType`).
 *         - The number of iterations conducted (of type `int`), where -1 indicates that the iteration limit was reached
 *           without achieving convergence.
 *
 * \exception std::invalid_argument Thrown if the matrix is not square or is zero-sized.
 */
template< typename MatrixType >
std::tuple< MatrixType, MatrixType, int >
QRAlgorithm( MatrixType matrix,
             const typename MatrixType::RealType& epsilon,
             const TNL::Matrices::Factorization::QR::FactorizationMethod& QRMethod,
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
   if( size == 1 )
      return std::make_tuple( matrix, accQ, 1 );
   while( true ) {
      TNL::Matrices::Factorization::QR::QRFactorization( matrix, Q, R, QRMethod );
      matrix.getMatrixProduct( R, Q );
      MatrixType Q_acc_temp( size, size );
      Q_acc_temp.getMatrixProduct( accQ, Q );
      accQ = std::move( Q_acc_temp );
      bool converged = true;
      iterations++;
      for( IndexType i = 0; i < size - 1; i++ ) {
         if( std::isnan( matrix.getElement( i + 1, i ) ) ) {
            return std::make_tuple( matrix, accQ, -1 );
         }
         if( std::abs( matrix.getElement( i + 1, i ) ) >= epsilon ) {
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
   return std::make_tuple( std::move( matrix ), std::move( accQ ), iterations );
}

}  // namespace TNL::Solvers::Eigenvalues::experimental
