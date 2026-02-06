// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>
#include <utility>

#include <TNL/Matrices/Factorization/QR/GramSchmidt.h>
#include <TNL/Matrices/Factorization/QR/Givens.h>
#include <TNL/Matrices/Factorization/QR/Householder.h>

namespace TNL::Matrices::Factorization::QR {

enum class FactorizationMethod : std::uint8_t
{
   GramSchmidt,
   Givens,
   Householder
};

/**
 * \brief Performs QR factorization on a matrix A using a specified method (Gram-Schmidt, Givens, or Householder),
 * producing an orthogonal matrix Q and an upper triangular matrix R, such that A = QR.
 *
 * \tparam MatrixType The type of the input matrix A.
 *
 * \param A The matrix to be factorized, remains unchanged.
 * \param Q Reference to a matrix where the orthogonal matrix Q will be stored.
 * \param R Reference to a matrix where the upper triangular matrix R will be stored.
 * \param QRMethod The method for QR factorization: GramSchmidt, Givens, or Householder.
 *
 * \exception std::invalid_argument Thrown if an incorrect QR factorization type is provided.
 */
template< typename MatrixType >
void
QRFactorization( const MatrixType& A, MatrixType& Q, MatrixType& R, const FactorizationMethod& QRMethod )
{
   if constexpr( MatrixType::getOrganization() == Algorithms::Segments::ColumnMajorOrder ) {
      switch( QRMethod ) {
         case FactorizationMethod::GramSchmidt:
            TNL::Matrices::Factorization::QR::GramSchmidt( A, Q, R );
            break;
         case FactorizationMethod::Givens:
            TNL::Matrices::Factorization::QR::Givens( A, Q, R );
            break;
         case FactorizationMethod::Householder:
            TNL::Matrices::Factorization::QR::Householder( A, Q, R );
            break;
         default:
            throw std::invalid_argument( "Wrong QR factorization type for dense matrix with column-major order organization." );
      }
   }
   else {
      switch( QRMethod ) {
         case FactorizationMethod::Givens:
            TNL::Matrices::Factorization::QR::Givens( A, Q, R );
            break;
         default:
            throw std::invalid_argument( "Wrong QR factorization type for dense matrix with row-major order organization." );
      }
   }
}

/**
 * \brief Overload for QR factorization that returns a pair of matrices (Q, R).
 *
 * \tparam MatrixType The type of the input matrix A.
 *
 * \param A The matrix to be factorized.
 * \param QRMethod The method for QR factorization: GramSchmidt, Givens, or Householder.
 *
 * \return A std::pair containing the orthogonal matrix Q (of type `MatrixType`) and the upper triangular matrix R (of type
 * `MatrixType`).
 */
template< typename MatrixType >
std::pair< MatrixType, MatrixType >
QRFactorization( const MatrixType& A, const FactorizationMethod& QRMethod )
{
   MatrixType Q;
   MatrixType R;
   QRFactorization( A, Q, R, QRMethod );
   return { Q, R };
}

}  // namespace TNL::Matrices::Factorization::QR
