// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "TNL/Matrices/Factorization/QR/GramSchmidt.h"
#include "TNL/Matrices/Factorization/QR/Givens.h"
#include "TNL/Matrices/Factorization/QR/Householder.h"
#include <utility>

namespace TNL::Matrices::Factorization::QR {

enum QRfactorizationType
{
   GramSchmidtType,
   GivensType,
   HouseholderType
};

/**
 * \brief Performs QR factorization on a matrix A using a chosen method (Gram-Schmidt, Givens, or Householder),
 * producing an orthogonal matrix Q and an upper triangular matrix R, such that A = QR.
 *
 * QR factorization is a matrix decomposition technique that decomposes a matrix A into a product of an orthogonal matrix Q
 * and an upper triangular matrix R. This function supports three methods for QR factorization: Gram-Schmidt process,
 * Givens rotations, and Householder reflections. The choice among these methods influences the numerical stability and
 * computational efficiency of the operation. The resulting matrices Q and R satisfy the equation A = QR, where A is the
 * original matrix, Q is orthogonal, and R is upper triangular.
 *
 * \tparam MatrixType The type of the input matrix A, which must be compatible with the operations required by the
 * selected factorization method.
 *
 * \param A The matrix to be factorized. It is a constant reference, ensuring that A remains unchanged by the function.
 * \param Q Reference to a matrix where the orthogonal matrix Q from the QR factorization will be stored.
 * \param R Reference to a matrix where the upper triangular matrix R from the QR factorization will be stored.
 * \param QRtype An enumeration value specifying the method for QR factorization. Options include GramSchmidtType,
 * GivensType, and HouseholderType, each affecting the factorization's efficiency and stability.
 *
 * \note The function does not modify the input matrix A. The matrices Q and R are output parameters that contain the
 * results of the QR factorization.
 * \note The selection of the QR factorization method (QRtype) significantly affects the process's numerical stability
 * and computational requirements. Each method has its own advantages and is suited to different types of matrices and
 * computational contexts.
 *
 * Overload:
 * A convenience overload of the QRfactorization function is provided, which takes the matrix A and the factorization
 * method type as inputs and returns a pair of matrices (Q, R) as the result. This overload simplifies cases where
 * direct access to the Q and R matrices post-factorization is desired without the need to declare them beforehand.
 *
 * \param A The matrix to be factorized, remains unchanged.
 * \param QRtype Specifies the QR factorization method to be used.
 *
 * \return A std::pair containing the orthogonal matrix Q and the upper triangular matrix R resulting from the QR
 * factorization of A.
 */
template< typename MatrixType >
void
QRfactorization( const MatrixType& A, MatrixType& Q, MatrixType& R, const QRfactorizationType& QRtype )
{
   if constexpr( MatrixType::getOrganization() == Algorithms::Segments::ColumnMajorOrder ) {
      switch( QRtype ) {
         case QRfactorizationType::GramSchmidtType:
            TNL::Matrices::Factorization::QR::GramSchmidt( A, Q, R );
            break;
         case QRfactorizationType::GivensType:
            TNL::Matrices::Factorization::QR::Givens( A, Q, R );
            break;
         case QRfactorizationType::HouseholderType:
            TNL::Matrices::Factorization::QR::Householder( A, Q, R );
            break;
         default:
            throw std::invalid_argument( "Wrong factorization type." );
            break;
      }
   }
   else {
      switch( QRtype ) {
         case QRfactorizationType::GivensType:
            TNL::Matrices::Factorization::QR::Givens( A, Q, R );
            break;
         default:
            throw std::invalid_argument( "Wrong factorization type." );
            break;
      }
   }
}

template< typename MatrixType >
std::pair< MatrixType, MatrixType >
QRfactorization( const MatrixType& A, const QRfactorizationType& QRtype )
{
   MatrixType Q;
   MatrixType R;
   QRfactorization( A, Q, R, QRtype );
   return { Q, R };
}

}  //namespace TNL::Matrices::Factorization::QR
