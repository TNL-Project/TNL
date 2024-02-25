// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once


#include <tuple>

#include "TNL/Containers/Expressions/ExpressionTemplates.h"
#include "TNL/Backend/DeviceInfo.h"
#include "TNL/Math.h"
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/fillRandom.h>

namespace TNL::Matrices::Eigen {

/**
 * \brief Calculates the largest eigenvalue and its corresponding eigenvector of a given matrix using the power iteration method.
 *
 * This function employs the power iteration algorithm, which is an eigenvalue algorithm: given a square matrix, it produces a
 * number (the eigenvalue) and a non-zero vector (the eigenvector) such that multiplying the matrix by the vector results in a
 * scalar multiple of the vector. The algorithm is typically used to find the largest eigenvalue and the corresponding
 * eigenvector. It is an iterative method, starting with an initial vector and repeatedly applying the matrix to this vector,
 * then normalizing the result, until convergence is achieved.
 *
 * \tparam T is the data type of the matrix elements (e.g., float, double), which determines the precision of the computations.
 * \tparam Device is the computational device where the data is stored and operations are performed (e.g., CPU, GPU).
 * \tparam MatrixType is the type of the matrix (e.g., dense matrix, sparse matrix).
 *
 * \param matrix is the matrix for which the largest eigenvalue and associated eigenvector are to be calculated. The matrix should
 * be square.
 * \param epsilon is the precision of the calculation, which determines the convergence threshold for the iterative method.
 *
 * \return A tuple containing three elements:
 *         1. The largest eigenvalue (of type T).
 *         2. The eigenvector associated with the largest eigenvalue (of type TNL::Containers::Vector< T, Device >).
 *         3. The number of iterations (of type uint) that were performed to reach convergence.
 *
 * \note This method converges under certain conditions (e.g., the matrix should have a dominant eigenvalue). It may not be
 * suitable for all matrices, especially those where the largest eigenvalue is not well-separated from the others.
 * \note The initial vector for the power iteration is chosen randomly. Different initial vectors can lead to different
 * convergence rates.
 */
template< typename T, typename Device, typename MatrixType >
static std::tuple< T, TNL::Containers::Vector< T, Device >, uint >
powerIterationTuple( const MatrixType& matrix, const T& epsilon )
{
   using IndexType = typename MatrixType::IndexType;
   IndexType vecSize = matrix.getColumns();
   TNL::Containers::Vector< T, Device > eigenVec( vecSize );
   TNL::Containers::Vector< T, Device > eigenVecOut( vecSize );
   if constexpr( std::is_integral_v< T > ) {
      TNL::Algorithms::fillRandom< Device >( eigenVec.getData(), vecSize, (T) -10000, (T) 10000 );
   }
   else {
      TNL::Algorithms::fillRandom< Device >( eigenVec.getData(), vecSize, (T) -1, (T) 1 );
   }
   T norm = 0;
   T normOld = 0;
   uint iterations = 0;
   while( true ) {
      matrix.vectorProduct( eigenVec, eigenVecOut );
      auto [ v, i ] = TNL::argMax( abs( eigenVecOut ) );
      norm = eigenVecOut.getElement( i );
      eigenVec = eigenVecOut / norm;
      if( fabs( norm - normOld ) < epsilon )
         break;
      normOld = norm;
      iterations++;
   }
   return std::make_tuple( norm, eigenVec, iterations );
}

/**
 * \brief Calculates the largest eigenvalue and its corresponding eigenvector of a given matrix using the power iteration method.
 *
 * This function employs the power iteration algorithm, which is an eigenvalue algorithm: given a square matrix, it produces a
 * number (the eigenvalue) and a non-zero vector (the eigenvector) such that multiplying the matrix by the vector results in a
 * scalar multiple of the vector. The algorithm is typically used to find the largest eigenvalue and the corresponding
 * eigenvector. It is an iterative method, starting with an initial vector and repeatedly applying the matrix to this vector,
 * then normalizing the result, until convergence is achieved.
 *
 * \tparam T is the data type of the matrix elements (e.g., float, double), which determines the precision of the computations.
 * \tparam Device is the computational device where the data is stored and operations are performed (e.g., CPU, GPU).
 * \tparam MatrixType is the type of the matrix (e.g., dense matrix, sparse matrix).
 *
 * \param matrix is the matrix for which the largest eigenvalue and associated eigenvector are to be calculated. The matrix should
 * be square.
 * \param epsilon is the precision of the calculation, which determines the convergence threshold for the iterative method.
 *
 * \return A pair consisting of:
 *         1. The largest eigenvalue (of type T).
 *         2. The eigenvector associated with the largest eigenvalue (of type TNL::Containers::Vector< T, Device >).
 *
 * \note This method converges under certain conditions (e.g., the matrix should have a dominant eigenvalue). It may not be
 * suitable for all matrices, especially those where the largest eigenvalue is not well-separated from the others.
 * \note The initial vector for the power iteration is chosen randomly. Different initial vectors can lead to different
 * convergence rates.
 */
template< typename T, typename Device, typename MatrixType >
static std::pair< T, TNL::Containers::Vector< T, Device > >
powerIteration( const MatrixType& matrix, const T& epsilon )
{
   std::tuple< T, TNL::Containers::Vector< T, Device >, uint > tuple = powerIterationTuple< T, Device >( matrix, epsilon );
   return std::make_pair( std::get< 0 >( tuple ), std::get< 1 >( tuple ) );
}

}  //namespace TNL::Matrices::Eigen
