// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/Eigen/PowerIteration.h>

namespace TNL::Matrices::Eigen {

/**
 * \brief Calculates an eigenvalue and its corresponding eigenvector of a matrix using the shifted power iteration method.
 *
 * The shifted power iteration method is an adaptation of the standard power iteration algorithm. It accelerates the convergence
 * of the power iteration to a specific eigenvalue that is closer to the shift value. This method is particularly useful when
 * the largest eigenvalue does not converge quickly or when interest lies in an eigenvalue that is not the largest.
 *
 * \tparam T is the data type of the matrix elements (e.g., float, double), which determines the precision of the computations.
 * \tparam Device is the computational device where the data is stored and operations are performed (e.g., CPU, GPU).
 * \tparam MatrixType is the type of the matrix (e.g., dense matrix, sparse matrix).
 *
 * \param matrix is the original matrix for which the eigenvalue and eigenvector are to be calculated.
 * \param precision is the precision of the calculation, which determines the convergence threshold for the iterative method.
 * \param shiftValue is the value used to shift the spectrum of the matrix. The choice of shiftValue influences which
 * eigenvalue the method will converge to.
 *
 * \return A tuple containing three elements:
 *         1. The eigenvalue (of type T) closest to the shiftValue.
 *         2. The corresponding eigenvector (of type TNL::Containers::Vector< T, Device >).
 *         3. The number of iterations (of type uint) that were performed to reach convergence.
 *
 * \note This method assumes that the input matrix is square and that a dominant eigenvalue exists close to the shift value.
 * \note The effectiveness of convergence depends on the choice of shiftValue. A value closer to the desired eigenvalue
 * generally leads to faster convergence.
 */
template< typename T, typename Device, typename MatrixType >
static std::tuple< T, TNL::Containers::Vector< T, Device >, uint >
powerIterationShiftTuple( const MatrixType& matrix, const T& precision, const T& shiftValue )
{
   int size = matrix.getColumns();
   MatrixType shiftedMatrix( size, size );
   for( int i = 0; i < size; i++ )
      shiftedMatrix.setElement( i, i, shiftValue );
   shiftedMatrix.addMatrix( matrix );
   std::tuple< T, TNL::Containers::Vector< T, Device >, uint > tuple =
      TNL::Matrices::Eigen::powerIterationTuple< T, Device >( shiftedMatrix, precision );
   return std::make_tuple(std::get<0>(tuple) - shiftValue, std::get<1>(tuple), std::get<2>(tuple));
}

/**
 * \brief Calculates an eigenvalue and its corresponding eigenvector of a matrix using the shifted power iteration method.
 *
 * The shifted power iteration method is an adaptation of the standard power iteration algorithm. It accelerates the convergence
 * of the power iteration to a specific eigenvalue that is closer to the shift value. This method is particularly useful when
 * the largest eigenvalue does not converge quickly or when interest lies in an eigenvalue that is not the largest.
 *
 * \tparam T is the data type of the matrix elements (e.g., float, double), which determines the precision of the computations.
 * \tparam Device is the computational device where the data is stored and operations are performed (e.g., CPU, GPU).
 * \tparam MatrixType is the type of the matrix (e.g., dense matrix, sparse matrix).
 *
 * \param matrix is the original matrix for which the eigenvalue and eigenvector are to be calculated.
 * \param precision is the precision of the calculation, which determines the convergence threshold for the iterative method.
 * \param shiftValue is the value used to shift the spectrum of the matrix. The choice of shiftValue influences which
 * eigenvalue the method will converge to.
 *
 * \return A pair consisting of:
 *         1. The eigenvalue (of type T) closest to the shiftValue.
 *         2. The corresponding eigenvector (of type TNL::Containers::Vector< T, Device >).
 *
 * \note This method assumes that the input matrix is square and that a dominant eigenvalue exists close to the shift value.
 * \note The effectiveness of convergence depends on the choice of shiftValue. A value closer to the desired eigenvalue
 * generally leads to faster convergence.
 */
template< typename T, typename Device, typename MatrixType >
static std::pair< T, TNL::Containers::Vector< T, Device > >
powerIterationShift( const MatrixType& matrix, const T& precision, const T& shiftValue )
{
   std::tuple< T, TNL::Containers::Vector< T, Device >, uint > tuple =
      powerIterationShiftTuple< T, Device >( matrix, precision, shiftValue );
   return std::make_pair( std::get< 0 >( tuple ), std::get< 1 >( tuple ) );
}

}  //namespace TNL::Matrices::Eigen
