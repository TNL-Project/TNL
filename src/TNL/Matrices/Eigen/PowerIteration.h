// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <utility>

#include <TNL/Math.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/fillRandom.h>

namespace TNL::Matrices::Eigen {

/**
 * \brief Calculates the largest eigenvalue and its corresponding eigenvector of a given matrix using the power iteration
 * method.
 *
 * This function implements the power iteration algorithm to find the largest eigenvalue and its associated eigenvector for a
 * square matrix. The process involves repeatedly applying the matrix to an initial vector, normalizing the resulting vector,
 * and checking for convergence against a specified precision threshold.
 *
 * \tparam Real Data type of the matrix elements (e.g., float, double).
 * \tparam Device Computational device (e.g., CPU, GPU) where data is stored and operations are executed.
 * \tparam MatrixType Type of the matrix (e.g., dense matrix, sparse matrix) used in the computation.
 *
 * \param matrix The square matrix for which to calculate the largest eigenvalue and associated eigenvector.
 * \param epsilon Precision threshold for convergence. The iterative process halts when the difference between successive
 * iterations falls below this value.
 * \param initialVec (Overload) The initial vector to start the iteration with. This vector
 * must have the same size as the matrix dimensions and should not be the zero vector.
 * \param maxIterations (Optional) Maximum
 * number of iterations to perform. Defaults to 100000. If this limit is reached before convergence, the function returns the
 * last computed values with the iteration count set to 0.
 *
 * \return A tuple containing:
 *         - The largest eigenvalue of the matrix (of type Real).
 *         - The eigenvector associated with the largest eigenvalue (of type `TNL::Containers::Vector<T, Device>`).
 *         - The number of iterations performed to reach convergence (of type int). A value of 0 indicates non-convergence
 * within the specified maximum number of iterations, and -1 indicates a computational error resulting in NaN.
 *
 * \exception std::invalid_argument Thrown if the input matrix is not square, is zero-sized, or if the initial vector's size
 * does not match the matrix dimensions.
 */
template< typename Real, typename Device, typename MatrixType >
std::tuple< Real, TNL::Containers::Vector< Real, Device >, int >
powerIteration( const MatrixType& matrix,
                const Real& epsilon,
                TNL::Containers::Vector< Real, Device >& initialVec,
                const int& maxIterations = 100000 )
{
   static_assert( std::is_same_v< Device, typename MatrixType::DeviceType > );
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Power iteration is possible only for square matrices" );
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   if( matrix.getRows() != initialVec.getSize() )
      throw std::invalid_argument( "The initial vector must have the same size as the matrix" );
   using IndexType = typename MatrixType::IndexType;
   IndexType vecSize = matrix.getColumns();
   TNL::Containers::Vector< Real, Device > eigenVecOut( vecSize );
   eigenVecOut.setValue( 0 );
   Real norm = 0;
   Real normOld = 0;
   int iterations = 0;
   TNL::Containers::Vector< Real, Device > eigenVecOld( vecSize );
   eigenVecOld.setValue( 0 );
   norm = TNL::l2Norm( initialVec );
   if( norm == 0 )
      throw std::invalid_argument( "The initial vector must be nonzero" );
   if( norm != 1 )
      initialVec = initialVec / norm;
   while( true ) {
      matrix.vectorProduct( initialVec, eigenVecOut );
      norm = TNL::l2Norm( eigenVecOut );
      if( std::isnan( norm ) )
         return std::make_tuple( norm, initialVec, -1 );
      initialVec = std::move( eigenVecOut / norm );
      iterations++;
      if( TNL::abs( normOld - norm ) < epsilon ) {
         if( TNL::all( TNL::less( TNL::abs( initialVec - eigenVecOld ), epsilon ) ) )
            return std::make_tuple( norm, initialVec, iterations );
         if( TNL::all( TNL::less( TNL::abs( initialVec + eigenVecOld ), epsilon ) ) )
            return std::make_tuple( -norm, initialVec, iterations );
      }
      if( iterations == maxIterations )
         return std::make_tuple( norm, initialVec, 0 );
      eigenVecOld = initialVec;
      normOld = norm;
   }
   iterations = -1;
   return std::make_tuple( norm, initialVec, iterations );
}

/**
 * \brief Calculates the largest eigenvalue and its corresponding eigenvector of a given matrix using the power iteration
 * method.
 *
 * This function implements the power iteration algorithm to find the largest eigenvalue and its associated eigenvector for a
 * square matrix. The process involves repeatedly applying the matrix to an initial vector, normalizing the resulting vector,
 * and checking for convergence against a specified precision threshold.
 *
 * \tparam Real Data type of the matrix elements (e.g., float, double).
 * \tparam Device Computational device (e.g., CPU, GPU) where data is stored and operations are executed.
 * \tparam MatrixType Type of the matrix (e.g., dense matrix, sparse matrix) used in the computation.
 *
 * \param matrix The square matrix for which to calculate the largest eigenvalue and associated eigenvector.
 * \param epsilon Precision threshold for convergence. The iterative process halts when the difference between successive
 * iterations falls below this value.
 * \param maxIterations (Optional) Maximum number of iterations to perform. Defaults to 100000. If this limit is reached before
 * convergence, the function returns the last computed values with the iteration count set to 0.
 *
 * \return A tuple containing:
 *         - The largest eigenvalue of the matrix (of type Real).
 *         - The eigenvector associated with the largest eigenvalue (of type `TNL::Containers::Vector<T, Device>`).
 *         - The number of iterations performed to reach convergence (of type int). A value of 0 indicates non-convergence
 * within the specified maximum number of iterations, and -1 indicates a computational error resulting in NaN.
 *
 * \exception std::invalid_argument Thrown if the input matrix is not square, is zero-sized, or if the initial vector size does
 * not match the matrix dimensions.
 *
 * \note Without an initial vector argument, the function generates a random initial vector
 * appropriate for the matrix size and type. The nature of the initial vector can affect the convergence speed of the algorithm.
 */
template< typename Real, typename Device, typename MatrixType >
std::tuple< Real, TNL::Containers::Vector< Real, Device >, int >
powerIteration( const MatrixType& matrix, const Real& epsilon, const int& maxIterations = 100000 )
{
   static_assert( std::is_same_v< Device, typename MatrixType::DeviceType > );
   using IndexType = typename MatrixType::IndexType;
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   IndexType vecSize = matrix.getRows();
   TNL::Containers::Vector< Real, Device > initialVec( vecSize );
   initialVec.resize( vecSize );
   do {
      if constexpr( std::is_integral_v< Real > ) {
         TNL::Algorithms::fillRandom< Device >( initialVec.getData(), vecSize, (Real) -10000, (Real) 10000 );
      }
      else {
         TNL::Algorithms::fillRandom< Device >( initialVec.getData(), vecSize, (Real) -1, (Real) 1 );
      }
   } while( TNL::l2Norm( initialVec ) == 0 );
   return powerIteration( matrix, epsilon, initialVec, maxIterations );
}

}  // namespace TNL::Matrices::Eigen
