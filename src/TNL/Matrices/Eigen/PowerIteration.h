// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <stdexcept>
#include <tuple>

#include <TNL/Math.h>
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/fillRandom.h>
#include <TNL/Matrices/Eigen/ShiftedPowerIteration.h>
#include <utility>

namespace TNL::Matrices::Eigen {

/**
 * \brief Calculates the largest eigenvalue and its corresponding eigenvector of a given matrix using the power iteration
 * method.
 *
 * This function implements the power iteration algorithm to find the largest eigenvalue and its associated eigenvector for a
 * square matrix. The process involves repeatedly applying the matrix to an initial vector, normalizing the resulting vector,
 * and checking for convergence against a specified precision threshold.
 *
 * \tparam MatrixType Type of the matrix, which defines both the data type of the matrix elements (e.g., float, double)
 * and the computational device (e.g., CPU, GPU). The matrix must define two nested types: `RealType` (the data type of the
 * elements) and `DeviceType` (the device where data is stored and operations are executed).
 *
 * \param matrix The square matrix for which to calculate the largest eigenvalue and associated eigenvector. The matrix type
 * determines both the element data type and the device used for computation.
 * \param epsilon Precision threshold for convergence. The iterative process halts when the difference between successive
 * iterations falls below this value. This should be of the same type as `MatrixType::RealType`.
 * \param initialVec The initial vector to start the iteration with. This vector must have the same size as the matrix
 * dimensions and should not be the zero vector. It is of type `TNL::Containers::Vector<typename MatrixType::RealType, typename
 * MatrixType::DeviceType>`.
 * \param maxIterations (Optional) Maximum number of iterations to perform. Defaults to 100000. If
 * this limit is reached before convergence, the function returns the last computed values with the iteration count set to 0.
 *
 * \return A tuple containing:
 *         - The largest eigenvalue of the matrix (of type `MatrixType::RealType`).
 *         - The eigenvector associated with the largest eigenvalue (of type `TNL::Containers::Vector<typename
 * MatrixType::RealType, typename MatrixType::DeviceType>`).
 *         - The number of iterations performed to reach convergence (of type `int`). A value of 0 indicates non-convergence
 * within the specified maximum number of iterations, and -1 indicates a computational error resulting in NaN.
 *
 * \exception std::invalid_argument Thrown if the input matrix is not square, is zero-sized, or if the initial vector's size
 * does not match the matrix dimensions.
 */
template< typename MatrixType >
std::tuple< typename MatrixType::RealType,
            TNL::Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType >,
            int >
powerIteration( const MatrixType& matrix,
                const typename MatrixType::RealType& epsilon,
                TNL::Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType >& initialVec,
                const int& maxIterations = 100000 )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Power iteration is possible only for square matrices" );
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   if( matrix.getRows() != initialVec.getSize() )
      throw std::invalid_argument( "The initial vector must have the same size as the matrix" );
   return TNL::Matrices::Eigen::shiftedPowerIteration< MatrixType >( matrix, epsilon, 0, initialVec, maxIterations );
}

/**
 * \brief Calculates the largest eigenvalue and its corresponding eigenvector of a given matrix using the power iteration
 * method.
 *
 * This function implements the power iteration algorithm to find the largest eigenvalue and its associated eigenvector for a
 * square matrix. The process involves repeatedly applying the matrix to an initial vector, normalizing the resulting vector,
 * and checking for convergence against a specified precision threshold.
 *
 * \tparam MatrixType Type of the matrix, which defines both the data type of the matrix elements (e.g., float, double)
 * and the computational device (e.g., CPU, GPU). The matrix must define two nested types: `RealType` (the data type of the
 * elements) and `DeviceType` (the device where data is stored and operations are executed).
 *
 * \param matrix The square matrix for which to calculate the largest eigenvalue and associated eigenvector. The matrix type
 * determines both the element data type and the device used for computation.
 * \param epsilon Precision threshold for convergence. The iterative process halts when the difference between successive
 * iterations falls below this value. This should be of the same type as `MatrixType::RealType`.
 * \param maxIterations (Optional) Maximum number of iterations to perform. Defaults to 100000. If this limit is reached before
 * convergence, the function returns the last computed values with the iteration count set to 0.
 *
 * \return A tuple containing:
 *         - The largest eigenvalue of the matrix (of type `MatrixType::RealType`).
 *         - The eigenvector associated with the largest eigenvalue (of type `TNL::Containers::Vector<typename
 * MatrixType::RealType, typename MatrixType::DeviceType>`).
 *         - The number of iterations performed to reach convergence (of type `int`). A value of 0 indicates non-convergence
 * within the specified maximum number of iterations, and -1 indicates a computational error resulting in NaN.
 *
 * \exception std::invalid_argument Thrown if the input matrix is not square, is zero-sized, or if the initial vector size does
 * not match the matrix dimensions.
 *
 * \note Without an initial vector argument, the function generates a random initial vector
 * appropriate for the matrix size and type. The nature of the initial vector can affect the convergence speed of the algorithm.
 */
template< typename MatrixType >
std::tuple< typename MatrixType::RealType,
            TNL::Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType >,
            int >
powerIteration( const MatrixType& matrix, const typename MatrixType::RealType& epsilon, const int& maxIterations = 100000 )
{
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   using IndexType = typename MatrixType::IndexType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   IndexType vecSize = matrix.getRows();
   TNL::Containers::Vector< RealType, DeviceType > initialVec( vecSize );
   initialVec.resize( vecSize );
   do {
      if constexpr( std::is_integral_v< RealType > ) {
         TNL::Algorithms::fillRandom< DeviceType >( initialVec.getData(), vecSize, (RealType) -10000, (RealType) 10000 );
      }
      else {
         TNL::Algorithms::fillRandom< DeviceType >( initialVec.getData(), vecSize, (RealType) -1, (RealType) 1 );
      }
   } while( TNL::l2Norm( initialVec ) == 0 );
   return powerIteration( matrix, epsilon, initialVec, maxIterations );
}

}  // namespace TNL::Matrices::Eigen
