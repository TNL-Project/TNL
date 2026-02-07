// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <utility>
#include <TNL/Matrices/LambdaMatrix.h>
#include "PowerIteration.h"

namespace TNL::Solvers::Eigenvalues::experimental {

/**
 * \brief Calculates an eigenvalue and its corresponding eigenvector of a matrix using the shifted power iteration method.
 *
 * This function extends the standard power iteration method by adding a specified shift value to the diagonal of the matrix.
 * It is useful when the largest eigenvalue converges slowly.
 *
 * \tparam MatrixType Type of the matrix, which defines both the data type of the matrix elements (e.g., float, double)
 * and the computational device (e.g., CPU, GPU). The matrix must define two nested types: `RealType` (the data type of the
 * elements) and `DeviceType` (the device where data is stored and operations are executed).
 *
 * \param matrix The matrix for which to calculate the eigenvalue and eigenvector. The matrix type determines both the element
 * data type and the device used for computation.
 * \param epsilon Precision threshold for convergence. The iterative process halts when the difference between successive
 * iterations falls below this value. This should be of the same type as `MatrixType::RealType`.
 * \param shiftValue Shift value applied to the matrix spectrum to accelerate convergence. This should be of the same type
 * as `MatrixType::RealType`.
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
 * \exception std::invalid_argument Thrown if the matrix is not square, is zero-sized, or if the initial vector size does not
 * match the matrix dimensions.
 */
template< typename MatrixType >
std::tuple< typename MatrixType::RealType,
            TNL::Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType >,
            int >
shiftedPowerIteration( const MatrixType& matrix,
                       const typename MatrixType::RealType& epsilon,
                       const typename MatrixType::RealType& shiftValue,
                       TNL::Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType >& initialVec,
                       const int& maxIterations = 100000 )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Shifted power iteration is possible only for square matrices" );
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   if( matrix.getRows() != initialVec.getSize() )
      throw std::invalid_argument( "The initial vector must have the same size as the matrix" );
   using IndexType = typename MatrixType::IndexType;
   using RealType = typename MatrixType::RealType;
   using DeviceType = typename MatrixType::DeviceType;
   IndexType vecSize = matrix.getColumns();
   TNL::Containers::Vector< RealType, DeviceType > eigenVecOut( vecSize );
   eigenVecOut.setValue( 0 );
   RealType norm = 0;
   RealType normOld = 0;
   int iterations = 0;
   TNL::Containers::Vector< RealType, DeviceType > eigenVecOld( vecSize );
   eigenVecOld.setValue( 0 );
   norm = TNL::l2Norm( initialVec );
   if( norm == 0 )
      throw std::invalid_argument( "The initial vector must be nonzero" );
   if( norm != 1 )
      initialVec = initialVec / norm;
   while( true ) {
      matrix.vectorProduct( initialVec, eigenVecOut );
      if( shiftValue != 0 ) {
         eigenVecOut += initialVec * shiftValue;
      }
      norm = TNL::l2Norm( eigenVecOut );
      if( std::isnan( norm ) )
         return std::make_tuple( norm, eigenVecOut, -1 );
      initialVec = std::move( eigenVecOut / norm );
      iterations++;
      if( TNL::abs( normOld - norm ) < epsilon ) {
         if( TNL::all( TNL::less( TNL::abs( initialVec - eigenVecOld ), epsilon ) ) )
            return std::make_tuple( norm - shiftValue, initialVec, iterations );
         if( TNL::all( TNL::less( TNL::abs( initialVec + eigenVecOld ), epsilon ) ) )
            return std::make_tuple( -norm - shiftValue, initialVec, iterations );
      }
      if( iterations == maxIterations )
         return std::make_tuple( norm - shiftValue, initialVec, 0 );
      eigenVecOld = initialVec;
      normOld = norm;
   }
}

/**
 * \brief Calculates an eigenvalue and its corresponding eigenvector of a matrix using the shifted power iteration method.
 *
 * This function extends the standard power iteration method by adding a specified shift value to the diagonal of the matrix.
 * It is useful when the largest eigenvalue converges slowly.
 *
 * \tparam MatrixType Type of the matrix, which defines both the data type of the matrix elements (e.g., float, double)
 * and the computational device (e.g., CPU, GPU). The matrix must define two nested types: `RealType` (the data type of the
 * elements) and `DeviceType` (the device where data is stored and operations are executed).
 *
 * \param matrix The matrix for which to calculate the eigenvalue and eigenvector. The matrix type determines both the element
 * data type and the device used for computation.
 * \param epsilon Precision threshold for convergence. The iterative process halts when the difference between successive
 * iterations falls below this value. This should be of the same type as `MatrixType::RealType`.
 * \param shiftValue Shift value applied to the matrix spectrum to accelerate convergence. This should be of the same type
 * as `MatrixType::RealType`.
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
 * \exception std::invalid_argument Thrown if the matrix is not square, is zero-sized, or if the initial vector size does not
 * match the matrix dimensions.
 *
 * \note Without an initial vector argument, the function generates a random initial vector
 * appropriate for the matrix size and type. The nature of the initial vector can affect the convergence speed of the algorithm.
 */
template< typename MatrixType >
std::tuple< typename MatrixType::RealType,
            TNL::Containers::Vector< typename MatrixType::RealType, typename MatrixType::DeviceType >,
            int >
shiftedPowerIteration( const MatrixType& matrix,
                       const typename MatrixType::RealType& epsilon,
                       const typename MatrixType::RealType& shiftValue,
                       const int& maxIterations = 100000 )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Shifted power iteration is possible only for square matrices" );
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
   return shiftedPowerIteration( matrix, epsilon, shiftValue, std::move( initialVec ), maxIterations );
}

}  // namespace TNL::Solvers::Eigenvalues::experimental
