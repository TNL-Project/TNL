// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/Eigen/PowerIteration.h>
#include <TNL/Matrices/LambdaMatrix.h>

namespace TNL::Matrices::Eigen {

/**
 * \brief Calculates an eigenvalue and its corresponding eigenvector of a matrix using the shifted power iteration method.
 *
 * This function extends the standard power iteration method by adding a specified shift value to the diagonal of the matrix.
 * It is useful when the largest eigenvalue converges slowly.
 *
 * \tparam Real Data type of the matrix elements (e.g., float, double).
 * \tparam Device Computational device (e.g., CPU, GPU).
 * \tparam MatrixType Type of matrix (e.g., dense, sparse).
 *
 * \param matrix The matrix for which to calculate the eigenvalue and eigenvector.
 * \param epsilon Precision threshold for convergence.
 * \param shiftValue Shift value applied to the matrix spectrum.
 * \param initialVec (Overload) The initial vector to start the iteration with. This vector must have the same size as the
 * matrix dimensions and should not be the zero vector.
 * \param maxIterations (Optional) Maximum number of iterations to perform. Defaults to 100000. If this limit is reached before
 * convergence, the function returns the last computed values with the iteration count set to 0.
 *
 * \return A tuple containing:
 *         - The largest eigenvalue of the matrix (of type Real).
 *         - The eigenvector associated with the largest eigenvalue (of type `TNL::Containers::Vector<T, Device>`).
 *         - The number of iterations performed to reach convergence (of type int). A value of 0 indicates non-convergence
 * within the specified maximum number of iterations, and -1 indicates a computational error resulting in NaN.
 *
 * \exception std::invalid_argument Thrown if the matrix is not square, is zero-sized, or if the initial vector size does not
 * match the matrix dimensions.
 */
template< typename Real, typename Device, typename MatrixType >
std::tuple< Real, TNL::Containers::Vector< Real, Device >, int >
shiftedPowerIteration( const MatrixType& matrix,
                       const Real& epsilon,
                       const Real& shiftValue,
                       TNL::Containers::Vector< Real, Device >& initialVec,
                       const int& maxIterations = 100000 )
{
   static_assert( std::is_same_v< Device, typename MatrixType::DeviceType > );
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Shifted power iteration is possible only for square matrices" );
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
   iterations = -1;
   return std::make_tuple( norm - shiftValue, initialVec, iterations );
}

/**
 * \brief Calculates an eigenvalue and its corresponding eigenvector of a matrix using the shifted power iteration method.
 *
 * This function extends the standard power iteration method by adding a specified shift value to the diagonal of the matrix.
 * It is useful when the largest eigenvalue converges slowly.
 *
 * \tparam Real Data type of the matrix elements (e.g., float, double).
 * \tparam Device Computational device (e.g., CPU, GPU).
 * \tparam MatrixType Type of matrix (e.g., dense, sparse).
 *
 * \param matrix The matrix for which to calculate the eigenvalue and eigenvector.
 * \param epsilon Precision threshold for convergence.
 * \param shiftValue Shift value applied to the matrix spectrum.
 * \param maxIterations (Optional) Maximum number of iterations to perform. Defaults to 100000. If this limit is reached before
 * convergence, the function returns the last computed values with the iteration count set to 0.
 *
 * \return A tuple containing:
 *         - The largest eigenvalue of the matrix (of type Real).
 *         - The eigenvector associated with the largest eigenvalue (of type `TNL::Containers::Vector<T, Device>`).
 *         - The number of iterations performed to reach convergence (of type int). A value of 0 indicates non-convergence
 * within the specified maximum number of iterations, and -1 indicates a computational error resulting in NaN.
 *
 * \exception std::invalid_argument Thrown if the matrix is not square, is zero-sized, or if the initial vector size does not
 * match the matrix dimensions.
 *
 * \note Without an initial vector argument, the function generates a random initial vector
 * appropriate for the matrix size and type. The nature of the initial vector can affect the convergence speed of the algorithm.
 */
template< typename Real, typename Device, typename MatrixType >
std::tuple< Real, TNL::Containers::Vector< Real, Device >, int >
shiftedPowerIteration( const MatrixType& matrix,
                       const Real& epsilon,
                       const Real& shiftValue,
                       const int& maxIterations = 100000 )
{
   static_assert( std::is_same_v< Device, typename MatrixType::DeviceType > );
   using IndexType = typename MatrixType::IndexType;
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Shifted power iteration is possible only for square matrices" );
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
   return shiftedPowerIteration( matrix, epsilon, shiftValue, initialVec, maxIterations );
}

}  // namespace TNL::Matrices::Eigen
