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
   using IndexType = typename MatrixType::IndexType;
   IndexType size = matrix.getColumns();
   auto rowLengths =
      [ = ] __cuda_callable__( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType
   {
      if( shiftValue != 0 && matrix.getElement( rowIdx, rowIdx ) == 0 && size > matrix.getRowCapacity( rowIdx ) ) {
         return matrix.getRowCapacity( rowIdx ) + 1;
      }
      else {
         return matrix.getRowCapacity( rowIdx );
      }
   };
   auto matrixElements = [ = ] __cuda_callable__( const IndexType rows,
                                                  const IndexType columns,
                                                  const IndexType rowIdx,
                                                  const IndexType localIdx,
                                                  IndexType& columnIdx,
                                                  Real& value )
   {
      auto row = matrix.getRow( rowIdx );
      IndexType size = row.getSize();
      if( size == localIdx ) {
         columnIdx = rowIdx;
         value = shiftValue;
      }
      else {
         columnIdx = row.getColumnIndex( localIdx );
         if( columnIdx == rowIdx )
            value = row.getValue( localIdx ) + shiftValue;
         else
            value = row.getValue( localIdx );
      }
   };
   using MatrixFactory = TNL::Matrices::LambdaMatrixFactory< Real, Device, IndexType >;
   auto shiftedMatrix = MatrixFactory::create( size, size, matrixElements, rowLengths );
   std::tuple< Real, TNL::Containers::Vector< Real, Device >, int > tuple =
      TNL::Matrices::Eigen::powerIteration< Real, Device >( shiftedMatrix, epsilon, initialVec, maxIterations );
   return std::make_tuple( std::get< 0 >( tuple ) - shiftValue, std::get< 1 >( tuple ), std::get< 2 >( tuple ) );
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
