// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/Eigen/PowerIteration.h>
#include <TNL/Matrices/LambdaMatrix.h>
#include <concepts>

namespace TNL::Matrices::Eigen {

/**
 * \brief Calculates an eigenvalue and its corresponding eigenvector of a matrix using the shifted power iteration method.
 *
 * The shifted power iteration method enhances the standard power iteration algorithm by targeting convergence towards an
 * eigenvalue closer to a specified shift value. This is achieved by shifting the spectrum of the matrix, making a specific
 * eigenvalue more dominant and thus accelerating convergence. This method is particularly advantageous when seeking an
 * eigenvalue other than the largest, or when the largest eigenvalue converges slowly.
 *
 * \tparam T Data type of the matrix elements (e.g., float, double), influencing the precision of computations.
 * \tparam Device Computational device (e.g., CPU, GPU) for data storage and operations.
 * \tparam MatrixType Type of matrix (e.g., dense, sparse) involved in the computation.
 *
 * \param matrix Original matrix for which to calculate the eigenvalue and eigenvector.
 * \param epsilon Precision threshold for convergence, determining when the iterative process should halt.
 * \param shiftValue Shift value applied to the matrix spectrum, influencing the convergence target eigenvalue.
 * \param initialVec (Overload) Initial vector for iteration, required to have the same dimension as the matrix.
 * \param maxIterations (Optional) Maximum number of iterations to perform, defaults to 100000. Convergence failure within
 *                      this limit results in returning the last computed values with an iteration count of -1.
 *
 * \return A tuple containing three elements:
 *         - The eigenvalue closest to the shiftValue (of type T).
 *         - The corresponding eigenvector (of type TNL::Containers::Vector<T, Device>).
 *         - The number of iterations performed to reach convergence (of type int), where -1 indicates non-convergence
 *           within the specified iteration limit.
 *
 * \exception std::invalid_argument Thrown if the input matrix is not square, if it is zero-sized, or if the initial vector
 *                                  (in the overload that requires one) does not match the matrix's dimensions.
 *
 * \note The effectiveness of the shifted power iteration depends heavily on the choice of shiftValue; a shift closer to
 *       the target eigenvalue can significantly enhance convergence speed. This method assumes the input matrix is square
 *       and possesses a dominant eigenvalue near the shift value for successful convergence.
 * \note In the overload without an initial vector, a random vector is generated as a starting point. The nature of this
 *       initial vector can affect the algorithm's convergence speed and success.
 */
template< typename T, typename Device, typename MatrixType >
static std::tuple< T, TNL::Containers::Vector< T, Device >, int >
ShiftedPowerIteration( const MatrixType& matrix,
                       const T& epsilon,
                       const T& shiftValue,
                       TNL::Containers::Vector< T, Device >& initialVec,
                       const uint& maxIterations = 100000 )
{
   if( matrix.getRows() != matrix.getColumns() )
      throw std::invalid_argument( "Shifted power iteration is possible only for square matrices" );
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   using IndexType = typename MatrixType::IndexType;
   int size = matrix.getColumns();
   auto rowLengths = [ = ] __cuda_callable__( const IndexType rows, const IndexType columns, const IndexType rowIdx ) -> IndexType
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
                                                  T& value )
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
   using MatrixFactory = TNL::Matrices::LambdaMatrixFactory< T, Device, IndexType >;
   auto shiftedMatrix = MatrixFactory::create( size, size, matrixElements, rowLengths );
   std::tuple< T, TNL::Containers::Vector< T, Device >, int > tuple =
      TNL::Matrices::Eigen::powerIteration< T, Device >( shiftedMatrix, epsilon, initialVec , maxIterations );
   return std::make_tuple( std::get< 0 >( tuple ) - shiftValue, std::get< 1 >( tuple ), std::get< 2 >( tuple ) );
}

template< typename T, typename Device, typename MatrixType >
static std::tuple< T, TNL::Containers::Vector< T, Device >, int >
ShiftedPowerIteration( const MatrixType& matrix, const T& epsilon, const T& shiftValue, const uint& maxIterations = 100000 )
{
   using IndexType = typename MatrixType::IndexType;
   if( matrix.getRows() == 0 )
      throw std::invalid_argument( "Zero-sized matrices are not allowed" );
   IndexType vecSize = matrix.getRows();
   TNL::Containers::Vector< T, Device > initialVec( vecSize );
   initialVec.resize( vecSize );
   if constexpr( std::is_integral_v< T > ) {
      TNL::Algorithms::fillRandom< Device >( initialVec.getData(), vecSize, (T) -10000, (T) 10000 );
   }
   else {
      TNL::Algorithms::fillRandom< Device >( initialVec.getData(), vecSize, (T) -1, (T) 1 );
   }
   return ShiftedPowerIteration( matrix, epsilon, shiftValue, initialVec, maxIterations );
}

}  //namespace TNL::Matrices::Eigen
