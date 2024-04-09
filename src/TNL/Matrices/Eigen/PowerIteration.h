// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <tuple>

#include "TNL/Containers/Expressions/ExpressionTemplates.h"
#include "TNL/Math.h"
#include <TNL/Containers/Vector.h>
#include <TNL/Algorithms/fillRandom.h>
#include <utility>

namespace TNL::Matrices::Eigen {

/**
 * \brief Calculates the largest eigenvalue and its corresponding eigenvector of a given matrix using the power iteration
 * method.
 *
 * This template function implements the power iteration algorithm to find the largest eigenvalue and its associated eigenvector
 * for a square matrix. The power iteration is a simple yet effective eigenvalue algorithm that, given a square matrix,
 * iteratively produces a number (the largest eigenvalue) and a nonzero vector (the corresponding eigenvector). The process
 * involves repeatedly applying the matrix to an initial vector, normalizing the resulting vector, and checking for convergence
 * against a specified precision threshold.
 *
 * \tparam T Data type of the matrix elements (e.g., float, double). This determines the computation precision.
 * \tparam Device Computational device (e.g., CPU, GPU) where data is stored and operations are executed.
 * \tparam MatrixType Type of the matrix (e.g., dense matrix, sparse matrix) used in the computation.
 *
 * \param matrix The square matrix for which to calculate the largest eigenvalue and associated eigenvector.
 * \param epsilon Precision threshold for convergence. The iterative process halts when the difference between successive
 *                iterations falls below this value.
 * \param initialVec (Overload) The initial vector to start the iteration with. This vector must have the same size as the
 *                   matrix dimensions and should not be the zero vector. Providing a suitable initial vector can improve
 *                   convergence rates.
 * \param maxIterations (Optional) Maximum number of iterations to perform. Defaults to 100000. If this limit is reached before
 *                      convergence, the function returns the last computed values with the iteration count set to -1.
 *
 * \return A tuple containing three elements:
 *         - The largest eigenvalue of the matrix (of type T).
 *         - The eigenvector associated with the largest eigenvalue (of type TNL::Containers::Vector<T, Device>).
 *         - The number of iterations performed to reach convergence (of type int). A value of -1 indicates that convergence was
 *           not achieved within the specified maximum number of iterations.
 *
 * \exception std::invalid_argument Thrown if the input matrix is not square, is zero-sized, or if the initial vector's size
 *                                  does not match the matrix dimensions (for the overload requiring an initial vector).
 *
 * \note This method assumes the matrix has a dominant eigenvalue for convergence. It may not work as intended for matrices
 *       without a well-separated largest eigenvalue or for those with certain types of spectral properties.
 * \note The overload without an initial vector argument generates a random initial vector appropriate for the matrix size
 *       and type. The nature of the initial vector can affect the convergence speed of the algorithm.
 */
template< typename T, typename Device, typename MatrixType >
static std::tuple< T, TNL::Containers::Vector< T, Device >, int >
powerIteration( const MatrixType& matrix,
                const T& epsilon,
                TNL::Containers::Vector< T, Device >& initialVec,
                const uint& maxIterations = 100000 )  //Vytvořit přetížení fce pro initialVec
{
   // upravit vyjímky na std excpetion, invalid argument
   //přidat vyjímku pro nulovou matici
   if( matrix.getRows() == matrix.getColumns() )
      std::invalid_argument( "Power iteration is possible only for square matrices" );
   if( matrix.getRows() == 0 )
      std::invalid_argument( "Zero-sized matrices are not allowed" );
   if( matrix.getRows() == initialVec.getSize() )
      std::invalid_argument( "The initial vector must have the same size as the matrix" );
   using IndexType = typename MatrixType::IndexType;
   IndexType vecSize = matrix.getColumns();
   TNL::Containers::Vector< T, Device > eigenVecOut( vecSize );
   T norm = 0;
   T normOld = 0;
   int iterations = 0;
   TNL::Containers::Vector< T, Device > eigenVecOld( vecSize );
   eigenVecOld.setValue( 0 );
   //std::cout << "start" << "\n";
   while( true ) {
      std::cout << initialVec << "\n";
      matrix.vectorProduct( initialVec, eigenVecOut );
      //auto [ v, i ] = TNL::argMax( abs( eigenVecOut ) );
      //norm = eigenVecOut.getElement( i );
      norm = TNL::l2Norm( eigenVecOut );
      initialVec = std::move( eigenVecOut / norm );
      iterations++;
      if( iterations == maxIterations )
         return std::make_tuple( norm, initialVec, -1 );
      if( TNL::abs( normOld - norm ) < epsilon ) {
         if( TNL::all( TNL::less( TNL::abs( initialVec - eigenVecOld ), epsilon ) ) )
            return std::make_tuple( norm, initialVec, iterations );
         if( TNL::all( TNL::less( TNL::abs( initialVec + eigenVecOld ), epsilon ) ) )
            return std::make_tuple( -norm, initialVec, iterations );
         if( TNL::abs( normOld - norm ) < epsilon * epsilon )
            break;
      }
      eigenVecOld = initialVec;
      normOld = norm;
   }
   iterations = -1;
   //matrix.vectorProduct(initialVec, eigenVecOld);
   //norm = (initialVec, eigenVecOld);
   return std::make_tuple( norm, initialVec, iterations );
}

template< typename T, typename Device, typename MatrixType >
static std::tuple< T, TNL::Containers::Vector< T, Device >, int >
powerIteration( const MatrixType& matrix, const T& epsilon, const uint& maxIterations = 100000 )
{
   using IndexType = typename MatrixType::IndexType;
   if( matrix.getRows() == 0 )
      std::invalid_argument( "Zero-sized matrices are not allowed" );
   IndexType vecSize = matrix.getRows();
   TNL::Containers::Vector< T, Device > initialVec( vecSize );
   initialVec.resize( vecSize );
   if constexpr( std::is_integral_v< T > ) {
      TNL::Algorithms::fillRandom< Device >( initialVec.getData(), vecSize, (T) -10000, (T) 10000 );
   }
   else {
      TNL::Algorithms::fillRandom< Device >( initialVec.getData(), vecSize, (T) -1, (T) 1 );
   }
   return powerIteration( matrix, epsilon, initialVec, maxIterations );
}

}  //namespace TNL::Matrices::Eigen
