// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/Matrices/MatrixBase.h>
#include <TNL/Matrices/DenseMatrixBase.h>
#include <TNL/Matrices/SparseMatrixBase.h>

namespace TNL::Matrices {

/**
 * \brief This checks if given type is matrix.
 */
[[nodiscard]] constexpr std::false_type
isMatrix( ... )
{
   return {};
}

template< typename Real, typename Device, typename Index, typename MatrixType, ElementsOrganization Organization >
[[nodiscard]] constexpr std::true_type
isMatrix( const MatrixBase< Real, Device, Index, MatrixType, Organization >& )
{
   return {};
}

template< typename T >
using is_matrix = decltype( isMatrix( std::declval< T >() ) );

template< typename T >
constexpr bool is_matrix_v = is_matrix< T >::value;

/**
 * \brief This checks if the matrix is dense matrix.
 */
[[nodiscard]] constexpr std::false_type
isDenseMatrix( ... )
{
   return {};
}

template< typename Real, typename Device, typename Index, ElementsOrganization Organization >
[[nodiscard]] constexpr std::true_type
isDenseMatrix( const DenseMatrixBase< Real, Device, Index, Organization >& )
{
   return {};
}
template< typename T >
using is_dense_matrix = decltype( isDenseMatrix( std::declval< T >() ) );

template< typename T >
constexpr bool is_dense_matrix_v = is_dense_matrix< T >::value;

/**
 * \brief This checks if the sparse matrix is stored in CSR format.
 */
[[nodiscard]] constexpr std::false_type
isSparseMatrix( ... )
{
   return {};
}

template< typename Real, typename Device, typename Index, typename MatrixType, typename SegmentsView, typename ComputeReal >
[[nodiscard]] constexpr std::true_type
isSparseMatrix( const SparseMatrixBase< Real, Device, Index, MatrixType, SegmentsView, ComputeReal >& )
{
   return {};
}

template< typename Matrix >
using is_sparse_csr_matrix = decltype( isSparseMatrix( std::declval< Matrix >() ) );

template< typename Matrix >
constexpr bool is_sparse_csr_matrix_v = is_sparse_csr_matrix< Matrix >::value;

}  // namespace TNL::Matrices
