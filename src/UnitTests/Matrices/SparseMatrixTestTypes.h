// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

// Type lists of sparse matrices for the formats of segments defined in SegmentsTestFormats.h.

#include <TNL/Matrices/SparseMatrix.h>

#include "../Algorithms/Segments/SegmentsTestFormats.h"

namespace TNL::UnitTests {

//! \brief Pair of the real and index type of a sparse matrix.
template< typename Real, typename Index >
struct MatrixElementTypes
{};

namespace detail {

template< typename Format, typename Device, typename... ElementTypes >
struct SparseMatrixTestTypes;

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... Reals,
   typename... Indices >
struct SparseMatrixTestTypes< SegmentsFormat< DefaultSegments, Segments... >, Device, MatrixElementTypes< Reals, Indices >... >
{
   using type =
      ::testing::Types< Matrices::SparseMatrix< Reals, Device, Indices, Matrices::GeneralMatrix, DefaultSegments >... >;
};

}  // namespace detail

/**
 * \brief Type list with general sparse matrices with the default segments of the given format
 * for the given device and pairs of real and index types.
 *
 * For example, `SparseMatrixTestTypes< EllpackFormat, Devices::Host, MatrixElementTypes< float, int > >`
 * gives `::testing::Types< SparseMatrix< float, Host, int, GeneralMatrix, Ellpack > >`.
 */
template< typename Format, typename Device, typename... ElementTypes >
using SparseMatrixTestTypes = typename detail::SparseMatrixTestTypes< Format, Device, ElementTypes... >::type;

}  // namespace TNL::UnitTests
