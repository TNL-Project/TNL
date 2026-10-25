// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

// Type lists of sparse matrices for the formats of segments defined in SegmentsTestFormats.h.

#include <type_traits>

#include <TNL/Arithmetics/Complex.h>
#include <TNL/Matrices/SparseMatrix.h>

#include "../Algorithms/Segments/SegmentsTestFormats.h"

namespace TNL::UnitTests {

//! \brief Real, index and compute real type of a sparse matrix (the default compute real type is the
//! same as in SparseMatrix).
template< typename Real, typename Index, typename ComputeReal = std::conditional_t< std::is_same_v< Real, bool >, Index, Real > >
struct MatrixElementTypes
{};

namespace detail {

// type list with general sparse matrices with one segments template for all element types
template< template< typename, typename, typename > class Segments, typename Device, typename... ElementTypes >
struct SparseMatrixSegmentsTypes;

template<
   template< typename, typename, typename > class Segments,
   typename Device,
   typename... Reals,
   typename... Indices,
   typename... ComputeReals >
struct SparseMatrixSegmentsTypes< Segments, Device, MatrixElementTypes< Reals, Indices, ComputeReals >... >
{
   using type =
      ::testing::Types< Matrices::SparseMatrix< Reals, Device, Indices, Matrices::GeneralMatrix, Segments, ComputeReals >... >;
};

template< typename Format, typename Device, typename... ElementTypes >
struct SparseMatrixTestTypes;

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... ElementTypes >
struct SparseMatrixTestTypes< SegmentsFormat< DefaultSegments, Segments... >, Device, ElementTypes... >
{
   using type = typename SparseMatrixSegmentsTypes< DefaultSegments, Device, ElementTypes... >::type;
};

template< typename Format, typename Device, typename... ElementTypes >
struct SparseMatrixVariantsTestTypes;

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... ElementTypes >
struct SparseMatrixVariantsTestTypes< SegmentsFormat< DefaultSegments, Segments... >, Device, ElementTypes... >
{
   using type = typename ConcatTypes< typename SparseMatrixSegmentsTypes< Segments, Device, ElementTypes... >::type... >::type;
};

}  // namespace detail

/**
 * \brief Type list with general sparse matrices with the default segments of the given format
 * for the given device and element types.
 *
 * For example, `SparseMatrixTestTypes< EllpackFormat, Devices::Host, MatrixElementTypes< float, int > >`
 * gives `::testing::Types< SparseMatrix< float, Host, int, GeneralMatrix, Ellpack > >`.
 */
template< typename Format, typename Device, typename... ElementTypes >
using SparseMatrixTestTypes = typename detail::SparseMatrixTestTypes< Format, Device, ElementTypes... >::type;

/**
 * \brief Type list with general sparse matrices with all variants of segments of the given
 * format (e.g. row-major and column-major) for the given device and element types.
 *
 * The types are ordered by the segments templates and then by the element types.
 */
template< typename Format, typename Device, typename... ElementTypes >
using SparseMatrixVariantsTestTypes = typename detail::SparseMatrixVariantsTestTypes< Format, Device, ElementTypes... >::type;

}  // namespace TNL::UnitTests
