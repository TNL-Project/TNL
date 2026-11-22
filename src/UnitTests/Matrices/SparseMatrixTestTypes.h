// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

// Type lists of sparse matrices for the variants of segments defined in SegmentsTestVariants.h.

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <TNL/Arithmetics/Complex.h>
#include <TNL/Matrices/SparseMatrix.h>

#include "../Algorithms/Segments/SegmentsTestVariants.h"

namespace TNL::UnitTests {

//! \brief Real, index and compute real type of a sparse matrix (the default compute real type is the
//! same as in SparseMatrix).
template< typename Real, typename Index, typename ComputeReal = std::conditional_t< std::is_same_v< Real, bool >, Index, Real > >
struct MatrixElementTypes
{};

namespace detail {

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

template< typename Variants, typename Device, typename... ElementTypes >
struct SparseMatrixTestTypes;

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... ElementTypes >
struct SparseMatrixTestTypes< SegmentsVariants< DefaultSegments, Segments... >, Device, ElementTypes... >
: SparseMatrixSegmentsTypes< DefaultSegments, Device, ElementTypes... >
{};

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... ElementTypes >
struct SparseMatrixTestTypes< SortedSegmentsVariants< DefaultSegments, Segments... >, Device, ElementTypes... >
: SparseMatrixSegmentsTypes< DefaultSegments, Device, ElementTypes... >
{};

// all combinations of the segment templates and element types
template< typename Device, typename ElementTypes, template< typename, typename, typename > class... Segments >
struct SparseMatrixProductTypes;

template< typename Device, typename... ElementTypes, template< typename, typename, typename > class... Segments >
struct SparseMatrixProductTypes< Device, std::tuple< ElementTypes... >, Segments... >
{
   using type = typename ConcatTypes< typename SparseMatrixSegmentsTypes< Segments, Device, ElementTypes... >::type... >::type;
};

template< template< typename, typename, typename > class Segments, typename Device, typename ElementTypes >
struct SparseMatrixType;

template<
   template< typename, typename, typename > class Segments,
   typename Device,
   typename Real,
   typename Index,
   typename ComputeReal >
struct SparseMatrixType< Segments, Device, MatrixElementTypes< Real, Index, ComputeReal > >
{
   using type = Matrices::SparseMatrix< Real, Device, Index, Matrices::GeneralMatrix, Segments, ComputeReal >;
};

// covering set of the segment templates and element types, see SegmentsCoveringTemplateTypes
template<
   template< typename, typename, typename > class Segments,
   std::size_t TemplateIdx,
   std::size_t TemplatesCount,
   typename Device,
   typename... ElementTypes >
struct SparseMatrixCoveringTemplateTypes
{
   static constexpr std::size_t elementTypesCount = sizeof...( ElementTypes );
   static constexpr std::size_t count = std::max( TemplatesCount, elementTypesCount );

   template< std::size_t... J >
   static auto select( std::index_sequence< J... > ) -> ::testing::Types< typename SparseMatrixType<
      Segments,
      Device,
      std::tuple_element_t< ( TemplateIdx + J * TemplatesCount ) % elementTypesCount, std::tuple< ElementTypes... > > >::
                                                                             type... >;

   using type =
      decltype( select( std::make_index_sequence< ( count - TemplateIdx + TemplatesCount - 1 ) / TemplatesCount >{} ) );
};

template<
   typename Device,
   typename ElementTypes,
   typename TemplateIndices,
   template< typename, typename, typename > class... Segments >
struct SparseMatrixCoveringTypesImpl;

template<
   typename Device,
   typename... ElementTypes,
   std::size_t... TemplateIdx,
   template< typename, typename, typename > class... Segments >
struct SparseMatrixCoveringTypesImpl< Device, std::tuple< ElementTypes... >, std::index_sequence< TemplateIdx... >, Segments... >
{
   using type = typename ConcatTypes<
      typename SparseMatrixCoveringTemplateTypes< Segments, TemplateIdx, sizeof...( Segments ), Device, ElementTypes... >::
         type... >::type;
};

template< typename Device, typename ElementTypes, template< typename, typename, typename > class... Segments >
using SparseMatrixCoveringTypes =
   SparseMatrixCoveringTypesImpl< Device, ElementTypes, std::make_index_sequence< sizeof...( Segments ) >, Segments... >;

template< bool Covering, typename Variants, typename Device, typename... ElementTypes >
struct SparseMatrixVariantsTestTypes;

template<
   bool Covering,
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... ElementTypes >
struct SparseMatrixVariantsTestTypes< Covering, SegmentsVariants< DefaultSegments, Segments... >, Device, ElementTypes... >
: std::conditional_t<
     Covering,
     SparseMatrixCoveringTypes< Device, std::tuple< ElementTypes... >, Segments... >,
     SparseMatrixProductTypes< Device, std::tuple< ElementTypes... >, Segments... > >
{};

template<
   bool Covering,
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... ElementTypes >
struct SparseMatrixVariantsTestTypes< Covering, SortedSegmentsVariants< DefaultSegments, Segments... >, Device, ElementTypes... >
: std::conditional_t<
     Covering,
     SparseMatrixCoveringTypes< Device, std::tuple< ElementTypes... >, Segments... >,
     SparseMatrixProductTypes< Device, std::tuple< ElementTypes... >, Segments... > >
{};

}  // namespace detail

/**
 * \brief Type list with general sparse matrices with the default segments of the given
 * variants for the given device and element types (see \ref MatrixElementTypes).
 *
 * For example, `SparseMatrixTestTypes< EllpackVariants, Devices::Host, MatrixElementTypes< float, int > >`
 * gives `::testing::Types< SparseMatrix< float, Host, int, GeneralMatrix, Ellpack > >`.
 */
template< typename Variants, typename Device, typename... ElementTypes >
using SparseMatrixTestTypes = typename detail::SparseMatrixTestTypes< Variants, Device, ElementTypes... >::type;

/**
 * \brief Type list with general sparse matrices with all the given variants of
 * segments (e.g. row-major and column-major) for the given device and element types.
 *
 * The types are all combinations of the variants and element types, ordered by the variants
 * and then by the element types.
 */
template< typename Variants, typename Device, typename... ElementTypes >
using SparseMatrixVariantsTestTypes =
   typename detail::SparseMatrixVariantsTestTypes< false, Variants, Device, ElementTypes... >::type;

/**
 * \brief Covering set of general sparse matrices with the given variants of
 * segments and the given element types for the given device.
 *
 * Each variant and each element type appear at least once: the k-th type is made of the
 * (k mod V)-th variant and the (k mod E)-th element types, where V and E are the numbers of
 * variants and element types, for k = 0, ..., max(V, E) - 1. It is used for the GPU builds
 * of the tests (see src/UnitTests/ARCHITECTURE.md).
 */
template< typename Variants, typename Device, typename... ElementTypes >
using SparseMatrixVariantsCoveringTestTypes =
   typename detail::SparseMatrixVariantsTestTypes< true, Variants, Device, ElementTypes... >::type;

}  // namespace TNL::UnitTests
