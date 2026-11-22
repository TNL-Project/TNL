// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

// Segments types for which the unit tests of segments (and of the algorithms and
// data structures based on segments) are instantiated. Each of them is given by the segment
// templates with the signature < Device, Index, IndexAllocator >, which is the same as
// the signature of the Segments template parameter of SparseMatrix. The type lists for
// the tests of segments are generated with SegmentsTestTypes, the type lists for the
// tests of sparse matrices with SparseMatrixTestTypes (see Matrices/SparseMatrixTestTypes.h).

#include <algorithm>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

#include <TNL/Algorithms/Segments/AdaptiveCSR.h>
#include <TNL/Algorithms/Segments/BiEllpack.h>
#include <TNL/Algorithms/Segments/ChunkedEllpack.h>
#include <TNL/Algorithms/Segments/CSR.h>
#include <TNL/Algorithms/Segments/Ellpack.h>
#include <TNL/Algorithms/Segments/SlicedEllpack.h>
#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Allocators/Default.h>
#include <TNL/Devices/Host.h>

#if defined( __CUDACC__ ) || defined( __HIP__ )
   #include <TNL/Devices/GPU.h>
#endif

#include <gtest/gtest.h>

namespace TNL::UnitTests {

//! \brief Device for which the tests are instantiated in the current translation unit.
#if defined( __CUDACC__ ) || defined( __HIP__ )
using TestDevice = Devices::GPU;
#else
using TestDevice = Devices::Host;
#endif

/**
 * \brief Variants of a segments type (e.g. row-major and column-major Ellpack).
 *
 * \tparam DefaultSegments is the segments template with the default organization for the
 *         device. It is used in the tests of the data structures based on segments, such as
 *         sparse matrices.
 * \tparam Segments are all variants of the segments type (e.g. row-major and column-major), which
 *         are used in the tests of segments (see \ref SegmentsTestTypes).
 */
template<
   template< typename Device, typename Index, typename IndexAllocator > class DefaultSegments,
   template< typename Device, typename Index, typename IndexAllocator > class... Segments >
struct SegmentsVariants
{};

/**
 * \brief Variants of a sorted segments type.
 *
 * The template parameters are the same as for \ref SegmentsVariants. Sorted segments wrap the
 * segments of another type, which is tested on its own. Hence, the tests of sorted segments
 * on GPUs are instantiated only for a covering set of the segment templates and index types
 * (see \ref SegmentsTestTypes).
 */
template<
   template< typename Device, typename Index, typename IndexAllocator > class DefaultSegments,
   template< typename Device, typename Index, typename IndexAllocator > class... Segments >
struct SortedSegmentsVariants
{};

// clang-format off
using CSRVariants = SegmentsVariants< Algorithms::Segments::CSR,
                                    Algorithms::Segments::CSR >;
using AdaptiveCSRVariants = SegmentsVariants< Algorithms::Segments::AdaptiveCSR,
                                            Algorithms::Segments::AdaptiveCSR >;
using EllpackVariants = SegmentsVariants< Algorithms::Segments::Ellpack,
                                        Algorithms::Segments::RowMajorEllpack, Algorithms::Segments::ColumnMajorEllpack >;
using SlicedEllpackVariants = SegmentsVariants< Algorithms::Segments::SlicedEllpack,
                                              Algorithms::Segments::RowMajorSlicedEllpack, Algorithms::Segments::ColumnMajorSlicedEllpack >;
using ChunkedEllpackVariants = SegmentsVariants< Algorithms::Segments::ChunkedEllpack,
                                               Algorithms::Segments::RowMajorChunkedEllpack, Algorithms::Segments::ColumnMajorChunkedEllpack >;
using BiEllpackVariants = SegmentsVariants< Algorithms::Segments::BiEllpack,
                                          Algorithms::Segments::RowMajorBiEllpack, Algorithms::Segments::ColumnMajorBiEllpack >;
using SortedCSRVariants = SortedSegmentsVariants< Algorithms::Segments::SortedCSR,
                                                Algorithms::Segments::SortedCSR >;
using SortedAdaptiveCSRVariants = SortedSegmentsVariants< Algorithms::Segments::SortedAdaptiveCSR,
                                                        Algorithms::Segments::SortedAdaptiveCSR >;
using SortedEllpackVariants = SortedSegmentsVariants< Algorithms::Segments::SortedEllpack,
                                                    Algorithms::Segments::SortedRowMajorEllpack, Algorithms::Segments::SortedColumnMajorEllpack >;
using SortedSlicedEllpackVariants = SortedSegmentsVariants< Algorithms::Segments::SortedSlicedEllpack,
                                                          Algorithms::Segments::SortedRowMajorSlicedEllpack, Algorithms::Segments::SortedColumnMajorSlicedEllpack >;
using SortedChunkedEllpackVariants = SortedSegmentsVariants< Algorithms::Segments::SortedChunkedEllpack,
                                                           Algorithms::Segments::SortedRowMajorChunkedEllpack, Algorithms::Segments::SortedColumnMajorChunkedEllpack >;
using SortedBiEllpackVariants = SortedSegmentsVariants< Algorithms::Segments::SortedBiEllpack,
                                                      Algorithms::Segments::SortedRowMajorBiEllpack, Algorithms::Segments::SortedColumnMajorBiEllpack >;
// clang-format on

namespace detail {

template< typename... TypeLists >
struct ConcatTypes;

template< typename... Types >
struct ConcatTypes< ::testing::Types< Types... > >
{
   using type = ::testing::Types< Types... >;
};

template< typename... Types1, typename... Types2, typename... Rest >
struct ConcatTypes< ::testing::Types< Types1... >, ::testing::Types< Types2... >, Rest... >
: ConcatTypes< ::testing::Types< Types1..., Types2... >, Rest... >
{};

template< template< typename, typename, typename > class Segments, typename Device, typename Index >
using SegmentsType = Segments< Device, Index, typename Allocators::Default< Device >::template Allocator< Index > >;

// type list with one segment template for all index types
template< template< typename, typename, typename > class Segments, typename Device, typename... Indices >
struct SegmentsTemplateTypes
{
   using type = ::testing::Types< SegmentsType< Segments, Device, Indices >... >;
};

// all combinations of the segment templates and index types
template< typename Device, typename Indices, template< typename, typename, typename > class... Segments >
struct SegmentsProductTypes;

template< typename Device, typename... Indices, template< typename, typename, typename > class... Segments >
struct SegmentsProductTypes< Device, std::tuple< Indices... >, Segments... >
{
   using type = typename ConcatTypes< typename SegmentsTemplateTypes< Segments, Device, Indices... >::type... >::type;
};

// covering set of the segment templates and index types: the k-th type of the list is made of the
// (k mod S)-th segment template and the (k mod I)-th index type, where S and I are the numbers of
// segment templates and index types, for k = 0, ..., max(S, I) - 1. This part of the list contains
// the types with the segment template Segments at the position TemplateIdx.
template<
   template< typename, typename, typename > class Segments,
   std::size_t TemplateIdx,
   std::size_t TemplatesCount,
   typename Device,
   typename... Indices >
struct SegmentsCoveringTemplateTypes
{
   static constexpr std::size_t indicesCount = sizeof...( Indices );
   static constexpr std::size_t count = std::max( TemplatesCount, indicesCount );

   template< std::size_t... J >
   static auto select( std::index_sequence< J... > ) -> ::testing::Types< SegmentsType<
      Segments,
      Device,
      std::tuple_element_t< ( TemplateIdx + J * TemplatesCount ) % indicesCount, std::tuple< Indices... > > >... >;

   using type =
      decltype( select( std::make_index_sequence< ( count - TemplateIdx + TemplatesCount - 1 ) / TemplatesCount >{} ) );
};

template<
   typename Device,
   typename Indices,
   typename TemplateIndices,
   template< typename, typename, typename > class... Segments >
struct SegmentsCoveringTypesImpl;

template<
   typename Device,
   typename... Indices,
   std::size_t... TemplateIdx,
   template< typename, typename, typename > class... Segments >
struct SegmentsCoveringTypesImpl< Device, std::tuple< Indices... >, std::index_sequence< TemplateIdx... >, Segments... >
{
   using type = typename ConcatTypes<
      typename SegmentsCoveringTemplateTypes< Segments, TemplateIdx, sizeof...( Segments ), Device, Indices... >::type... >::
      type;
};

template< typename Device, typename Indices, template< typename, typename, typename > class... Segments >
using SegmentsCoveringTypes =
   SegmentsCoveringTypesImpl< Device, Indices, std::make_index_sequence< sizeof...( Segments ) >, Segments... >;

template< typename Variants, typename Device, typename... Indices >
struct SegmentsTestTypes;

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... Indices >
struct SegmentsTestTypes< SegmentsVariants< DefaultSegments, Segments... >, Device, Indices... >
{
   using type = typename SegmentsProductTypes< Device, std::tuple< Indices... >, Segments... >::type;
};

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... Indices >
struct SegmentsTestTypes< SortedSegmentsVariants< DefaultSegments, Segments... >, Device, Indices... >
{
   using type = typename std::conditional_t<
      std::is_same_v< Device, Devices::Host >,
      SegmentsProductTypes< Device, std::tuple< Indices... >, Segments... >,
      SegmentsCoveringTypes< Device, std::tuple< Indices... >, Segments... > >::type;
};

}  // namespace detail

/**
 * \brief Type list with the given variants of segments for the given device and index types.
 *
 * The types are ordered by the segment templates of the variants and then by the index types,
 * e.g. `SegmentsTestTypes< EllpackVariants, Devices::Host, int, long >` gives
 * `::testing::Types< RowMajorEllpack< Host, int >, RowMajorEllpack< Host, long >,
 * ColumnMajorEllpack< Host, int >, ColumnMajorEllpack< Host, long > >`.
 *
 * For a \ref SortedSegmentsVariants and a device other than \ref Devices::Host, the list is
 * reduced to a covering set, where each segment template and each index type appear at least
 * once, e.g. `SegmentsTestTypes< SortedEllpackVariants, Devices::GPU, int, long >` gives
 * `::testing::Types< SortedRowMajorEllpack< GPU, int >, SortedColumnMajorEllpack< GPU, long > >`.
 */
template< typename Variants, typename Device, typename... Indices >
using SegmentsTestTypes = typename detail::SegmentsTestTypes< Variants, Device, Indices... >::type;

}  // namespace TNL::UnitTests
