// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

// Formats of segments for which the unit tests of segments (and of the algorithms and
// data structures based on segments) are instantiated. Each format is a list of segment
// templates with the signature < Device, Index, IndexAllocator >, which is the same as
// the signature of the Segments template parameter of SparseMatrix. The type lists for
// a test suite are generated with SegmentsTestTypes.

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

//! \brief List of segment templates that make up one format.
template< template< typename Device, typename Index, typename IndexAllocator > class... Segments >
struct SegmentsFormat
{};

/**
 * \brief List of segment templates that make up one format of sorted segments.
 *
 * Sorted segments wrap the segments of another format, which is tested on its own. Hence,
 * the tests of sorted segments on GPUs are instantiated only for a covering set of the
 * segment templates and index types (see \ref SegmentsTestTypes).
 */
template< template< typename Device, typename Index, typename IndexAllocator > class... Segments >
struct SortedSegmentsFormat
{};

// clang-format off
using CSRFormat = SegmentsFormat< Algorithms::Segments::CSR >;
using AdaptiveCSRFormat = SegmentsFormat< Algorithms::Segments::AdaptiveCSR >;
using EllpackFormat = SegmentsFormat< Algorithms::Segments::RowMajorEllpack, Algorithms::Segments::ColumnMajorEllpack >;
using SlicedEllpackFormat = SegmentsFormat< Algorithms::Segments::RowMajorSlicedEllpack, Algorithms::Segments::ColumnMajorSlicedEllpack >;
using ChunkedEllpackFormat = SegmentsFormat< Algorithms::Segments::RowMajorChunkedEllpack, Algorithms::Segments::ColumnMajorChunkedEllpack >;
using BiEllpackFormat = SegmentsFormat< Algorithms::Segments::RowMajorBiEllpack, Algorithms::Segments::ColumnMajorBiEllpack >;
using SortedCSRFormat = SortedSegmentsFormat< Algorithms::Segments::SortedCSR >;
using SortedAdaptiveCSRFormat = SortedSegmentsFormat< Algorithms::Segments::SortedAdaptiveCSR >;
using SortedEllpackFormat = SortedSegmentsFormat< Algorithms::Segments::SortedRowMajorEllpack, Algorithms::Segments::SortedColumnMajorEllpack >;
using SortedSlicedEllpackFormat = SortedSegmentsFormat< Algorithms::Segments::SortedRowMajorSlicedEllpack, Algorithms::Segments::SortedColumnMajorSlicedEllpack >;
using SortedChunkedEllpackFormat = SortedSegmentsFormat< Algorithms::Segments::SortedRowMajorChunkedEllpack, Algorithms::Segments::SortedColumnMajorChunkedEllpack >;
using SortedBiEllpackFormat = SortedSegmentsFormat< Algorithms::Segments::SortedRowMajorBiEllpack, Algorithms::Segments::SortedColumnMajorBiEllpack >;
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

template< typename Format, typename Device, typename... Indices >
struct SegmentsTestTypes;

template< template< typename, typename, typename > class... Segments, typename Device, typename... Indices >
struct SegmentsTestTypes< SegmentsFormat< Segments... >, Device, Indices... >
{
   using type = typename SegmentsProductTypes< Device, std::tuple< Indices... >, Segments... >::type;
};

template< template< typename, typename, typename > class... Segments, typename Device, typename... Indices >
struct SegmentsTestTypes< SortedSegmentsFormat< Segments... >, Device, Indices... >
{
   using type = typename std::conditional_t<
      std::is_same_v< Device, Devices::Host >,
      SegmentsProductTypes< Device, std::tuple< Indices... >, Segments... >,
      SegmentsCoveringTypes< Device, std::tuple< Indices... >, Segments... > >::type;
};

}  // namespace detail

/**
 * \brief Type list with the segments of the given format for the given device and index types.
 *
 * The types are ordered by the segment templates of the format and then by the index types,
 * e.g. `SegmentsTestTypes< EllpackFormat, Devices::Host, int, long >` gives
 * `::testing::Types< RowMajorEllpack< Host, int >, RowMajorEllpack< Host, long >,
 * ColumnMajorEllpack< Host, int >, ColumnMajorEllpack< Host, long > >`.
 *
 * For a \ref SortedSegmentsFormat and a device other than \ref Devices::Host, the list is
 * reduced to a covering set, where each segment template and each index type appear at least
 * once, e.g. `SegmentsTestTypes< SortedEllpackFormat, Devices::GPU, int, long >` gives
 * `::testing::Types< SortedRowMajorEllpack< GPU, int >, SortedColumnMajorEllpack< GPU, long > >`.
 */
template< typename Format, typename Device, typename... Indices >
using SegmentsTestTypes = typename detail::SegmentsTestTypes< Format, Device, Indices... >::type;

}  // namespace TNL::UnitTests
