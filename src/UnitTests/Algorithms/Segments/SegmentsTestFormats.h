// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

// Formats of segments for which the unit tests of segments (and of the algorithms and
// data structures based on segments) are instantiated. Each format consists of segment
// templates with the signature < Device, Index, IndexAllocator >, which is the same as
// the signature of the Segments template parameter of SparseMatrix. The type lists for
// the tests of segments are generated with SegmentsTestTypes, the type lists for the
// tests of sparse matrices with SparseMatrixTestTypes (see Matrices/SparseMatrixTestTypes.h).

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
 * \brief Format of segments.
 *
 * \tparam DefaultSegments is the segments template with the default organization for the
 *         device. It is used in the tests of the data structures based on segments, such as
 *         sparse matrices.
 * \tparam Segments are all variants of the format (e.g. row-major and column-major), which
 *         are used in the tests of segments (see \ref SegmentsTestTypes).
 */
template<
   template< typename Device, typename Index, typename IndexAllocator > class DefaultSegments,
   template< typename Device, typename Index, typename IndexAllocator > class... Segments >
struct SegmentsFormat
{};

// clang-format off
using CSRFormat = SegmentsFormat< Algorithms::Segments::CSR,
                                  Algorithms::Segments::CSR >;
using AdaptiveCSRFormat = SegmentsFormat< Algorithms::Segments::AdaptiveCSR,
                                          Algorithms::Segments::AdaptiveCSR >;
using EllpackFormat = SegmentsFormat< Algorithms::Segments::Ellpack,
                                      Algorithms::Segments::RowMajorEllpack, Algorithms::Segments::ColumnMajorEllpack >;
using SlicedEllpackFormat = SegmentsFormat< Algorithms::Segments::SlicedEllpack,
                                            Algorithms::Segments::RowMajorSlicedEllpack, Algorithms::Segments::ColumnMajorSlicedEllpack >;
using ChunkedEllpackFormat = SegmentsFormat< Algorithms::Segments::ChunkedEllpack,
                                             Algorithms::Segments::RowMajorChunkedEllpack, Algorithms::Segments::ColumnMajorChunkedEllpack >;
using BiEllpackFormat = SegmentsFormat< Algorithms::Segments::BiEllpack,
                                        Algorithms::Segments::RowMajorBiEllpack, Algorithms::Segments::ColumnMajorBiEllpack >;
using SortedCSRFormat = SegmentsFormat< Algorithms::Segments::SortedCSR,
                                        Algorithms::Segments::SortedCSR >;
using SortedAdaptiveCSRFormat = SegmentsFormat< Algorithms::Segments::SortedAdaptiveCSR,
                                                Algorithms::Segments::SortedAdaptiveCSR >;
using SortedEllpackFormat = SegmentsFormat< Algorithms::Segments::SortedEllpack,
                                            Algorithms::Segments::SortedRowMajorEllpack, Algorithms::Segments::SortedColumnMajorEllpack >;
using SortedSlicedEllpackFormat = SegmentsFormat< Algorithms::Segments::SortedSlicedEllpack,
                                                  Algorithms::Segments::SortedRowMajorSlicedEllpack, Algorithms::Segments::SortedColumnMajorSlicedEllpack >;
using SortedChunkedEllpackFormat = SegmentsFormat< Algorithms::Segments::SortedChunkedEllpack,
                                                   Algorithms::Segments::SortedRowMajorChunkedEllpack, Algorithms::Segments::SortedColumnMajorChunkedEllpack >;
using SortedBiEllpackFormat = SegmentsFormat< Algorithms::Segments::SortedBiEllpack,
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

// type list with one segment template for all index types
template< template< typename, typename, typename > class Segments, typename Device, typename... Indices >
struct SegmentsTemplateTypes
{
   using type =
      ::testing::Types< Segments< Device, Indices, typename Allocators::Default< Device >::template Allocator< Indices > >... >;
};

template< typename Format, typename Device, typename... Indices >
struct SegmentsTestTypes;

template<
   template< typename, typename, typename > class DefaultSegments,
   template< typename, typename, typename > class... Segments,
   typename Device,
   typename... Indices >
struct SegmentsTestTypes< SegmentsFormat< DefaultSegments, Segments... >, Device, Indices... >
{
   using type = typename ConcatTypes< typename SegmentsTemplateTypes< Segments, Device, Indices... >::type... >::type;
};

}  // namespace detail

/**
 * \brief Type list with the segments of the given format for the given device and index types.
 *
 * The types are ordered by the segment templates of the format and then by the index types,
 * e.g. `SegmentsTestTypes< EllpackFormat, Devices::Host, int, long >` gives
 * `::testing::Types< RowMajorEllpack< Host, int >, RowMajorEllpack< Host, long >,
 * ColumnMajorEllpack< Host, int >, ColumnMajorEllpack< Host, long > >`.
 */
template< typename Format, typename Device, typename... Indices >
using SegmentsTestTypes = typename detail::SegmentsTestTypes< Format, Device, Indices... >::type;

}  // namespace TNL::UnitTests
