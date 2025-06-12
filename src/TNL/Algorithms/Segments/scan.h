// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <TNL/Containers/ArrayView.h>
#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Compute inclusive prefix-sum (scan) within specified segments in a range.
 *
 * This function computes inclusive prefix-sum within segments in the range [ \e begin, \e end).
 * Each segment is processed independently using sequential scan. The scan operation
 * is performed based on the provided \e fetch, \e reduce, and \e write functions.
 *
 * \tparam Segments Type of the segments container.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param begin The beginning of the range of segments to scan.
 * \param end The end of the range of segments to scan.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template<
   typename Segments,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduce,
   typename Write,
   typename T = typename std::enable_if< std::is_integral< IndexBegin >::value && std::is_integral< IndexEnd >::value >::type >
void
inclusiveScanSegments( const Segments& segments,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute exclusive prefix-sum (scan) within specified segments in a range.
 *
 * This function computes exclusive prefix-sum within segments in the range [ \e begin, \e end).
 * Each segment is processed independently using sequential scan. The scan operation
 * is performed based on the provided \e fetch, \e reduce, and \e write functions.
 *
 * \tparam Segments Type of the segments container.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param begin The beginning of the range of segments to scan.
 * \param end The end of the range of segments to scan.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template<
   typename Segments,
   typename IndexBegin,
   typename IndexEnd,
   typename Fetch,
   typename Reduce,
   typename Write,
   typename T = typename std::enable_if< std::is_integral< IndexBegin >::value && std::is_integral< IndexEnd >::value >::type >
void
exclusiveScanSegments( const Segments& segments,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute inclusive prefix-sum (scan) within all segments.
 *
 * This is a convenience function that computes inclusive prefix-sum in all segments. It internally
 * calls \e inclusiveScanSegments with the full range of segments.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments, typename Fetch, typename Reduce, typename Write >
void
inclusiveScanAllSegments( const Segments& segments,
                          Fetch&& fetch,
                          Reduce&& reduce,
                          Write&& write,
                          LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute exclusive prefix-sum (scan) within all segments.
 *
 * This is a convenience function that computes exclusive prefix-sum in all segments. It internally
 * calls \e exclusiveScanSegments with the full range of segments.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments, typename Fetch, typename Reduce, typename Write >
void
exclusiveScanAllSegments( const Segments& segments,
                          Fetch&& fetch,
                          Reduce&& reduce,
                          Write&& write,
                          LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute inclusive prefix-sum (scan) within specified segments using a segment index array and a range.
 *
 * This function computes inclusive prefix-sum within segments specified by the \e segmentIndexes array and the range [ \e
 * begin, \e end). Each specified segment is processed independently using sequential scan.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Array Type of the segment indexes array.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param segmentIndexes Array containing indices of segments to scan.
 * \param begin The beginning of the range in segmentIndexes.
 * \param end The end of the range in segmentIndexes.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduce,
          typename Write,
          typename T = typename std::enable_if_t< TNL::IsArrayType< Array >::value
                                                  && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
inclusiveScanSegments( const Segments& segments,
                       const Array& segmentIndexes,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute exclusive prefix-sum (scan) within specified segments using a segment index array and a range.
 *
 * This function computes exclusive prefix-sum within segments specified by the \e segmentIndexes array and the range [ \e
 * begin, \e end). Each specified segment is processed independently using sequential scan.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Array Type of the segment indexes array.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param segmentIndexes Array containing indices of segments to scan.
 * \param begin The beginning of the range in segmentIndexes.
 * \param end The end of the range in segmentIndexes.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduce,
          typename Write,
          typename T = typename std::enable_if_t< TNL::IsArrayType< Array >::value
                                                  && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
exclusiveScanSegments( const Segments& segments,
                       const Array& segmentIndexes,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute inclusive prefix-sum (scan) within segments specified by a segment index array.
 *
 * This is a convenience function that computes inclusive prefix-sum in segments specified by the \e segmentIndexes array.
 * It internally calls \e inclusiveScanSegments with the full range of the segment index array.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Array Type of the segment indexes array.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param segmentIndexes Array containing indices of segments to scan.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments, typename Array, typename Fetch, typename Reduce, typename Write >
void
inclusiveScanSegments( const Segments& segments,
                       const Array& segmentIndexes,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute exclusive prefix-sum (scan) within all segments specified by a segment index array.
 *
 * This is a convenience function that computes exclusive prefix-sum in all segments specified by the \e segmentIndexes array.
 * It internally calls \e exclusiveScanSegments with the full range of the segment index array.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Array Type of the segment indexes array.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param segmentIndexes Array containing indices of segments to scan.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments, typename Array, typename Fetch, typename Reduce, typename Write >
void
exclusiveScanSegments( const Segments& segments,
                       const Array& segmentIndexes,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute inclusive conditional prefix-sum (scan) within specified segments in a range.
 *
 * This function computes inclusive prefix-sum within segments in the range [ \e begin, \e end),
 * but only for elements that satisfy the given condition. Each segment is processed
 * independently using sequential scan.
 *
 * \tparam Segments Type of the segments container.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Condition Type of the condition function.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param begin The beginning of the range of segments to scan.
 * \param end The end of the range of segments to scan.
 * \param condition Function that returns true for elements to include in scan.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduce,
          typename Write,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
inclusiveScanSegmentsIf( const Segments& segments,
                         IndexBegin begin,
                         IndexEnd end,
                         Condition&& condition,
                         Fetch&& fetch,
                         Reduce&& reduce,
                         Write&& write,
                         LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute exclusive conditional prefix-sum (scan) within specified segments in a range.
 *
 * This function computes exclusive prefix-sum within segments in the range [ \e begin, \e end),
 * but only for elements that satisfy the given condition. Each segment is processed
 * independently using sequential scan.
 *
 * \tparam Segments Type of the segments container.
 * \tparam IndexBegin Type of the begin index.
 * \tparam IndexEnd Type of the end index.
 * \tparam Condition Type of the condition function.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param begin The beginning of the range of segments to scan.
 * \param end The end of the range of segments to scan.
 * \param condition Function that returns true for elements to include in scan.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduce,
          typename Write,
          typename T = typename std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
exclusiveScanSegmentsIf( const Segments& segments,
                         IndexBegin begin,
                         IndexEnd end,
                         Condition&& condition,
                         Fetch&& fetch,
                         Reduce&& reduce,
                         Write&& write,
                         LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute inclusive conditional prefix-sum (scan) within all segments.
 *
 * This is a convenience function that computes inclusive prefix-sum in all segments,
 * but only for elements that satisfy the given condition. It internally calls
 * \e inclusiveScanSegmentsIf with the full range of segments.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Condition Type of the condition function.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param condition Function that returns true for elements to include in scan. Should have signature:
 *        `bool condition(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments, typename Condition, typename Fetch, typename Reduce, typename Write >
void
inclusiveScanAllSegmentsIf( const Segments& segments,
                            Condition&& condition,
                            Fetch&& fetch,
                            Reduce&& reduce,
                            Write&& write,
                            LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute exclusive conditional prefix-sum (scan) within all segments.
 *
 * This is a convenience function that computes exclusive prefix-sum in all segments,
 * but only for elements that satisfy the given condition. It internally calls
 * \e exclusiveScanSegmentsIf with the full range of segments.
 *
 * \tparam Segments Type of the segments container.
 * \tparam Condition Type of the condition function.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a function object performing the reduction, some \ref ReductionFunctionObjects.
 * \tparam Write Type of the write function.
 *
 * \param segments The segments container.
 * \param condition Function that returns true for elements to include in scan. Should have signature:
 *        `bool condition(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param fetch Function to fetch element value at given position. Should have signature:
 *        `ValueType fetch(IndexType segmentIdx, IndexType localIdx, IndexType globalIdx)`.
 * \param reduce Function object performing the reduction.
 * \param write Function to write result at given position. The implementation allows inplace scan (the write function may
 * modify the input array) Should have signature:
 *        `void write(IndexType globalIdx, ValueType value)`.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_scan.cpp
 *
 * \par Output
 * \include SegmentsExample_scan.out
 */
template< typename Segments, typename Condition, typename Fetch, typename Reduce, typename Write >
void
exclusiveScanAllSegmentsIf( const Segments& segments,
                            Condition&& condition,
                            Fetch&& fetch,
                            Reduce&& reduce,
                            Write&& write,
                            LaunchConfiguration launchConfig = LaunchConfiguration() );

}  // namespace TNL::Algorithms::Segments

#include "scan.hpp"
