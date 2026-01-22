// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
#include <TNL/Containers/ArrayView.h>
#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"

// clang-format off
/**
 * \page SegmentScanOverview Overview of Segment Scan Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all scan (prefix-sum) functions available for segment operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section SegmentScanFunctionCategories Function Categories
 *
 * \subsection SegmentScanBasicFunctions Basic Scan Functions
 *
 * These functions compute scan within specified segments:
 *
 * | Function                                                         | Segments Scanned      | Scan Type |
 * |------------------------------------------------------------------|-----------------------|-----------|
 * | \ref Segments_inclusiveScanAllSegments                           | All segments          | Inclusive |
 * | \ref Segments_exclusiveScanAllSegments                           | All segments          | Exclusive |
 * | \ref Segments_inclusiveScanSegments_range (range)                | Segments [begin, end) | Inclusive |
 * | \ref Segments_exclusiveScanSegments_range (range)                | Segments [begin, end) | Exclusive |
 * | \ref Segments_inclusiveScanSegments_with_segment_indices (array) | Segments in array     | Inclusive |
 * | \ref Segments_exclusiveScanSegments_with_segment_indices (array) | Segments in array     | Exclusive |
 *
 * **When to use:**
 * - Use `*AllSegments` when you need to scan all segments
 * - Use range overload when scanning a contiguous range of segments
 * - Use array overload when you have a specific, non-contiguous set of segment indices
 *
 * \subsection SegmentScanConditionalFunctions Conditional Scan Functions
 *
 * These functions add segment-level conditions, including only segments that satisfy the condition in the scan:
 *
 * | Function                                                         | Segments Scanned      | Scan Type                     |
 * |------------------------------------------------------------------|-----------------------|-------------------------------|
 * | \ref Segments_inclusiveScanAllSegmentsIf                         | All segments          | Inclusive with segment filter |
 * | \ref Segments_exclusiveScanAllSegmentsIf                         | All segments          | Exclusive with segment filter |
 * | \ref Segments_inclusiveScanSegmentsIf_range                      | Segments [begin, end) | Inclusive with segment filter |
 * | \ref Segments_exclusiveScanSegmentsIf_range                      | Segments [begin, end) | Exclusive with segment filter |
 *
 * **When to use:**
 * - Use these when you want to skip certain elements within segments
 * - Example: Compute prefix-sum only for positive values, or only for elements meeting specific criteria
 *
 * \subsection SegmentScanLowLevelFunctions Low-Level Functions
 *
 * | Function                                                         | Purpose                                           |
 * |------------------------------------------------------------------|---------------------------------------------------|
 * | \ref Segments_inclusiveScanSegment                               | Computes inclusive scan for a single segment view |
 * | \ref Segments_exclusiveScanSegment                               | Computes exclusive scan for a single segment view |
 *
 * **When to use:**
 * - Use these when you have a `SegmentView` object and need fine-grained control
 * - Useful when implementing custom segment algorithms
 *
 * \section SegmentScanParameters Common Parameters
 *
 * All scan functions share these common parameters:
 *
 * - **segments**: The segments container to scan
 * - **fetch**: Lambda that retrieves element values (see \ref SegmentScanFetchLambda)
 * - **reduce**: Function object that combines values (see \ref SegmentScanReduction)
 * - **write**: Lambda that stores the scan results (see \ref SegmentScanWriteLambda)
 * - **launchConfig**: Configuration for parallel execution (optional)
 *
 * Conditional variants additionally require:
 * - **condition**: Lambda that determines if an element should be included (see \ref SegmentScanConditionLambda)
 *
 * \section SegmentScanUsageGuidelines Usage Guidelines
 *
 * **In-place vs. out-of-place:**
 * - All scan implementations support in-place operations
 * - Your `write` lambda can modify the same array that `fetch` reads from
 * - For out-of-place scans, write to a different array
 *
 * **Performance considerations:**
 * - Scan operations are sequential within each segment
 * - Different segments are processed in parallel
 * - Use element conditions (`*If` variants) to skip unnecessary elements
 * - Choose appropriate reduction function objects for best performance
 *
 * \section SegmentScanRelatedPages Related Pages
 *
 * - \ref SegmentScanLambdas - Detailed lambda function signatures
 * - \ref ReductionFunctionObjects - Available reduction operations
 */
// clang-format on

/**
 * \page SegmentScanLambdas Lambda Functions for Segment Scan Operations
 *
 * \tableofcontents
 *
 * This page describes the lambda function signatures used in segment scan (prefix-sum) operations.
 *
 * \section SegmentScanFetchLambda Fetch Lambda
 *
 * The fetch lambda retrieves the value of an element for the scan operation.
 * It has one of the following forms:
 *
 * **Full form:**
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx )
 * {
 *    // Return the value at the given position
 *    return ...;
 * };
 * ```
 *
 * **Brief form:**
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx )
 * {
 *    // Return the value at the given position
 *    return ...;
 * };
 * ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * \section SegmentScanReduction Reduction Function Object
 *
 * The scan operation uses a reduction function object to combine values.
 * See \ref ReductionFunctionObjects for available reduction operations like:
 * - `TNL::Plus` - Addition
 * - `TNL::Multiplies` - Multiplication
 * - `TNL::Min` - Minimum
 * - `TNL::Max` - Maximum
 *
 * \section SegmentScanWriteLambda Write Lambda
 *
 * The write lambda stores the result of the scan operation.
 * It has the following signature:
 *
 * ```cpp
 * auto write = [=] __cuda_callable__ ( IndexType globalIdx, ValueType value )
 * {
 *    // Write the value at the given position
 *    output[globalIdx] = value;
 * };
 * ```
 *
 * - \e globalIdx is the index where the result should be written.
 * - \e value is the scan result to write.
 *
 * Note: The implementation allows in-place scan (the write function may modify the input array).
 *
 * \section SegmentScanConditionLambda Condition Lambda
 *
 * The condition lambda determines which elements should be included in the scan operation.
 * It has the following signature:
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> bool
 * {
 *    // Return true if element should be included in scan
 *    return ...;
 * };
 * ```
 *
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * The lambda should return `true` if the element should be included in the scan.
 */

namespace TNL::Algorithms::Segments {

/**
 * \brief Compute inclusive prefix-sum (scan) within all segments.
 * \anchor Segments_inclusiveScanAllSegments
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
 * \anchor Segments_exclusiveScanAllSegments
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
 * \brief Compute inclusive prefix-sum (scan) within specified segments in a range.
 * \anchor Segments_inclusiveScanSegments_range
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
   typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
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
 * \anchor Segments_exclusiveScanSegments_range
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
   typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
exclusiveScanSegments( const Segments& segments,
                       IndexBegin begin,
                       IndexEnd end,
                       Fetch&& fetch,
                       Reduce&& reduce,
                       Write&& write,
                       LaunchConfiguration launchConfig = LaunchConfiguration() );

/**
 * \brief Compute inclusive prefix-sum (scan) within segments specified by a segment index array.
 * \anchor Segments_inclusiveScanSegments_with_segment_indices
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
 * \anchor Segments_exclusiveScanSegments_with_segment_indices
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
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
 * \brief Compute inclusive conditional prefix-sum (scan) within all segments.
 * \anchor Segments_inclusiveScanAllSegmentsIf
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
 * \param condition Function that returns true for elements to include in scan. See \ref SegmentScanConditionLambda.
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
 * \anchor Segments_exclusiveScanAllSegmentsIf
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
 * \param condition Function that returns true for elements to include in scan. See \ref SegmentScanConditionLambda.
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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

/**
 * \brief Computes an inclusive scan (or prefix sum) within a segment.
 * \anchor Segments_inclusiveScanSegment
 *
 * \tparam SegmentView Type of the segment view.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a type of function performing the reduction.
 * \tparam Write Type of the write function.
 *
 * \param segment The segment view to scan.
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 */
template< typename SegmentView, typename Fetch, typename Reduce, typename Write >
__cuda_callable__
void
inclusiveScanSegment( SegmentView& segment, Fetch&& fetch, Reduce&& reduce, Write&& write );

/**
 * \brief Computes an exclusive scan (or prefix sum) within a segment.
 * \anchor Segments_exclusiveScanSegment
 *
 * \tparam SegmentView Type of the segment view.
 * \tparam Fetch Type of the fetch function.
 * \tparam Reduce is a type of function performing the reduction.
 * \tparam Write Type of the write function.
 *
 * \param segment The segment view to scan.
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 */
template< typename SegmentView, typename Fetch, typename Reduce, typename Write >
__cuda_callable__
void
exclusiveScanSegment( SegmentView& segment, Fetch&& fetch, Reduce&& reduce, Write&& write );

/**
 * \brief Compute inclusive conditional prefix-sum (scan) within specified segments in a range.
 * \anchor Segments_inclusiveScanSegmentsIf_range
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
 * \param condition Function that returns true for elements to include in scan. See \ref SegmentScanConditionLambda.
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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
 * \anchor Segments_exclusiveScanSegmentsIf_range
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
 * \param condition Function that returns true for elements to include in scan. See \ref SegmentScanConditionLambda.
 * \param fetch Function to fetch element value at given position. See \ref SegmentScanFetchLambda.
 * \param reduce Function object performing the reduction. See \ref SegmentScanReduction.
 * \param write Function to write result at given position. See \ref SegmentScanWriteLambda.
 * \param launchConfig Configuration for parallel execution.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_scan.cpp
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

}  // namespace TNL::Algorithms::Segments

#include "scan.hpp"
