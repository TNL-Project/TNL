// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "LaunchConfiguration.h"

namespace TNL::Algorithms::Segments {

/**
 * \page SegmentTraversalOverview Overview of Segment Traversal Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all traversal functions available for segment operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section SegmentTraversalFunctionCategories Function Categories
 *
 * Traversal functions are organized along two main dimensions:
 *
 * \subsection SegmentTraversalElementVsSegment Element-wise vs. Segment-wise Traversal
 *
 * | Category | Operates On | Lambda Parameter | Use Case |
 * |----------|------------|------------------|----------|
 * | **Element-wise** (`forElements`, `forAllElements`) | Individual elements | Element indices | Operate on each element
 * separately |
 * | **Segment-wise** (`forSegments`, `forAllSegments`) | Whole segments | SegmentView object | Operate on segments
 * as units |
 *
 * **When to use:**
 * - **Element-wise**: When you need to process each element independently (e.g., apply transformation, update values)
 * - **Segment-wise**: When you need access to the entire segment structure or segment context (e.g., custom
 * segment algorithms, segment-level operations)
 *
 * \subsection SegmentTraversalScopeAndConditional Scope and Conditional Variants
 *
 * Similar to other segment operations, traversal functions come in different scopes and conditional variants.
 *
 * | Scope | Segments Processed | Parameters |
 * |-------|-------------------|------------|
 * | **All** | All segments | No range/array parameters |
 * | **Range** | Segments [begin, end) | `begin` and `end` indices |
 * | **Array** | Specific segments | Array of segment indices |
 * | **If** | Segment condition | Process segments based on segment-level properties |
 *
 * \section SegmentTraversalElementFunctions Element-wise Traversal Functions
 *
 * These functions iterate over individual elements within segments:
 *
 * \subsection SegmentTraversalBasicElementFunctions Basic Element Traversal
 *
 * | Function | Segments Processed | Description |
 * |----------|-------------------|-------------|
 * | \ref forAllElements | All segments | Process all elements in all segments |
 * | \ref forElements (range) | Segments [begin, end) | Process elements in segment range |
 * | \ref forElements (array) | Segments in array | Process elements in specified segments |
 * | \ref forAllElementsIf | All segments | Segment-level condition |
 * | \ref forElementsIf | Segments [begin, end) | Segment-level condition |
 *
 * \section SegmentTraversalSegmentFunctions Segment-wise Traversal Functions
 *
 * These functions iterate over segments as whole units:
 *
 * \subsection SegmentTraversalBasicSegmentFunctions Basic Segment Traversal
 *
 * | Function | Segments Processed | Description |
 * |----------|-------------------|-------------|
 * | \ref forAllSegments | All segments | Process all segments |
 * | \ref forSegments (range) | Segments [begin, end) | Process segments in range |
 * | \ref forSegments (array) | Segments in array | Process specified segments |
 * | \ref forAllSegmentsIf | All segments | Segment-level condition |
 * | \ref forSegmentsIf | Segments [begin, end) | Segment-level condition |
 *
 * \section SegmentTraversalParameters Common Parameters
 *
 * All traversal functions share these common parameters:
 *
 * - **segments**: The segments container to traverse
 * - **f**: Lambda function to apply (see \ref SegmentElementLambda_Full, \ref SegmentElementLambda_Brief, or \ref
 * SegmentViewLambda)
 * - **launchConfig**: Configuration for parallel execution (optional)
 *
 * Additional parameters for specific variants:
 * - **Scope variants**: `begin`, `end` (range) or `segmentIndexes` (array)
 * - **If variants**: `condition` lambda for filtering (see \ref SegmentConditionLambda)
 *
 * \section SegmentTraversalRelatedPages Related Pages
 *
 * - \ref SegmentTraversalLambdas - Detailed lambda function signatures
 * - \ref TNL::Algorithms::Segments::SegmentView - Segment view documentation
 */

/**
 * \page SegmentTraversalLambdas Segment Traversal Lambda Function Reference
 *
 * This page provides a comprehensive reference for all lambda function signatures used
 * in segment traversal operations.
 *
 * \tableofcontents
 *
 * \section SegmentElementLambdas Element Traversal Lambda Functions
 *
 * These lambda functions are used when iterating over individual elements within segments.
 *
 * \subsection SegmentElementLambda_Full Full Form (With All Parameters)
 *
 * ```cpp
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
 * ```
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment to which the given element belongs
 * - \e localIdx - The rank (position) of the element within the segment
 * - \e globalIdx - The global index of the element within the range of all elements managed by the segments
 *
 * \subsection SegmentElementLambda_Brief Brief Form (Without Local Index)
 *
 * ```cpp
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType globalIdx ) {...}
 * ```
 *
 * In case when the local index within the segment is not required, this brief form can be used. It may
 * lead to better performance.
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment to which the given element belongs
 * - \e globalIdx - The global index of the element within the range of all elements managed by the segments
 *
 * \section SegmentViewLambdas Segment View Lambda Functions
 *
 * These lambda functions are used when iterating over segments as a whole, operating on SegmentView objects.
 *
 * \subsection SegmentViewLambda Segment View Lambda
 *
 * ```cpp
 * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
 * ```
 *
 * **Parameters:**
 * - \e segment - A view representing the given segment (see \ref TNL::Algorithms::Segments::SegmentView).
 *   The segment view provides access to segment properties and its elements.
 *
 * \section SegmentConditionLambdas Segment Condition Lambda Functions
 *
 * These lambda functions are used to determine whether a segment should be processed (used in "If" variants).
 *
 * \subsection SegmentConditionLambda Condition Check
 *
 * ```cpp
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool {...}
 * ```
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment
 * - Returns: \e true if the segment (or its elements) should be processed, \e false otherwise
 *
 * \section SegmentTraversalLambdasRelatedPages Related Pages
 *
 * - \ref SegmentTraversalOverview - Overview of segment traversal functions
 */

/**
 * \brief Iterates in parallel over all elements of **all** segments and applies the specified lambda function.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param function The lambda function to be applied to each element. See \ref SegmentElementLambda_Full or \ref
 * SegmentElementLambda_Brief.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forElements.cpp
 * \par Output
 * \include SegmentsExample_forElements.out
 */
template< typename Segments, typename Function >
void
forAllElements( const Segments& segments,
                Function&& function,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements in the given range of segments and applies the specified lambda function.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments whose elements we want to process using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments whose elements we want to process using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of segments whose elements
 *    will be processed using the lambda function.
 * \param function The lambda function to be applied to each element. See \ref SegmentElementLambda_Full or \ref
 * SegmentElementLambda_Brief.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forElements.cpp
 * \par Output
 * \include SegmentsExample_forElements.out
 */
template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of segments with the given indexes and applies the specified lambda function.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param function The lambda function to be applied to each element. See \ref SegmentElementLambda_Full or \ref
 * SegmentElementLambda_Brief.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forElementsWithIndexes.cpp
 * \par Output
 * \include SegmentsExample_forElementsWithIndexes.out
 */
template< typename Segments, typename Array, typename Function >
void
forElements( const Segments& segments,
             const Array& segmentIndexes,
             Function function,
             LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of **all** segments based on a condition.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * For each segment, a condition lambda function is evaluated based on the segment index.
 * If the condition lambda function returns \e true, all elements of the segment are traversed,
 * and the specified lambda function is applied to each element. If the condition lambda function returns
 * \e false, the segment is skipped.
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param function The lambda function to be applied to each element. See \ref SegmentElementLambda_Full or \ref
 * SegmentElementLambda_Brief.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forElementsIf.cpp
 * \par Output
 * \include SegmentsExample_forElementsIf.out
 */
template< typename Segments, typename Condition, typename Function >
void
forAllElementsIf( const Segments& segments,
                  Condition condition,
                  Function function,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements in a given range of segments based on a condition.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * For each segment, a condition lambda function is evaluated based on the segment index.
 * If the condition lambda function returns \e true, all elements of the segment are traversed,
 * and the specified lambda function is applied to each element. If the condition lambda function returns
 * \e false, the segment is skipped.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments whose elements will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments whose elements will be processed using the lambda function.
 * \tparam Condition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of segments whose elements
 *    will be processed using the lambda function.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param function The lambda function to be applied to each element. See \ref SegmentElementLambda_Full or \ref
 * SegmentElementLambda_Brief.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forElementsIf.cpp
 * \par Output
 * \include SegmentsExample_forElementsIf.out
 */
template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Segments& segments,
               IndexBegin begin,
               IndexEnd end,
               Condition condition,
               Function function,
               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIfSparse( const Segments& segments,
                        Condition condition,
                        Function function,
                        LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIfSparse( const Segments& segments,
                     IndexBegin begin,
                     IndexEnd end,
                     Condition condition,
                     Function function,
                     LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over **all** segments and applies the given lambda function to each segment.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param function The lambda function to be applied to each segment. See \ref SegmentViewLambda.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forSegments-1.cpp
 * \par Output
 * \include SegmentsExample_forSegments-1.out
 *
 * \include Algorithms/Segments/SegmentsExample_forSegments-2.cpp
 * \par Output
 * \include SegmentsExample_forSegments-2.out
 */
template< typename Segments, typename Function >
void
forAllSegments( const Segments& segments,
                Function&& function,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over segments within the specified range of segment indexes
 * and applies the given lambda function to each segment.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments
 *    that will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of segments
 *    that will be processed using the lambda function.
 * \param function The lambda function to be applied to each segment. See \ref SegmentViewLambda.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forSegments-1.cpp
 * \par Output
 * \include SegmentsExample_forSegments-1.out
 *
 * \include Algorithms/Segments/SegmentsExample_forSegments-2.cpp
 * \par Output
 * \include SegmentsExample_forSegments-2.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forSegments( const Segments& segments,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over segments with the given indexes and applies the specified
 * lambda function to each segment.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param function The lambda function to be applied to each segment. See \ref SegmentViewLambda.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forSegmentsWithIndexes.cpp
 * \par Output
 * \include SegmentsExample_forSegmentsWithIndexes.out
 */
template< typename Segments, typename Array, typename Function, typename T = std::enable_if_t< IsArrayType< Array >::value > >
void
forSegments( const Segments& segments,
             const Array& segmentIndexes,
             Function&& function,
             LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );
/**
 * \brief Iterates in parallel over **all** segments, applying a condition
 * to determine whether each segment should be processed.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * For each segment, a condition lambda function is evaluated based on the segment index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the segment.
 * If the condition lambda function returns \e false, the segment is skipped.
 *
 * \tparam Segments The type of the segments.
 * \tparam SegmentCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param segmentCondition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param function The lambda function to be applied to each segment. See \ref SegmentViewLambda.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_forSegmentsIf.out
 */
template< typename Segments, typename SegmentCondition, typename Function >
void
forAllSegmentsIf( const Segments& segments,
                  SegmentCondition&& segmentCondition,
                  Function&& function,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over segments within the given range of segment indexes, applying a condition
 * to determine whether each segment should be processed.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * For each segment, a condition lambda function is evaluated based on the segment index.
 * If the condition lambda function returns \e true, the specified lambda function is executed for the segment.
 * If the condition lambda function returns \e false, the segment is skipped.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments on which the lambda function will be applied.
 * \tparam SegmentCondition The type of the condition lambda function.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed using the lambda function.
 * \param segmentCondition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param function The lambda function to be applied to each segment. See \ref SegmentViewLambda.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_forSegmentsIf.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentCondition,
          typename Function,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forSegmentsIf( const Segments& segments,
               IndexBegin begin,
               IndexEnd end,
               SegmentCondition&& segmentCondition,
               Function&& function,
               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

// TODO: Sequential variants should be achieved via LaunchConfiguration
/**
 * \brief Iterates sequentially over segments in given range of segment indexes and call given lambda function for each segment.
 *
 * This function is just a sequential variant of \ref TNL::Algorithms::Segments::forSegments.
 */
template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
sequentialForSegments( const Segments& segments, IndexBegin begin, IndexEnd end, Function&& function );

// TODO: Sequential variants should be achieved via LaunchConfiguration
/**
 * \brief Iterates in parallel over **all** segments and call given lambda function for each segment.
 *
 * See also: \ref SegmentTraversalOverview
 *
 * This function is just a sequential variant of \ref TNL::Algorithms::Segments::forAllSegments.
 */
template< typename Segments, typename Function >
void
sequentialForAllSegments( const Segments& segments, Function&& function );

}  // namespace TNL::Algorithms::Segments

#include "traverse.hpp"
