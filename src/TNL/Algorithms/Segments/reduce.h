// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

/**
 * \page SegmentReductionOverview Overview of Segment Reduction Functions
 *
 * \tableofcontents
 *
 * This page provides an overview of all reduction functions available for segment operations,
 * helping to understand the differences between variants and choose the right function for your needs.
 *
 * \section SegmentReductionFunctionCategories Function Categories
 *
 * The segment reduction functions are organized along three independent axes:
 *
 * \subsection SegmentReductionBasicVsArgument Basic vs. WithArgument Variants
 *
 * | Category | Tracks Position? | Use Case |
 * |----------|-----------------|----------|
 * | **Basic** | No | Only the reduced value is needed (e.g., sum, product) |
 * | **WithArgument** | Yes | Need both the value and its position (e.g., max value and where it occurs) |
 *
 * \subsection SegmentReductionScopeAndConditionalVariants Scope and Conditional Variants (Which Segments to Process)
 *
 * | Scope | Segments Processed | Parameters |
 * |-------|-------------------|------------|
 * | **All** | All segments | No range/array parameters |
 * | **Range** | Segments [begin, end) | `begin` and `end` indices |
 * | **Array** | Specific segments | Array of segment indices |
 * | **If** | Segment condition | Process segments based on segment-level properties |
 *
 * \section SegmentReductionCompleteMatrix Complete Function Matrix
 *
 * All reduction functions follow this naming pattern:
 * `reduce[Scope][WithArgument][If]`
 *
 * \subsection SegmentReductionBasicFunctions Basic Reduction Functions
 *
 * | Function | Scope | Conditional | Tracks Position |
 * |----------|-------|-------------|-----------------|
 * | \ref Segments_reduceAllSegments | All | No | No |
 * | \ref Segments_reduceSegments_range (range) | Range [begin,end) | No | No |
 * | \ref Segments_reduceSegments_with_segment_indices (array) | Segment array | No | No |
 * | \ref Segments_reduceAllSegmentsIf | All | Yes | No |
 * | \ref Segments_reduceSegmentsIf | Range [begin,end) | Yes | No |
 *
 * \subsection SegmentReductionWithArgumentFunctions WithArgument Reduction Functions
 *
 * | Function | Scope | Conditional | Tracks Position |
 * |----------|-------|-------------|-----------------|
 * | \ref Segments_reduceAllSegmentsWithArgument | All | No | Yes |
 * | \ref Segments_reduceSegmentsWithArgument_range (range) | Range [begin,end) | No | Yes |
 * | \ref Segments_reduceSegmentsWithArgument_with_segment_indices (array) | Segment array | No | Yes |
 * | \ref Segments_reduceAllSegmentsWithArgumentIf | All | Yes | Yes |
 * | \ref Segments_reduceSegmentsWithArgumentIf | Range [begin,end) | Yes | Yes |
 *
 * \section SegmentReductionParameters Common Parameters
 *
 * All reduction functions share these common parameters:
 *
 * - **segments**: The segments container to reduce
 * - **fetch**: Lambda that retrieves element values (see \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief)
 * - **reduction**: Lambda or function object that combines values (see \ref SegmentReductionLambda_Basic or \ref
 * SegmentReductionLambda_WithArgument)
 * - **storer**: Lambda that stores the reduction results (see \ref SegmentStorerLambda_Basic or variants)
 * - **identity**: The identity element for the reduction (e.g., 0 for addition, 1 for multiplication)
 * - **launchConfig**: Configuration for parallel execution (optional)
 *
 * Additional parameters for specific variants:
 * - **Scope variants**: `begin`, `end` (range) or `segmentIndexes` (array)
 * - **If variants**: `condition` lambda for segment filtering (see \ref SegmentConditionLambda)
 *
 * \section SegmentReductionUsageGuidelines Usage Guidelines
 *
 * **Performance considerations:**
 * - Use the brief form of fetch lambda when possible for better performance.
 * - Use `*If` variants to skip unnecessary segments.
 * - Consider using function objects instead of lambda reductions for common operations.
 * - Range and array overloads avoid processing unnecessary segments.
 * - The `fetch` lambda can even modify data on-the-fly if needed and thus it allows merging multiple operations into one
 * kernel and improving performance.
 *
 * \section SegmentReductionRelatedPages Related Pages
 *
 * - \ref SegmentReductionLambdas - Detailed lambda function signatures
 * - \ref ReductionFunctionObjects - Pre-defined reduction function objects
 * - \ref ReductionFunctionObjectsWithArgument - Reduction objects with position tracking
 */

/**
 * \page SegmentReductionLambdas Segment Reduction Lambda Function Reference
 *
 * This page provides a comprehensive reference for all lambda function signatures used
 * in segment reduction operations.
 *
 * \tableofcontents
 *
 * \section SegmentFetchLambdas Fetch Lambda Functions
 *
 * The \e fetch lambda is used to extract and transform values from segment elements during reduction.
 *
 * \subsection SegmentFetchLambda_Full Full Form (With All Parameters)
 *
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment
 * - \e localIdx - The rank (position) of the element within the segment
 * - \e globalIdx - The global index of the element in the corresponding container
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * \subsection SegmentFetchLambda_Brief Brief Form (Global Index Only)
 *
 * In case when only the global index is needed to fetch the value, the brief form can be used.
 * It often leads to better performance.
 *
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) -> FetchValue { ... }
 * ```
 *
 * **Parameters:**
 * - \e globalIdx - The global index of the element in the corresponding container
 * - Returns: A value of type \e FetchValue to be used in the reduction
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \section SegmentReductionLambdaFunctions Reduction Lambda Functions
 *
 * The \e reduction lambda defines how values are combined during the reduction operation.
 *
 * \subsection SegmentReductionLambda_Basic Basic Reduction (Without Arguments)
 *
 * ```cpp
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * **Parameters:**
 * - \e a - First value to be reduced
 * - \e b - Second value to be reduced
 * - Returns: The result of reducing \e a and \e b
 *
 * \subsection SegmentReductionLambda_WithArgument Reduction With Argument (Position Tracking)
 *
 * ```cpp
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) -> Result { ... }
 * ```
 *
 * **Parameters:**
 * - \e a - First value to be reduced (mutable reference)
 * - \e b - Second value to be reduced (const reference)
 * - \e aIdx - Index/position associated with value \e a (mutable reference for tracking)
 * - \e bIdx - Index/position associated with value \e b (const reference)
 *
 * Note: This variant is used when you need to track which element produced the final result
 * (e.g., finding the maximum value and its position within the segment).
 *
 * \section SegmentStorerLambdas Storer Lambda Functions
 *
 * The \e storer lambda is used to store the final reduction result for each segment.
 *
 * \subsection SegmentStorerLambda_Basic Basic Storer (Segment Index Only)
 *
 * ```cpp
 * auto storer = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment
 * - \e value - The result of the reduction for this segment
 *
 * \subsection SegmentStorerLambda_WithLocalIdx Storer With Local Index (Position Tracking)
 *
 * ```cpp
 * auto storer = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment
 * - \e localIdx - The local index (position) of the element within the segment that produced the final result
 * - \e value - The result of the reduction for this segment
 *
 * Note: This variant is typically used with \ref SegmentReductionLambda_WithArgument to track both
 * the value and its position within the segment.
 *
 * \subsection SegmentStorerLambda_WithIndexArray Storer With Segment Index Array
 *
 * ```cpp
 * auto storer = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e indexOfSegmentIdx - The position within the \e segmentIndexes array (when reducing over a subset of segments)
 * - \e segmentIdx - The actual index of the segment
 * - \e value - The result of the reduction for this segment
 *
 * \subsection SegmentStorerLambda_WithIndexArrayAndLocalIdx Storer With Index Array and Local Index
 *
 * ```cpp
 * auto storer = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType localIdx, const Value&
 * value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e indexOfSegmentIdx - The position within the \e segmentIndexes array
 * - \e segmentIdx - The actual index of the segment
 * - \e localIdx - The local index (position) of the element within the segment
 * - \e value - The result of the reduction for this segment
 *
 * \section SegmentConditionLambdas Condition Lambda Functions
 *
 * The \e condition lambda determines which segments should be processed (used in "If" variants).
 *
 * \subsection SegmentConditionLambda Condition Check
 *
 * ```cpp
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment
 * - Returns: \e true if the segment should be processed, \e false otherwise
 *
 * \section SegmentReductionFunctionObjects Reduction Function Objects
 *
 * Instead of lambda functions, reduction operations can also be specified using function objects
 * from \ref ReductionFunctionObjects or
 * \ref ReductionFunctionObjectsWithArgument.
 *
 * When using function objects:
 * - They must provide a static template method \e getIdentity to automatically deduce the identity value
 * - For WithArgument variants, they must be instances of \ref ReductionFunctionObjectsWithArgument
 * - Common examples: \e Min, \e Max, \e Sum, \e Product, \e MinWithArg, \e MaxWithArg
 *
 * \section SegmentReductionLambdasRelatedPages Related Pages
 *
 * - \ref SegmentReductionOverview - Overview of segment reduction functions
 */

/**
 * \brief Performs parallel reduction within each segment.
 * \anchor Segments_reduceAllSegments_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultStorer&& storer,
                   const Value& identity,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment.
 * \anchor Segments_reduceAllSegments
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments, typename Fetch, typename Reduction, typename ResultStorer >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultStorer&& storer,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes.
 * \anchor Segments_reduceSegments_range_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultStorer&& storer,
                const Value& identity,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes.
 * \anchor Segments_reduceSegments_range
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultStorer&& storer,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes.
 * \anchor Segments_reduceSegments_with_segment_indices_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_WithIndexArray.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexes.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexes.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultStorer&& storer,
                const Value& identity,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes.
 * \anchor Segments_reduceSegments_with_segment_indices
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_WithIndexArray.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexes.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexes.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultStorer&& storer,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 * \anchor Segments_reduceAllSegmentsIf_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static typename Segments::IndexType
reduceAllSegmentsIf( const Segments& segments,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultStorer&& storer,
                     const Value& identity,
                     LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 * \anchor Segments_reduceAllSegmentsIf
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultStorer >
static typename Segments::IndexType
reduceAllSegmentsIf( const Segments& segments,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultStorer&& storer,
                     LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 * \anchor Segments_reduceSegmentsIf_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static typename Segments::IndexType
reduceSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  ResultStorer&& storer,
                  const Value& identity,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 * \anchor Segments_reduceSegmentsIf
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_Basic.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static typename Segments::IndexType
reduceSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  ResultStorer&& storer,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 * \anchor Segments_reduceAllSegmentsWithArgument_identity
 *
 * See also: \ref SegmentReductionOverview
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation with argument tracking. See \ref
 * SegmentReductionLambda_WithArgument.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultStorer&& storer,
                               const Value& identity,
                               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 * \anchor Segments_reduceAllSegmentsWithArgument
 *
 * See also: \ref SegmentReductionOverview
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments, typename Fetch, typename Reduction, typename ResultStorer >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultStorer&& storer,
                               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 * \anchor Segments_reduceSegmentsWithArgument_range_identity
 *
 * See also: \ref SegmentReductionOverview
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation with argument tracking. See \ref
 * SegmentReductionLambda_WithArgument.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultStorer&& storer,
                            const Value& identity,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 *  returning also the position of the element of interest.
 * \anchor Segments_reduceSegmentsWithArgument_range
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename ResultStorer >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultStorer&& storer,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes while
 *  returning also the position of the element of interest.
 * \anchor Segments_reduceSegmentsWithArgument_with_segment_indices_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction with argument tracking. See \ref SegmentReductionLambda_WithArgument.
 * \param storer Lambda function for storing results. See \ref SegmentStorerLambda_WithIndexArrayAndLocalIdx.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultStorer&& storer,
                            const Value& identity,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes while
 *  returning also the position of the element of interest.
 * \anchor Segments_reduceSegmentsWithArgument_with_segment_indices
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results with index array and local index. See \ref
 * SegmentStorerLambda_WithIndexArrayAndLocalIdx.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultStorer&& storer,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 * \anchor Segments_reduceAllSegmentsWithArgumentIf_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation with argument tracking. See \ref
 * SegmentReductionLambda_WithArgument.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgumentIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgumentIf.out
 */
template< typename Segments,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static typename Segments::IndexType
reduceAllSegmentsWithArgumentIf( const Segments& segments,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultStorer&& storer,
                                 const Value& identity,
                                 LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 * \anchor Segments_reduceAllSegmentsWithArgumentIf
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgumentIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgumentIf.out
 */
template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultStorer >
static typename Segments::IndexType
reduceAllSegmentsWithArgumentIf( const Segments& segments,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultStorer&& storer,
                                 LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 * \anchor Segments_reduceSegmentsWithArgumentIf_identity
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation with argument tracking. See \ref
 * SegmentReductionLambda_WithArgument.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgumentIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgumentIf.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static typename Segments::IndexType
reduceSegmentsWithArgumentIf( const Segments& segments,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              ResultStorer&& storer,
                              const Value& identity,
                              LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 * \anchor Segments_reduceSegmentsWithArgumentIf
 *
 * See also: \ref SegmentReductionOverview
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param storer Lambda function for storing results with local index. See \ref SegmentStorerLambda_WithLocalIdx.
 *
 * \return The number of segments that were processed (i.e., for which the condition was true).
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgumentIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgumentIf.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultStorer,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static typename Segments::IndexType
reduceSegmentsWithArgumentIf( const Segments& segments,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              ResultStorer&& storer,
                              LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs a complete reduction over all segment reduction results with separate operations.
 * \anchor Segments_reduceAll_with_identities
 *
 * This function first performs reductions within each segment using segmentFetch and segmentReduction,
 * and then reduces all segment results into a single value using resultFetch and resultReduction.
 * This overload uses the identity value from the SegmentReduction type.
 *
 * \tparam Segments The type of the segments.
 * \tparam SegmentFetch The type of the lambda function used for fetching data within segments.
 * \tparam SegmentReduction The type of the reduction operation within segments.
 * \tparam FinalFetch The type of the lambda function used for fetching segment results.
 * \tparam FinalReduction The type of the reduction operation for segment results.
 * \tparam SegmentsReductionValue The type of the reduction result.
 * \tparam FinalReductionValue The type of the final reduction result.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentFetch Lambda function for fetching data within segments. See \ref SegmentFetchLambda_Full or \ref
 * SegmentFetchLambda_Brief.
 * \param segmentReduction Function object defining the reduction operation within segments. See \ref
 * SegmentReductionLambda_Basic.
 * \param finalFetch Lambda function for fetching segment results (transforms segment reduction values before final reduction).
 * \param finalReduction Function object defining the reduction operation for segment results. See \ref
 * SegmentReductionLambda_Basic.
 * \param segmentsReductionIdentity The initial value for the reduction operation within the segments.
 * \param finalReductionIdentity The initial value for the final reduction operation.
 *
 * \return The final reduction result combining all segment reduction results.
 */
template< typename Segments,
          typename SegmentFetch,
          typename SegmentReduction,
          typename FinalFetch,
          typename FinalReduction,
          typename SegmentsReductionValue,
          typename FinalReductionValue >
static FinalReductionValue
reduceAll( const Segments& segments,
           SegmentFetch&& segmentFetch,
           SegmentReduction&& segmentReduction,
           FinalFetch&& finalFetch,
           FinalReduction&& finalReduction,
           const SegmentsReductionValue& segmentsReductionIdentity,
           const FinalReductionValue& finalReductionIdentity,
           LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs a complete reduction over all segment reduction results with separate operations.
 * \anchor Segments_reduceAll
 *
 * This function first performs reductions within each segment using segmentFetch and segmentReduction,
 * and then reduces all segment results into a single value using resultFetch and resultReduction.
 * This overload uses the identity value from the SegmentReduction type.
 *
 * \tparam Segments The type of the segments.
 * \tparam SegmentFetch The type of the lambda function used for fetching data within segments.
 * \tparam SegmentReduction The type of the reduction operation within segments.
 * \tparam FinalFetch The type of the lambda function used for fetching segment results.
 * \tparam FinalReduction The type of the reduction operation for segment results.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentFetch Lambda function for fetching data within segments. See \ref SegmentFetchLambda_Full or \ref
 * SegmentFetchLambda_Brief.
 * \param segmentReduction Function object defining the reduction operation within segments. See \ref
 * SegmentReductionLambda_Basic.
 * \param finalFetch Lambda function for fetching segment results (transforms segment reduction values before final reduction).
 * \param finalReduction Function object defining the reduction operation for segment results. See \ref
 * SegmentReductionLambda_Basic.
 *
 * \return The final reduction result combining all segment reduction results.
 */
template< typename Segments, typename SegmentFetch, typename SegmentReduction, typename FinalFetch, typename FinalReduction >
static typename detail::FetchLambdaAdapter< typename Segments::IndexType, FinalFetch >::ReturnType
reduceAll( const Segments& segments,
           SegmentFetch&& segmentFetch,
           SegmentReduction&& segmentReduction,
           FinalFetch&& finalFetch,
           FinalReduction&& finalReduction,
           LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs a complete reduction over given segment reduction results with separate operations.
 *
 * This function first performs reductions within segments in the interval [begin,end) using segmentFetch and segmentReduction,
 * and then reduces all segment results into a single value using resultFetch and resultReduction.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam SegmentFetch The type of the lambda function used for fetching data within segments.
 * \tparam SegmentReduction The type of the reduction operation within segments.
 * \tparam FinalFetch The type of the lambda function used for fetching segment results.
 * \tparam FinalReduction The type of the reduction operation for segment results.
 * \tparam SegmentsReductionValue The type of the reduction result.
 * \tparam FinalReductionValue The type of the final reduction result.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param segmentFetch Lambda function for fetching data within segments. See \ref SegmentFetchLambda_Full or \ref
 * SegmentFetchLambda_Brief.
 * \param segmentReduction Function object defining the reduction operation within segments. See \ref
 * SegmentReductionLambda_Basic.
 * \param finalFetch Lambda function for fetching segment results (transforms segment reduction values before final reduction).
 * \param finalReduction Function object defining the reduction operation for segment results. See \ref
 * SegmentReductionLambda_Basic.
 *
 * \param segmentsReductionIdentity The initial value for the reduction operation within the segments.
 * \param finalReductionIdentity The initial value for the final reduction operation.
 *
 * \return The final reduction result combining all segment reduction results.
 *
 * \par Example
 * \includelineno Algorithms/Segments/SegmentsExample_reduceAll.cpp
 * \par Output
 * \include SegmentsExample_reduceAll.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentFetch,
          typename SegmentReduction,
          typename FinalFetch,
          typename FinalReduction,
          typename SegmentsReductionValue,
          typename FinalReductionValue,
          typename T = std::enable_if_t< isSegments_v< Segments > > >
static FinalReductionValue
reduce( const Segments& segments,
        IndexBegin begin,
        IndexEnd end,
        SegmentFetch&& segmentFetch,
        SegmentReduction&& segmentReduction,
        FinalFetch&& finalFetch,
        FinalReduction&& finalReduction,
        const SegmentsReductionValue& segmentsReductionIdentity,
        const FinalReductionValue& finalReductionIdentity,
        LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs a complete reduction over given segments reduction results with separate operations.
 *
 * This function first performs reductions within each segment in the interval [begin,end) using segmentFetch and
 * segmentReduction, and then reduces all segment results into a single value using resultFetch and resultReduction. This
 * overload uses the identity value from the SegmentReduction type.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *   of segment indexes where the reduction will be performed.
 * \tparam SegmentFetch The type of the lambda function used for fetching data within segments.
 * \tparam SegmentReduction The type of the reduction operation within segments.
 * \tparam FinalFetch The type of the lambda function used for fetching segment results.
 * \tparam FinalReduction The type of the reduction operation for segment results.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param segmentFetch Lambda function for fetching data within segments. See \ref SegmentFetchLambda_Full or \ref
 * SegmentFetchLambda_Brief.
 * \param segmentReduction Function object defining the reduction operation within segments. See \ref
 * SegmentReductionLambda_Basic.
 * \param finalFetch Lambda function for fetching segment results (transforms segment reduction values before final reduction).
 * \param finalReduction Function object defining the reduction operation for segment results. See \ref
 * SegmentReductionLambda_Basic.
 *
 * \return The final reduction result combining all segment reduction results.
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename SegmentFetch,
          typename SegmentReduction,
          typename FinalFetch,
          typename FinalReduction,
          typename T = std::enable_if_t< isSegments_v< Segments > > >
static typename detail::FetchLambdaAdapter< typename Segments::IndexType, SegmentFetch >::ReturnType
reduce( const Segments& segments,
        IndexBegin begin,
        IndexEnd end,
        SegmentFetch&& segmentFetch,
        SegmentReduction&& segmentReduction,
        FinalFetch&& finalFetch,
        FinalReduction&& finalReduction,
        LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Algorithms::Segments
#include "reduce.hpp"
