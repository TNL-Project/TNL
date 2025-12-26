// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

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
 * \section SegmentReductionLambdas Reduction Lambda Functions
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
 * \section SegmentKeeperLambdas Keeper Lambda Functions
 *
 * The \e keeper lambda is used to store the final reduction result for each segment.
 *
 * \subsection SegmentKeeperLambda_Basic Basic Keeper (Segment Index Only)
 *
 * ```cpp
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e segmentIdx - The index of the segment
 * - \e value - The result of the reduction for this segment
 *
 * \subsection SegmentKeeperLambda_WithLocalIdx Keeper With Local Index (Position Tracking)
 *
 * ```cpp
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
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
 * \subsection SegmentKeeperLambda_WithIndexArray Keeper With Segment Index Array
 *
 * ```cpp
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * **Parameters:**
 * - \e indexOfSegmentIdx - The position within the \e segmentIndexes array (when reducing over a subset of segments)
 * - \e segmentIdx - The actual index of the segment
 * - \e value - The result of the reduction for this segment
 *
 * \subsection SegmentKeeperLambda_WithIndexArrayAndLocalIdx Keeper With Index Array and Local Index
 *
 * ```cpp
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType localIdx, const Value&
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
 * from \ref TNL::Algorithms::Segments::ReductionFunctionObjects or
 * \ref TNL::Algorithms::Segments::ReductionFunctionObjectsWithArgument.
 *
 * When using function objects:
 * - They must provide a static template method \e getIdentity to automatically deduce the identity value
 * - For WithArgument variants, they must be instances of \ref ReductionFunctionObjectsWithArgument
 * - Common examples: \e Min, \e Max, \e Sum, \e Product, \e MinWithArg, \e MaxWithArg
 */

/**
 * \brief Performs parallel reduction within each segment.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   const Value& identity,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegments( const Segments& segments,
                   Fetch&& fetch,
                   Reduction&& reduction,
                   ResultKeeper&& keeper,
                   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                const Value& identity,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegments.cpp
 * \par Output
 * \include SegmentsExample_reduceSegments.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegments( const Segments& segments,
                IndexBegin begin,
                IndexEnd end,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_WithIndexArray.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexes.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexes.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                const Value& identity,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_WithIndexArray.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexes.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexes.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                Fetch&& fetch,
                Reduction&& reduction,
                ResultKeeper&& keeper,
                LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegmentsIf( const Segments& segments,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultKeeper&& keeper,
                     const Value& identity,
                     LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegmentsIf( const Segments& segments,
                     Condition&& condition,
                     Fetch&& fetch,
                     Reduction&& reduction,
                     ResultKeeper&& keeper,
                     LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation. See \ref SegmentReductionLambda_Basic.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  ResultKeeper&& keeper,
                  const Value& identity,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_Basic.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIf.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIf.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegmentsIf( const Segments& segments,
                  IndexBegin begin,
                  IndexEnd end,
                  Condition&& condition,
                  Fetch&& fetch,
                  Reduction&& reduction,
                  ResultKeeper&& keeper,
                  LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation with argument tracking. See \ref
 * SegmentReductionLambda_WithArgument.
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               const Value& identity,
                               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegmentsWithArgument( const Segments& segments,
                               Fetch&& fetch,
                               Reduction&& reduction,
                               ResultKeeper&& keeper,
                               LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation with argument tracking. See \ref
 * SegmentReductionLambda_WithArgument.
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            const Value& identity,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithArgument.out
 */
template< typename Segments, typename IndexBegin, typename IndexEnd, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            IndexBegin begin,
                            IndexEnd end,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction with argument tracking. See \ref SegmentReductionLambda_WithArgument.
 * \param keeper Lambda function for storing results. See \ref SegmentKeeperLambda_WithIndexArrayAndLocalIdx.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            const Value& identity,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results with index array and local index. See \ref
 * SegmentKeeperLambda_WithIndexArrayAndLocalIdx.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.out
 */
template< typename Segments,
          typename Array,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value > >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            Fetch&& fetch,
                            Reduction&& reduction,
                            ResultKeeper&& keeper,
                            LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Lambda function for reduction operation with argument tracking. See \ref
 * SegmentReductionLambda_WithArgument.
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIfWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIfWithArgument.out
 */
template< typename Segments,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType >
static void
reduceAllSegmentsIfWithArgument( const Segments& segments,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultKeeper&& keeper,
                                 const Value& identity,
                                 LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIfWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIfWithArgument.out
 */
template< typename Segments, typename Condition, typename Fetch, typename Reduction, typename ResultKeeper >
static void
reduceAllSegmentsIfWithArgument( const Segments& segments,
                                 Condition&& condition,
                                 Fetch&& fetch,
                                 Reduction&& reduction,
                                 ResultKeeper&& keeper,
                                 LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
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
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 *
 * \param identity The initial value for the reduction operation.
 *                 If the \e Reduction type does not provide a static member function
 *                 template \e getIdentity, this value must be supplied explicitly by the user.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIfWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIfWithArgument.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegmentsIfWithArgument( const Segments& segments,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              ResultKeeper&& keeper,
                              const Value& identity,
                              LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments where the reduction will be performed.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param end The end of the interval [ \e begin, \e end ) of segments where the reduction
 *    will be performed.
 * \param condition Lambda function for condition checking. See \ref SegmentConditionLambda.
 * \param fetch Lambda function for fetching data. See \ref SegmentFetchLambda_Full or \ref SegmentFetchLambda_Brief.
 * \param reduction Function object for reduction operation with argument tracking. See \ref SegmentReductionFunctionObjects.
 * \param keeper Lambda function for storing results with local index. See \ref SegmentKeeperLambda_WithLocalIdx.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsIfWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsIfWithArgument.out
 */
template< typename Segments,
          typename IndexBegin,
          typename IndexEnd,
          typename Condition,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T = std::enable_if_t< std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegmentsIfWithArgument( const Segments& segments,
                              IndexBegin begin,
                              IndexEnd end,
                              Condition&& condition,
                              Fetch&& fetch,
                              Reduction&& reduction,
                              ResultKeeper&& keeper,
                              LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs a complete reduction over all segment reduction results with separate operations.
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
 * \param segmentFetch A lambda function for fetching data within segments.
 * \param segmentReduction A function object defining the reduction operation within segments.
 * \param finalFetch A lambda function for fetching segment results.
 * \param finalReduction A function object defining the reduction operation for segment results.
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
 * \param segmentFetch A lambda function for fetching data within segments.
 * \param segmentReduction A function object defining the reduction operation within segments.
 * \param finalFetch A lambda function for fetching segment results.
 * \param finalReduction A function object defining the reduction operation for segment results.
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
 * \param segmentReduction Lambda function for reduction operation within segments. See \ref SegmentReductionLambda_Basic.
 * \param finalFetch Lambda function for fetching segment results (transforms segment reduction values before final reduction).
 * \param finalReduction Lambda function for final reduction operation over all segment results. See \ref
 * SegmentReductionLambda_Basic.
 *
 * \param segmentsReductionIdentity The initial value for the reduction operation within the segments.
 * \param finalReductionIdentity The initial value for the final reduction operation.
 *
 * \return The final reduction result combining all segment reduction results.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceAll.cpp
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
 * \param segmentFetch A lambda function for fetching data within segments.
 * \param segmentReduction A function object defining the reduction operation within segments.
 * \param finalFetch A lambda function for fetching segment results.
 * \param finalReduction A function object defining the reduction operation for segment results.
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
