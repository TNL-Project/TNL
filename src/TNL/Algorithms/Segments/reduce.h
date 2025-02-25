// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <TNL/TypeTraits.h>
#include "LaunchConfiguration.h"
#include "detail/FetchLambdaAdapter.h"

namespace TNL::Algorithms::Segments {

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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced.
 * - The function returns the result of the reduction.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjects.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \brief Performs parallel reduction within each segment.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced.
 * - The function returns the result of the reduction.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjects.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced.
 * - The function returns the result of the reduction.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e indexOfSegmentIdx is the index of the segment within the \e segmentIndexes container.
 * - \e segmentIdx is the actual index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value
                                                  && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
                IndexBegin begin,
                IndexEnd end,
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
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjects.
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e indexOfSegmentIdx is the index of the segment within the \e segmentIndexes container.
 * - \e segmentIdx is the actual index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexes.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexes.out
 */
template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value
                                                  && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegments( const Segments& segments,
                const Array& segmentIndexes,
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced.
 * - The function returns the result of the reduction.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e indexOfSegmentIdx is the index of the segment within the \e segmentIndexes container.
 * - \e segmentIdx is the actual index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjects.
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e indexOfSegmentIdx is the index of the segment within the \e segmentIndexes container.
 * - \e segmentIdx is the actual index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced.
 * - The function returns the result of the reduction.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjects.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \brief Performs parallel reduction within each segment over a given range of segment indexes based on a condition.
 *
 * \tparam Segments The type of the segments.
 * \tparam Condition The type of the lambda function used for the condition check.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced.
 * - The function returns the result of the reduction.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjects.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced. The result of the reduction shall be stored in \e a.
 * - \e aIdx and \e bIdx are the indexes of the elements in the corresponding container. The index of the element
 *   of interest should be stored in \e aIdx.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjectsWithArgument.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \brief Performs parallel reduction within each segment over a given range of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced. The result of the reduction shall be stored in \e a.
 * - \e aIdx and \e bIdx are the indexes of the elements in the corresponding container. The index of the element
 *   of interest should be stored in \e aIdx.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjectsWithArgument.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \brief Performs parallel reduction within segments specified by a given set of segment indexes while
 *  returning also the position of the element of interest.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced. The result of the reduction shall be stored in \e a.
 * - \e aIdx and \e bIdx are the indexes of the elements in the corresponding container. The index of the element
 *   of interest should be stored in \e aIdx.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType localIdx, const Value&
 * value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename Value = typename detail::FetchLambdaAdapter< typename Segments::IndexType, Fetch >::ReturnType,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value
                                                  && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
                            IndexBegin begin,
                            IndexEnd end,
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
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segment indexes where the reduction will be performed.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultKeeper The type of the lambda function used for storing results from individual segments.
 *
 * \param segments The segment data structure on which the reduction will be performed.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed for reduction.
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjectsWithArgument.
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType localIdx, const Value&
 * value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.cpp
 * \par Output
 * \include SegmentsExample_reduceSegmentsWithSegmentIndexesWithArgument.out
 */
template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Fetch,
          typename Reduction,
          typename ResultKeeper,
          typename T = typename std::enable_if_t< IsArrayType< Array >::value
                                                  && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
static void
reduceSegmentsWithArgument( const Segments& segments,
                            const Array& segmentIndexes,
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced. The result of the reduction shall be stored in \e a.
 * - \e aIdx and \e bIdx are the indexes of the elements in the corresponding container. The index of the element
 *   of interest should be stored in \e aIdx.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType localIdx, const Value&
 * value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjectsWithArgument.
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType indexOfSegmentIdx, IndexType segmentIdx, IndexType localIdx, const Value&
 * value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced. The result of the reduction shall be stored in \e a.
 * - \e aIdx and \e bIdx are the indexes of the elements in the corresponding container. The index of the element
 *   of interest should be stored in \e aIdx.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [=] __cuda_callable__ ( const Value& a, const Value& b ) -> Value { ... }
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced.
 * - The function returns the result of the reduction.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A lambda function representing the reduction operation. It should be defined as:
 *
 * ```
 * auto reduction = [] __cuda_callable__ ( Result& a, const Result& b, Index& aIdx, const Index& bIdx ) { return ... };
 * ```
 *
 * where:
 * - \e a and \e b are values to be reduced. The result of the reduction shall be stored in \e a.
 * - \e aIdx and \e bIdx are the indexes of the elements in the corresponding container. The index of the element
 *   of interest should be stored in \e aIdx.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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
 * \param condition A lambda function for checking the condition. It should be defined as:
 * ```
 * auto condition = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool { ... }
 * ```
 *
 * \param fetch A lambda function for fetching data. It should have one of the following forms:
 *
 * 1. **Full form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) { ... }
 *  ```
 * 2. **Brief form**
 *  ```
 *  auto fetch = [=] __cuda_callable__ ( IndexType globalIdx ) { ... }
 *  ```
 *
 * In both variants:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the index of the element in the corresponding container.
 *
 * **Note:** The actual behavior depends on the kernel type used for the reduction. Some optimized kernels
 * may perform significantly better with the brief variant of the \e fetch lambda function.
 *
 * \param reduction A function object defining the reduction operation.
 *   It must be an instance of \ref ReductionFunctionObjectsWithArgument.
 *
 * \param keeper A lambda function for storing the reduction results of individual segments. It should be defined as:
 *
 * ```
 * auto keeper = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, const Value& value ) { ... }
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment.
 * - \e localIdx is the position of the element within the segment.
 * - \e value is the result of the reduction for the given segment that needs to be stored.
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

}  // namespace TNL::Algorithms::Segments

#include "reduce.hpp"
