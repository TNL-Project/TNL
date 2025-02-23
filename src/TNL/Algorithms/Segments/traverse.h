// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "LaunchConfiguration.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Iterates in parallel over all elements in the given range of segments and
 * applies the specified lambda function.
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
 * \param function The lambda function to be applied to each element in the specified range of segments.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as either:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
 * ```
 * or
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType globalIdx ) {...}
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment to which the given element belongs.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the global index of the element within the range of all elements managed by the segments.
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
 * \brief Iterates in parallel over all elements of **all** segments and
 * applies the specified lambda function.
 *
 * \tparam Segments The type of the segments.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param function The lambda function to be applied to each element in the specified range of segments.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as either:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
 * ```
 * or
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType globalIdx ) {...}
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment to which the given element belongs.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the global index of the element within the range of all elements managed by the segments.
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
 * \brief Iterates in parallel over all elements of segments with the given indexes and
 * applies the specified lambda function.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segment indexes whose elements will be processed using the lambda function.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segment indexes whose elements will be processed using the lambda function.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes whose elements
 *    will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes whose elements
 *    will be processed using the lambda function.
 * \param function The lambda function to be applied to each element in the specified segments.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as either:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
 * ```
 * or
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType globalIdx ) {...}
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment to which the given element belongs.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the global index of the element within the range of all elements managed by the segments.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forElementsWithIndexes.cpp
 * \par Output
 * \include SegmentsExample_forElementsWithIndexes.out
 */
template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments& segments,
             const Array& segmentIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over all elements of segments with the given indexes and
 * applies the specified lambda function.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be applied to each element.
 *
 * \param segments The segments whose elements will be processed using the lambda function.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param function The lambda function to be applied to each element in the specified segments.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as either:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
 * ```
 * or
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType globalIdx ) {...}
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment to which the given element belongs.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the global index of the element within the range of all elements managed by the segments.
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
 * \brief Iterates in parallel over all elements in a given range of segments based on a condition.
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
 * \param condition The lambda function that determines whether a segment should be processed.
 * \param function The lambda function to be applied to each element in the selected segments.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e condition should be declared as:
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool {...}
 * ```
 * where \e segmentIdx is the index of the segment.
 *
 * The lambda function \e function should be declared as either:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
 * ```
 * or
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType globalIdx ) {...}
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment to which the given element belongs.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the global index of the element within the range of all elements managed by the segments.
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

/**
 * \brief Iterates in parallel over all elements of **all** segments based on a condition.
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
 * \param condition The lambda function that determines whether a segment should be processed.
 * \param function The lambda function to be applied to each element in the selected segments.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e condition should be declared as:
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool {...}
 * ```
 * where \e segmentIdx is the index of the segment.
 *
 * The lambda function \e function should be declared as either:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx ) {...}
 * ```
 * or
 *
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType globalIdx ) {...}
 * ```
 *
 * where:
 * - \e segmentIdx is the index of the segment to which the given element belongs.
 * - \e localIdx is the rank of the element within the segment.
 * - \e globalIdx is the global index of the element within the range of all elements managed by the segments.
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

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIfSparse( const Segments& segments,
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

/**
 * \brief Iterates in parallel over segments within the specified range of segment indexes
 * and applies the given lambda function to each segment.
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
 * \param function The lambda function to be applied to each segment.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
 * ```
 *
 * where \e segment represents the given segment (see \ref TNL::Algorithms::Segments::SegmentView).
 * Its type is defined by \e SegmentViewType.
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
 * \brief Iterates in parallel over **all** segments and applies the given lambda function to each segment.
 *
 * \tparam Segments The type of the segments.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param function The lambda function to be applied to each segment.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
 * ```
 *
 * where \e segment represents the given segment (see \ref TNL::Algorithms::Segments::SegmentView).
 * Its type is defined by \e SegmentViewType.
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
 * \brief Iterates in parallel over segments with the given indexes and applies the specified
 * lambda function to each segment.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam IndexBegin The type of the index defining the beginning of the interval [ \e begin, \e end )
 *    of segments on which the lambda function will be applied.
 * \tparam IndexEnd The type of the index defining the end of the interval [ \e begin, \e end )
 *    of segments on which the lambda function will be applied.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param begin The beginning of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed using the lambda function.
 * \param end The end of the interval [ \e begin, \e end ) of segment indexes
 *    whose corresponding segments will be processed using the lambda function.
 * \param function The lambda function to be applied to each segment.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
 * ```
 *
 * where \e segment represents the given segment (see \ref TNL::Algorithms::Segments::SegmentView).
 * Its type is defined by \e SegmentViewType.
 *
 * \par Example
 * \include Algorithms/Segments/SegmentsExample_forSegmentsWithIndexes.cpp
 * \par Output
 * \include SegmentsExample_forSegmentsWithIndexes.out
 */
template< typename Segments,
          typename Array,
          typename IndexBegin,
          typename IndexEnd,
          typename Function,
          typename T = std::enable_if_t< IsArrayType< Array >::value
                                         && std::is_integral_v< IndexBegin > && std::is_integral_v< IndexEnd > > >
void
forSegments( const Segments& segments,
             const Array& segmentIndexes,
             IndexBegin begin,
             IndexEnd end,
             Function&& function,
             LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Iterates in parallel over segments with the given indexes and applies the specified
 * lambda function to each segment.
 *
 * \tparam Segments The type of the segments.
 * \tparam Array The type of the array containing the indexes of the segments to iterate over.
 *   This can be containers such as \ref TNL::Containers::Array, \ref TNL::Containers::ArrayView,
 *   \ref TNL::Containers::Vector, or \ref TNL::Containers::VectorView.
 * \tparam Function The type of the lambda function to be executed on each segment.
 *
 * \param segments The segments on which the lambda function will be applied.
 * \param segmentIndexes The array containing the indexes of the segments to iterate over.
 * \param function The lambda function to be applied to each segment.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e function should be declared as:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
 * ```
 *
 * where \e segment represents the given segment (see \ref TNL::Algorithms::Segments::SegmentView).
 * Its type is defined by \e SegmentViewType.
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
 * \brief Iterates in parallel over segments within the given range of segment indexes, applying a condition
 * to determine whether each segment should be processed.
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
 * \param segmentCondition The condition lambda function.
 * \param function The lambda function to be applied to each segment.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e segmentCondition should be declared as:
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool {...}
 * ```
 * where \e segmentIdx is the index of the segment.
 *
 * The lambda function \e function should be declared as:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
 * ```
 *
 * where \e segment represents the given segment (see \ref TNL::Algorithms::Segments::SegmentView).
 * Its type is defined by \e SegmentViewType.
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

/**
 * \brief Iterates in parallel over **all** segments, applying a condition
 * to determine whether each segment should be processed.
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
 * \param segmentCondition The condition lambda function.
 * \param function The lambda function to be applied to each segment.
 * \param launchConfig The configuration of the launch - see \ref TNL::Algorithms::Segments::LaunchConfiguration.
 *
 * The lambda function \e segmentCondition should be declared as:
 * ```
 * auto f = [=] __cuda_callable__ ( IndexType segmentIdx ) -> bool {...}
 * ```
 * where \e segmentIdx is the index of the segment.
 *
 * The lambda function \e function should be declared as:
 *
 * ```
 * auto f = [=] __cuda_callable__ ( const SegmentView& segment ) {...}
 * ```
 *
 * where \e segment represents the given segment (see \ref TNL::Algorithms::Segments::SegmentView).
 * Its type is defined by \e SegmentViewType.
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
 * This function is just a sequential variant of \ref TNL::Algorithms::Segments::forAllSegments.
 */
template< typename Segments, typename Function >
void
sequentialForAllSegments( const Segments& segments, Function&& function );

}  // namespace TNL::Algorithms::Segments

#include "traverse.hpp"
