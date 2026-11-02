// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "LaunchConfiguration.h"

namespace TNL::Algorithms::Segments {

/**
 * \brief Performs `Size` independent reductions within each segment at once.
 * \anchor Segments_multiReduceAllSegments_identity
 *
 * For every segment and every `j = 0, ..., Size - 1`, the values returned by `fetch` for the
 * elements of the segment and the index `j` are reduced by `reduction` and the result is passed
 * to `storer`. All `Size` reductions of a segment are performed in one pass over its elements,
 * so the data shared by the reductions (e.g. the column indexes and values of a sparse matrix
 * when multiplying it with several vectors) are read only once.
 *
 * The reductions are computed by the same kernels as \ref Segments_reduceAllSegments, they work
 * with all segment types and launch configurations. The partial results of all `Size`
 * reductions are kept in registers, so `Size` should be small (up to 8 or 16 depending on
 * the value type).
 *
 * The `fetch` lambda has either the full form
 *
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType segmentIdx, IndexType localIdx, IndexType globalIdx, int j ) -> Value { ... }
 * ```
 *
 * or the brief form
 *
 * ```cpp
 * auto fetch = [=] __cuda_callable__ ( IndexType globalIdx, int j ) -> Value { ... }
 * ```
 *
 * where `j` is the index of the reduction. The `storer` lambda has the form
 *
 * ```cpp
 * auto storer = [=] __cuda_callable__ ( IndexType segmentIdx, int j, const Value& value ) { ... }
 * ```
 *
 * The `reduction` is a lambda or a function object with the form described in
 * \ref SegmentReductionLambda_Basic and it is used for all `Size` reductions.
 *
 * \tparam Size The number of reductions performed in each segment.
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing the results.
 *
 * \param segments The segment data structure on which the reductions will be performed.
 * \param fetch Lambda function for fetching data.
 * \param reduction Lambda function or function object for the reduction operation.
 * \param storer Lambda function for storing the results.
 * \param identity The identity element of the reduction operation.
 * \param launchConfig The configuration of the parallel execution.
 */
template< int Size, typename Segments, typename Fetch, typename Reduction, typename ResultStorer, typename Value >
static void
multiReduceAllSegments(
   const Segments& segments,
   Fetch&& fetch,
   Reduction&& reduction,
   ResultStorer&& storer,
   const Value& identity,
   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

/**
 * \brief Performs `Size` independent reductions within each segment at once.
 * \anchor Segments_multiReduceAllSegments
 *
 * The same as \ref Segments_multiReduceAllSegments_identity, but the identity is obtained
 * from the static member function template `getIdentity` of the `Reduction` function object
 * (see \ref SegmentReductionFunctionObjects).
 *
 * \tparam Size The number of reductions performed in each segment.
 * \tparam Segments The type of the segments.
 * \tparam Fetch The type of the lambda function used for data fetching.
 * \tparam Reduction The type of the function object defining the reduction operation.
 * \tparam ResultStorer The type of the lambda function used for storing the results.
 *
 * \param segments The segment data structure on which the reductions will be performed.
 * \param fetch Lambda function for fetching data.
 * \param reduction Function object for the reduction operation.
 * \param storer Lambda function for storing the results.
 * \param launchConfig The configuration of the parallel execution.
 */
template< int Size, typename Segments, typename Fetch, typename Reduction, typename ResultStorer >
static void
multiReduceAllSegments(
   const Segments& segments,
   Fetch&& fetch,
   Reduction&& reduction,
   ResultStorer&& storer,
   LaunchConfiguration launchConfig = Algorithms::Segments::LaunchConfiguration() );

}  // namespace TNL::Algorithms::Segments

#include "multiReduce.hpp"
