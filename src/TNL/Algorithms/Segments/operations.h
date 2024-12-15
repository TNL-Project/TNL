// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include "LaunchConfig.h"

namespace TNL::Algorithms::Segments {

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments&,
             IndexBegin begin,
             IndexEnd end,
             const LaunchConfig< Segments >& launchConfig,
             Function&& function );

template< typename Segments, typename Index, typename Function >
void
forElements( const Segments&, Index begin, Index end, Function&& function );

template< typename Segments, typename Function >
void
forAllElements( const Segments&, const LaunchConfig< Segments >& launchConfig, Function&& function );

template< typename Segments, typename Function >
void
forAllElements( const Segments&, Function&& function );

template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments&,
             const Array& segmentIndexes,
             IndexBegin begin,
             IndexEnd end,
             const LaunchConfig< Segments >& launchConfig,
             Function function );

template< typename Segments, typename Array, typename IndexBegin, typename IndexEnd, typename Function >
void
forElements( const Segments&, const Array& segmentIndexes, IndexBegin begin, IndexEnd end, Function function );

template< typename Segments, typename Array, typename Function >
void
forElements( const Segments&, const Array& segmentIndexes, const LaunchConfig< Segments >& launchConfig, Function function );

template< typename Segments, typename Array, typename Function >
void
forElements( const Segments&, const Array& segmentIndexes, Function function );

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Segments&,
               IndexBegin begin,
               IndexEnd end,
               const LaunchConfig< Segments >& launchConfig,
               Condition condition,
               Function function );

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Condition, typename Function >
void
forElementsIf( const Segments&, IndexBegin begin, IndexEnd end, Condition condition, Function function );

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIf( const Segments&, const LaunchConfig< Segments >& launchConfig, Condition condition, Function function );

template< typename Segments, typename Condition, typename Function >
void
forAllElementsIf( const Segments&, Condition condition, Function function );

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forSegments( const Segments&,
             IndexBegin begin,
             IndexEnd end,
             const LaunchConfig< Segments >& launchConfig,
             Function&& function );

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
forSegments( const Segments&, IndexBegin begin, IndexEnd end, Function&& function );

template< typename Segments, typename Function >
void
forAllSegments( const Segments&, const LaunchConfig< Segments >& launchConfig, Function&& function );

template< typename Segments, typename Function >
void
forAllSegments( const Segments&, Function&& function );

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
sequentialForSegments( const Segments&,
                       IndexBegin begin,
                       IndexEnd end,
                       const LaunchConfig< Segments >& launchConfig,
                       Function&& function );

template< typename Segments, typename IndexBegin, typename IndexEnd, typename Function >
void
sequentialForSegments( const Segments&, IndexBegin begin, IndexEnd end, Function&& function );

template< typename Segments, typename Function >
void
sequentialForAllSegments( const Segments&, const LaunchConfig< Segments >& launchConfig, Function&& function );

template< typename Segments, typename Function >
void
sequentialForAllSegments( const Segments&, Function&& function );

}  // namespace TNL::Algorithms::Segments

#include "operations.hpp"
