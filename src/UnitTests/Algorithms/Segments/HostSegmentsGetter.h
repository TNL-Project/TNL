#pragma once

#include <TNL/Algorithms/Segments/SortedSegments.h>
#include <TNL/Algorithms/Segments/TypeTraits.h>
#include <TNL/Devices/Host.h>

// Type of the segments of the same format on the host, used to verify the results on the host
// without reading the segments on the device element by element.
template< typename Segments, bool isSorted = TNL::Algorithms::Segments::isSortedSegments_v< Segments > >
struct HostSegmentsGetter
{
   using type = typename Segments::template Self< TNL::Devices::Host >;
};

template< typename Segments >
struct HostSegmentsGetter< Segments, true >
{
   using type =
      TNL::Algorithms::Segments::SortedSegments< typename Segments::EmbeddedSegmentsType::template Self< TNL::Devices::Host > >;
};
