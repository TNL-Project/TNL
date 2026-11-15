#pragma once

#include <functional>
#include <type_traits>

#ifdef HAVE_SORTING_REFERENCES
   #include <ips2ra.hpp>
   #include <ips4o.hpp>
#endif
#include <TNL/Algorithms/Sorting/detail/radixSort.h>
#include <TNL/Devices/Host.h>
#include <TNL/Devices/Sequential.h>

namespace TNL {

/**
 * In-place Parallel Super Scalar Radix Sort (IPS²Ra).
 *
 * The array is sorted in parallel with \ref Devices::Host::getMaxThreadsCount
 * threads for \ref Devices::Host and sequentially for \ref Devices::Sequential.
 * IPS²Ra sorts unsigned integer keys, so the values are mapped to the same
 * order-preserving keys as in \ref TNL::Algorithms::Sorting::RadixSort.
 */
struct IPS2RaSort
{
   template< typename Array >
   static void
   sort( Array& array )
   {
#ifdef HAVE_SORTING_REFERENCES
      using ValueType = typename Array::ValueType;
      using DeviceType = typename Array::DeviceType;
      static_assert(
         std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential >,
         "IPS2RaSort requires a host device" );

      const auto extractor = []( const ValueType& value )
      {
         return Algorithms::Sorting::detail::valueToRadixSortKey( value );
      };
      auto* begin = array.getData();
      auto* end = array.getData() + array.getSize();
      if constexpr( std::is_same_v< DeviceType, Devices::Host > )
         ips2ra::parallel::sort( begin, end, extractor, Devices::Host::getMaxThreadsCount() );
      else
         ips2ra::sort( begin, end, extractor );
#endif
   }
};

/**
 * In-place Parallel Super Scalar Samplesort (IPS⁴o).
 *
 * The array is sorted in parallel with \ref Devices::Host::getMaxThreadsCount
 * threads for \ref Devices::Host and sequentially for \ref Devices::Sequential.
 */
struct IPS4oSort
{
   template< typename Array >
   static void
   sort( Array& array )
   {
#ifdef HAVE_SORTING_REFERENCES
      using DeviceType = typename Array::DeviceType;
      static_assert(
         std::is_same_v< DeviceType, Devices::Host > || std::is_same_v< DeviceType, Devices::Sequential >,
         "IPS4oSort requires a host device" );

      auto* begin = array.getData();
      auto* end = array.getData() + array.getSize();
      if constexpr( std::is_same_v< DeviceType, Devices::Host > )
         ips4o::parallel::sort( begin, end, std::less<>{}, Devices::Host::getMaxThreadsCount() );
      else
         ips4o::sort( begin, end );
#endif
   }
};

}  // namespace TNL
