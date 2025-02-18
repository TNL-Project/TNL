// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

#include <type_traits>
namespace TNL::Algorithms::Segments {

template< typename T >
struct HasGetSegmentCountMethod
{
private:
   template< typename U >
   static constexpr auto
   check( U* ) -> std::enable_if_t< std::is_integral_v< decltype( std::declval< U >().getSegmentsCount() ) >, std::true_type >;

   template< typename >
   static constexpr std::false_type
   check( ... );

   using type = decltype( check< std::decay_t< T > >( nullptr ) );

public:
   static constexpr bool value = type::value;
};

template< typename Segments >
constexpr bool isSegments_v = HasGetSegmentCountMethod< Segments >::value;

}  // namespace TNL::Algorithms::Segments
