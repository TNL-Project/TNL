#pragma once

#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

/**
 * \brief Generates one random value of the given type.
 */
template< typename Value >
Value
randomValue( std::mt19937& rng )
{
   if constexpr( std::is_floating_point_v< Value > ) {
      std::uniform_real_distribution< Value > dist( -1000.0, 1000.0 );
      return dist( rng );
   }
   else if constexpr( sizeof( Value ) == 1 ) {
      const int value = static_cast< int >( rng() % 256 ) - ( std::is_signed_v< Value > ? 128 : 0 );
      return static_cast< Value >( value );
   }
   else if constexpr( std::is_signed_v< Value > ) {
      using Distribution = std::conditional_t<
         sizeof( Value ) == 4,
         std::uniform_int_distribution< int >,
         std::uniform_int_distribution< std::int64_t > >;
      using Result = typename Distribution::result_type;
      Distribution dist( std::numeric_limits< Result >::min() / 2, std::numeric_limits< Result >::max() / 2 );
      return static_cast< Value >( dist( rng ) );
   }
   else {
      using Distribution = std::conditional_t<
         sizeof( Value ) == 4,
         std::uniform_int_distribution< unsigned int >,
         std::uniform_int_distribution< std::uint64_t > >;
      using Result = typename Distribution::result_type;
      Distribution dist( 0, std::numeric_limits< Result >::max() / 2 );
      return static_cast< Value >( dist( rng ) );
   }
}

/**
 * \brief Generates a vector of random values of the given type.
 */
template< typename Value >
std::vector< Value >
makeRandomVector( std::size_t size, std::uint32_t seed )
{
   std::vector< Value > vec( size );
   std::mt19937 rng( seed );
   for( auto& value : vec )
      value = randomValue< Value >( rng );
   return vec;
}
