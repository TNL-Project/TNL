// SPDX-FileComment: This file is part of TNL - Template Numerical Library (https://tnl-project.org/)
// SPDX-License-Identifier: MIT

#pragma once

// Support for splitting the instantiation of a type-parameterized test suite into several
// translation units that are linked into one test executable (see `tnl_split_test_sources`
// in cmake/testing.cmake). Each translation unit is compiled with TNL_TEST_PART and
// TNL_TEST_PARTS defined and instantiates only a contiguous part of the type list. The names
// of the instantiated tests are the same as without the split.

#include <cstddef>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>

namespace TNL::UnitTests {

template< typename TypeList, std::size_t Part, std::size_t Parts >
struct TypeListPart;

template< typename... Types, std::size_t Part, std::size_t Parts >
struct TypeListPart< ::testing::Types< Types... >, Part, Parts >
{
   static_assert( Part < Parts, "invalid part index" );
   static_assert( Parts <= sizeof...( Types ), "the type list has fewer types than the number of parts" );

   static constexpr std::size_t begin = Part * sizeof...( Types ) / Parts;
   static constexpr std::size_t end = ( Part + 1 ) * sizeof...( Types ) / Parts;

   template< std::size_t... Indices >
   static auto select( std::index_sequence< Indices... > )
      -> ::testing::Types< std::tuple_element_t< begin + Indices, std::tuple< Types... > >... >;

   using type = decltype( select( std::make_index_sequence< end - begin >{} ) );

   // generates the same test names as ::testing::internal::DefaultNameGenerator for the whole type list
   struct NameGenerator
   {
      template< typename T >
      static std::string
      GetName( int i )
      {
         return std::to_string( begin + i );
      }
   };
};

}  // namespace TNL::UnitTests

#if defined( TNL_TEST_PART ) && defined( TNL_TEST_PARTS )
   #define TNL_INSTANTIATE_TYPED_TEST_SUITE_P( Prefix, SuiteName, Types )                                       \
      using Prefix##_##SuiteName##_Part = TNL::UnitTests::TypeListPart< Types, TNL_TEST_PART, TNL_TEST_PARTS >; \
      using Prefix##_##SuiteName##_PartTypes = Prefix##_##SuiteName##_Part::type;                               \
      using Prefix##_##SuiteName##_PartNames = Prefix##_##SuiteName##_Part::NameGenerator;                      \
      INSTANTIATE_TYPED_TEST_SUITE_P( Prefix, SuiteName, Prefix##_##SuiteName##_PartTypes, Prefix##_##SuiteName##_PartNames )
#else
   #define TNL_INSTANTIATE_TYPED_TEST_SUITE_P( Prefix, SuiteName, Types ) \
      INSTANTIATE_TYPED_TEST_SUITE_P( Prefix, SuiteName, Types )
#endif
