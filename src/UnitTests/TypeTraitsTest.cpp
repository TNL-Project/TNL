#include <array>

#include <TNL/TypeTraits.h>
#include <TNL/Arithmetics/Complex.h>
#include <TNL/Containers/StaticVector.h>
#include <TNL/Containers/Vector.h>

#include <gtest/gtest.h>

using namespace TNL;
using namespace TNL::Containers;
using namespace TNL::Arithmetics;

TEST( TypeTraitsTest, GetValueType )
{
   static_assert( std::is_same_v< GetValueType_t< float >, float > );
   static_assert( std::is_same_v< GetValueType_t< double >, double > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, double > >, double > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, float > >, float > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, Complex< double > > >, double > );
   static_assert( std::is_same_v< GetValueType_t< StaticVector< 1, Complex< float > > >, float > );
   static_assert( std::is_same_v< GetValueType_t< Vector< double > >, double > );
   static_assert( std::is_same_v< GetValueType_t< Vector< float > >, float > );
   static_assert( std::is_same_v< GetValueType_t< Vector< Complex< double > > >, double > );
   static_assert( std::is_same_v< GetValueType_t< Vector< Complex< float > > >, float > );
}

namespace {

// copy constructor which is not equivalent to copying the object representation
struct UserCopyable
{
   UserCopyable() = default;
   // NOLINTNEXTLINE(modernize-use-equals-default)
   UserCopyable( const UserCopyable& other )
   : value( other.value )
   {}
   UserCopyable&
   operator=( const UserCopyable& other ) = default;

   int value = 0;
};

struct TriviallyCopyable
{
   int a;
   double b;
};

}  // namespace

TEST( TypeTraitsTest, is_nested_trivially_copyable )
{
   // trivially copyable types
   static_assert( is_nested_trivially_copyable_v< char > );
   static_assert( is_nested_trivially_copyable_v< double > );
   static_assert( is_nested_trivially_copyable_v< Complex< double > > );
   static_assert( is_nested_trivially_copyable_v< TriviallyCopyable > );
   static_assert( is_nested_trivially_copyable_v< std::array< double, 3 > > );
   static_assert( is_nested_trivially_copyable_v< std::array< TriviallyCopyable, 3 > > );

   // static arrays of such types (not trivially copyable)
   static_assert( ! std::is_trivially_copyable_v< StaticVector< 2, double > > );
   static_assert( is_nested_trivially_copyable_v< StaticArray< 3, int > > );
   static_assert( is_nested_trivially_copyable_v< StaticVector< 2, double > > );
   static_assert( is_nested_trivially_copyable_v< StaticVector< 2, Complex< float > > > );
   static_assert( is_nested_trivially_copyable_v< StaticVector< 2, StaticVector< 3, double > > > );
   static_assert( ! std::is_trivially_copyable_v< std::array< StaticVector< 2, double >, 3 > > );
   static_assert( is_nested_trivially_copyable_v< std::array< StaticVector< 2, double >, 3 > > );
   static_assert( is_nested_trivially_copyable_v< StaticArray< 2, std::array< StaticVector< 2, double >, 3 > > > );

   // types with a user-provided copy constructor
   static_assert( ! is_nested_trivially_copyable_v< UserCopyable > );
   static_assert( ! is_nested_trivially_copyable_v< std::array< UserCopyable, 3 > > );
   static_assert( ! is_nested_trivially_copyable_v< StaticArray< 3, UserCopyable > > );
   static_assert( ! is_nested_trivially_copyable_v< StaticVector< 2, StaticArray< 3, UserCopyable > > > );

   // dynamic arrays
   static_assert( ! is_nested_trivially_copyable_v< Vector< double > > );
}

#include "main.h"
